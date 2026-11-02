"""ACE-Step instrumental music and MOSS sound effects in one serialized runtime.

Only one model resides in GPU memory. Model imports are deferred until the
worker receives a generation or preload job, leaving startup and profiles cheap.
"""
from __future__ import annotations

import contextlib
import gc
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

from .base import Backend, BackendProfile, GenerationRequest, GenerationResult

# Also consumed by the installer; this module must not import model runtimes.
ACE_MODEL_ID = "ACE-Step/Ace-Step1.5"
ACE_MODEL_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
ACE_RUNTIME_REVISION = "ca1e85fe9430179831e6bc6be790c332190a3866"
MOSS_MODEL_ID = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
MOSS_MODEL_REVISION = "e35df4d82fbe87fcd5d14e5d100e349c0c3c076d"
MOSS_RUNTIME_REVISION = "934d6826b084c46a0d033402174d5f8ac4ed2519"


def default_upstream_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "vendor" / "ACE-Step-1.5"


@contextlib.contextmanager
def _edit_source(path: str, seconds: float) -> Iterator[str]:
    """ACE derives edit duration from its source, ignoring the duration argument."""
    import numpy as np
    import soundfile as sf

    info = sf.info(path)
    target_frames = round(seconds * info.samplerate)
    if info.frames == target_frames:
        yield path
        return
    audio, sample_rate = sf.read(path, frames=target_frames, dtype="float32", always_2d=True)
    if len(audio) < target_frames:
        audio = np.pad(audio, ((0, target_frames - len(audio)), (0, 0)))
    with tempfile.TemporaryDirectory(prefix="audio-studio-ace-") as temporary:
        source = Path(temporary) / "source.wav"
        sf.write(source, audio, sample_rate, subtype="FLOAT")
        yield str(source)


class AceMossBackend(Backend):
    name = "ace-moss"
    model_label = "ACE-Step 1.5 / MOSS-SoundEffect v2"
    sample_rate = 48000
    max_seconds = 600.0

    def __init__(self, upstream_dir: Path | None = None, device: str | None = None) -> None:
        super().__init__()
        self.upstream_dir = Path(upstream_dir or os.environ.get("AUDIO_STUDIO_ACE_DIR")
                                 or default_upstream_dir()).expanduser().resolve()
        self.device = device or os.environ.get("AUDIO_STUDIO_DEVICE")
        self._model = None
        self._audio_type = None
        self._torch = None
        self._ace_api = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ace-moss")

    @property
    def executor(self) -> Executor:
        return self._pool

    @property
    def active_audio_type(self) -> str | None:
        return self._audio_type

    def profile(self, audio_type: str) -> BackendProfile:
        if audio_type == "sfx":
            return BackendProfile(name="moss-soundeffect", model_label="MOSS-SoundEffect v2",
                sample_rate=48000, max_seconds=30, default_steps=100, default_cfg=4,
                max_steps=200, supports_edit=False, uses_sa3_prefix=False,
                model_revision=MOSS_MODEL_REVISION, runtime_revision=MOSS_RUNTIME_REVISION)
        return BackendProfile(name="ace-step", model_label="ACE-Step 1.5 turbo",
            sample_rate=48000, max_seconds=600, min_seconds=10, max_steps=8,
            uses_sa3_prefix=False, supports_cfg=False, supports_negative_prompt=False,
            model_revision=ACE_MODEL_REVISION, runtime_revision=ACE_RUNTIME_REVISION)

    def loaded(self) -> bool:
        return self._model is not None

    def _load(self, audio_type: str) -> bool:
        if self._model is not None and self._audio_type == audio_type:
            return False
        self._unload()
        # Official MOSS Windows guidance: disable Dynamo before Torch is imported.
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        with contextlib.redirect_stdout(sys.stderr):
            import torch

            self._torch = torch
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if audio_type == "sfx":
                from moss_soundeffect_v2 import MossSoundEffectPipeline

                self._model = MossSoundEffectPipeline.from_pretrained(
                    MOSS_MODEL_ID, revision=MOSS_MODEL_REVISION,
                    torch_dtype=torch.bfloat16, device=self.device)
            else:
                if not (self.upstream_dir / "acestep" / "handler.py").is_file():
                    raise RuntimeError(
                        f"ACE-Step runtime not found at {self.upstream_dir}. Run install-hybrid.ps1 "
                        "or set AUDIO_STUDIO_ACE_DIR to the ACE-Step-1.5 checkout.")
                if str(self.upstream_dir) not in sys.path:
                    sys.path.insert(0, str(self.upstream_dir))
                from huggingface_hub import snapshot_download
                from acestep.handler import AceStepHandler
                from acestep.inference import GenerationConfig, GenerationParams, generate_music

                # The loader requires the complete bundle, including the unused LM.
                os.environ["ACESTEP_CHECKPOINTS_DIR"] = snapshot_download(
                    repo_id=ACE_MODEL_ID, revision=ACE_MODEL_REVISION)
                self._ace_api = (GenerationParams, GenerationConfig, generate_music)
                self._model = AceStepHandler()
                try:
                    status, ok = self._model.initialize_service(
                        project_root=str(self.upstream_dir), config_path="acestep-v15-turbo",
                        device=self.device, use_flash_attention=False, compile_model=False,
                        offload_to_cpu=False, offload_dit_to_cpu=False, quantization=None,
                        use_mlx_dit=False)
                    if not ok:
                        raise RuntimeError(f"ACE-Step initialization failed: {status}")
                except Exception:
                    self._unload()
                    raise
            self._audio_type = audio_type
        return True

    def _unload(self) -> bool:
        if self._model is None:
            return False
        with contextlib.redirect_stdout(sys.stderr):
            self._model = None
            self._audio_type = None
            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        return True

    def _preload(self) -> None:
        self._load("music")

    def _generate(self, seeds: list[int], req: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        result = GenerationResult()
        audio_type = "sfx" if req.audio_type == "sfx" else "music"
        with contextlib.redirect_stdout(sys.stderr), contextlib.ExitStack() as stack:
            result.included_model_load = self._load(audio_type)
            source = req.inpaint_audio or req.init_audio
            if audio_type == "music" and source:
                source = stack.enter_context(_edit_source(source, req.seconds))
            for seed in seeds:
                if audio_type == "sfx":
                    tensor = self._model(prompt=req.prompt, seconds=req.seconds,
                        num_inference_steps=req.steps, cfg_scale=req.cfg,
                        seed=seed, negative_prompt=req.negative_prompt)[0]
                else:
                    Params, Config, generate_music = self._ace_api
                    kwargs = dict(task_type="text2music", caption=req.prompt,
                        duration=req.seconds, bpm=req.bpm, lyrics="[Instrumental]",
                        instrumental=True, inference_steps=req.steps, guidance_scale=1.0,
                        seed=seed, thinking=False, use_cot_caption=False, use_cot_metas=False,
                        use_cot_language=False, use_cot_lyrics=False, dcw_enabled=False,
                        enable_normalization=False)
                    if req.init_audio:
                        kwargs.update(task_type="cover", src_audio=source,
                                      audio_cover_strength=1.0 - req.init_strength)
                    if req.inpaint_audio:
                        start, end = req.inpaint_range
                        kwargs.update(task_type="repaint", src_audio=source,
                            repainting_start=start, repainting_end=end, chunk_mask_mode="explicit")
                    generated = generate_music(self._model, None, Params(**kwargs),
                        Config(batch_size=1, use_random_seed=False), save_dir=None)
                    if not generated.success:
                        raise RuntimeError(f"ACE-Step generation failed: {generated.error}")
                    tensor = generated.audios[0]["tensor"]
                result.audios.append(tensor.detach().float().cpu().numpy())
        result.wall_seconds = time.perf_counter() - started
        return result
