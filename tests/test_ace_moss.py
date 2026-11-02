"""Hybrid backend contracts using doubles only at the external runtime boundary."""
import asyncio
import gc
import os
import subprocess
import sys
import threading
import weakref
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audio_studio_mcp.backends.base import Backend, GenerationRequest


def test_legacy_profile_and_positional_request_are_compatible():
    class Legacy(Backend):
        name, model_label, sample_rate, max_seconds = "legacy", "existing", 44100, 120
        executor = None
        loaded = _generate = _unload = _preload = lambda *args: None

    profile = Legacy().profile("sfx")
    assert (profile.name, profile.model_label, profile.sample_rate, profile.max_seconds) == (
        "legacy", "existing", 44100, 120)
    assert (profile.default_steps, profile.default_cfg, profile.max_steps) == (8, 1, 50)
    assert profile.supports_edit and profile.uses_sa3_prefix
    with pytest.raises(FrozenInstanceError):
        profile.name = "changed"
    request = GenerationRequest("old positional", 12, "", 9, 2, None, .4, None, None)
    assert (request.audio_type, request.bpm) == ("music", None)


def test_profiles_and_creation_import_no_model_runtime():
    code = """
import sys
from audio_studio_mcp.backends import create_backend
backend = create_backend('ace-moss')
for kind in ('music', 'instrument', 'raw'):
    p = backend.profile(kind)
    assert (p.name, p.sample_rate, p.min_seconds, p.max_seconds) == ('ace-step', 48000, 10, 600)
    assert (p.default_steps, p.default_cfg, p.max_steps) == (8, 1, 8)
    assert p.supports_edit and not p.uses_sa3_prefix
    assert not p.supports_cfg and not p.supports_negative_prompt
    assert p.model_revision == '19671f406d603126926c1b7e2adc169acbcade22'
    assert p.runtime_revision == 'ca1e85fe9430179831e6bc6be790c332190a3866'
p = backend.profile('sfx')
assert (p.name, p.sample_rate, p.min_seconds, p.max_seconds) == ('moss-soundeffect', 48000, .5, 30)
assert (p.default_steps, p.default_cfg, p.max_steps) == (100, 4, 200)
assert not p.supports_edit and not p.uses_sa3_prefix
assert p.supports_cfg and p.supports_negative_prompt
assert p.model_revision == 'e35df4d82fbe87fcd5d14e5d100e349c0c3c076d'
assert p.runtime_revision == '934d6826b084c46a0d033402174d5f8ac4ed2519'
assert not {'torch', 'acestep', 'moss_soundeffect_v2', 'huggingface_hub'} & sys.modules.keys()
backend.executor.shutdown()
"""
    run = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout == ""


class Tensor:
    def __init__(self, channels, seed):
        self.array = np.full((channels, 16), seed / 100, dtype=np.float64)

    def detach(self):
        return self

    def float(self):
        self.array = self.array.astype(np.float32)
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    events, calls, models, threads, sources = [], [], [], [], []
    ace_dir = tmp_path / "ACE-Step-1.5"
    (ace_dir / "acestep").mkdir(parents=True)
    (ace_dir / "acestep" / "handler.py").touch()
    monkeypatch.setenv("AUDIO_STUDIO_ACE_DIR", str(ace_dir))
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    monkeypatch.delenv("AUDIO_STUDIO_DEVICE", raising=False)
    monkeypatch.delenv("ACESTEP_CHECKPOINTS_DIR", raising=False)
    # Production adds the checkout to sys.path for its uninstalled official API.
    monkeypatch.syspath_prepend(str(ace_dir))

    def note(event):
        events.append(event)
        threads.append(threading.get_ident())

    class Ace:
        def __init__(self):
            assert all(ref() is None for ref in models), "previous model retained"
            models.append(weakref.ref(self))
            note("load:ace")

        def initialize_service(self, **kwargs):
            print("ACE initialize noisy stdout")
            calls.append(("ace-init", kwargs))
            assert os.environ["ACESTEP_CHECKPOINTS_DIR"] == str(tmp_path / "snapshot")
            return "ready", True

        def __del__(self):
            note("release:ace")

    class Moss:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            assert all(ref() is None for ref in models), "previous model retained"
            assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
            print("MOSS load noisy stdout")
            calls.append(("moss-load", (model, kwargs)))
            pipe = cls()
            models.append(weakref.ref(pipe))
            note("load:moss")
            return pipe

        def __call__(self, **kwargs):
            print("MOSS inference noisy stdout")
            note("generate:moss")
            calls.append(("moss", kwargs))
            return [Tensor(1, kwargs["seed"])]

        def __del__(self):
            note("release:moss")

    def generate_music(handler, llm, params, config, save_dir=None):
        assert isinstance(handler, Ace) and llm is None
        assert save_dir is None, "server owns saving"
        print("ACE inference noisy stdout")
        note("generate:ace")
        calls.append(("ace", (vars(params), vars(config))))
        if getattr(params, "src_audio", None):
            audio, sr = sf.read(params.src_audio, dtype="float32", always_2d=True)
            sources.append((Path(params.src_audio), audio, sr))
        return SimpleNamespace(success=True, error=None,
            audios=[{"tensor": Tensor(2, params.seed), "sample_rate": 48000, "path": ""}])

    def snapshot_download(**kwargs):
        calls.append(("snapshot", kwargs))
        print("download noisy stdout")
        return str(tmp_path / "snapshot")

    modules = {
        "torch": dict(cuda=SimpleNamespace(is_available=lambda: True,
                     empty_cache=lambda: note("empty-cache")), bfloat16="bf16"),
        "huggingface_hub": dict(snapshot_download=snapshot_download),
        "acestep": {}, "acestep.handler": dict(AceStepHandler=Ace),
        "acestep.inference": dict(GenerationParams=SimpleNamespace,
            GenerationConfig=SimpleNamespace, generate_music=generate_music),
        "moss_soundeffect_v2": dict(MossSoundEffectPipeline=Moss),
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    from audio_studio_mcp.backends.ace_moss import AceMossBackend
    backend = AceMossBackend()
    yield SimpleNamespace(backend=backend, events=events, calls=calls, threads=threads,
                          models=models, ace_dir=ace_dir, ace=Ace, sources=sources)
    asyncio.run(backend.unload())
    backend.executor.shutdown()
    gc.collect()


def test_seeds_audio_and_same_model_reuse(runtime, capsys):
    backend = runtime.backend
    assert backend.active_audio_type is None
    first = asyncio.run(backend.generate([11, 23], GenerationRequest("music", 20, bpm=112)))
    second = asyncio.run(backend.generate([31], GenerationRequest("again", 20)))
    assert first.included_model_load and not second.included_model_load
    assert backend.active_audio_type == "music"
    assert first.wall_seconds > 0 and second.wall_seconds > 0
    assert len(first.audios) == 2
    assert all(a.shape == (2, 16) and a.dtype == np.float32 for a in first.audios)
    np.testing.assert_allclose(first.audios[0], .11)
    assert runtime.events == ["load:ace", "generate:ace", "generate:ace", "generate:ace"]
    generated = [c[1] for c in runtime.calls if c[0] == "ace"]
    assert [p[0]["seed"] for p in generated] == [11, 23, 31]
    p, config = generated[0]
    assert (p["caption"], p["bpm"], p["duration"], p["inference_steps"]) == ("music", 112, 20, 8)
    assert p["lyrics"] == "[Instrumental]" and p["instrumental"]
    assert not any(p[k] for k in ("thinking", "use_cot_caption", "use_cot_metas", "use_cot_language", "dcw_enabled", "enable_normalization"))
    assert p["guidance_scale"] == 1
    assert config["batch_size"] == 1 and not config["use_random_seed"]
    assert capsys.readouterr().out == ""


def test_switch_releases_previous_model_before_next_load(runtime, capsys):
    backend = runtime.backend
    async def exercise():
        results = await asyncio.gather(
            backend.generate([1], GenerationRequest("bgm", 20)),
            backend.generate([2, 3], GenerationRequest("door", 3, "speech", 100, 4, audio_type="sfx")),
            backend.generate([4], GenerationRequest("piano", 20, audio_type="instrument")))
        assert await backend.unload()
        assert not await backend.unload()
        return results[1]
    result = asyncio.run(exercise())
    assert runtime.events == ["load:ace", "generate:ace", "release:ace", "empty-cache",
        "load:moss", "generate:moss", "generate:moss", "release:moss", "empty-cache",
        "load:ace", "generate:ace", "release:ace", "empty-cache"]
    assert len(set(runtime.threads)) == 1 and runtime.threads[0] != threading.get_ident()
    assert result.included_model_load
    assert all(a.shape == (1, 16) and a.dtype == np.float32 for a in result.audios)
    calls = [c[1] for c in runtime.calls if c[0] == "moss"]
    assert [c["seed"] for c in calls] == [2, 3]
    assert calls[0] == dict(prompt="door", seconds=3, num_inference_steps=100,
                           cfg_scale=4, seed=2, negative_prompt="speech")
    assert not backend.loaded()
    assert backend.active_audio_type is None
    assert capsys.readouterr().out == ""


def test_initialization_exception_releases_partial_model(runtime, monkeypatch):
    def failed_init(self, **kwargs):
        raise RuntimeError("initialization error")

    monkeypatch.setattr(runtime.ace, "initialize_service", failed_init)
    with pytest.raises(RuntimeError, match="initialization error"):
        asyncio.run(runtime.backend.generate([1], GenerationRequest("music", 20)))
    assert not runtime.backend.loaded()
    assert runtime.backend.active_audio_type is None
    gc.collect()  # The propagated exception's traceback can retain the failed handler.
    assert all(ref() is None for ref in runtime.models)
    assert "empty-cache" in runtime.events


def test_model_downloads_are_revision_pinned_and_ace_is_uncompiled(runtime):
    asyncio.run(runtime.backend.generate([7], GenerationRequest("music", 20)))
    asyncio.run(runtime.backend.generate([8], GenerationRequest("rain", 5, steps=100, cfg=4, audio_type="sfx")))
    assert ("snapshot", dict(repo_id="ACE-Step/Ace-Step1.5",
        revision="19671f406d603126926c1b7e2adc169acbcade22")) in runtime.calls
    assert ("ace-init", dict(project_root=str(runtime.ace_dir), config_path="acestep-v15-turbo",
        device="cuda", use_flash_attention=False, compile_model=False, offload_to_cpu=False,
        offload_dit_to_cpu=False, quantization=None, use_mlx_dit=False)) in runtime.calls
    assert ("moss-load", ("OpenMOSS-Team/MOSS-SoundEffect-v2.0", dict(
        revision="e35df4d82fbe87fcd5d14e5d100e349c0c3c076d", torch_dtype="bf16", device="cuda"))) in runtime.calls


@pytest.mark.parametrize("req, task, expected", [
    (GenerationRequest("variation", 20, init_audio="source.wav", init_strength=.8),
     "cover", {"audio_cover_strength": pytest.approx(.2)}),
    (GenerationRequest("inpaint", 20, inpaint_audio="source.wav", inpaint_range=(8, 14)),
     "repaint", {"repainting_start": 8, "repainting_end": 14, "chunk_mask_mode": "explicit"}),
    (GenerationRequest("continue", 30, inpaint_audio="source.wav", inpaint_range=(18, 30)),
     "repaint", {"repainting_start": 18, "repainting_end": 30, "chunk_mask_mode": "explicit"}),
])
def test_ace_edit_mapping(runtime, tmp_path, req, task, expected):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((int(req.seconds * 100), 2), np.float32), 100, subtype="FLOAT")
    req = replace(req, init_audio=str(source) if req.init_audio else None,
                  inpaint_audio=str(source) if req.inpaint_audio else None)
    asyncio.run(runtime.backend.generate([42], req))
    params = next(c[1][0] for c in runtime.calls if c[0] == "ace")
    assert params["task_type"] == task
    assert params["src_audio"] == str(source)
    assert {key: params[key] for key in expected} == expected


@pytest.mark.parametrize("seconds, edit", [(10, "variation"), (30, "variation"), (30, "repaint")])
def test_ace_edit_source_matches_target_duration_without_changing_original(runtime, tmp_path, seconds, edit):
    source = tmp_path / "original.wav"
    original_audio = np.linspace(-.6, .6, 4000, dtype=np.float32).reshape(2000, 2)
    sf.write(source, original_audio, 100, subtype="FLOAT")
    original_bytes = source.read_bytes()
    req = GenerationRequest("edit", seconds, init_audio=str(source)) if edit == "variation" else (
        GenerationRequest("extend", seconds, inpaint_audio=str(source), inpaint_range=(18, seconds)))
    asyncio.run(runtime.backend.generate([42, 43], req))
    assert len(runtime.sources) == 2
    used_path, audio, sr = runtime.sources[0]
    assert (sr, audio.shape) == (100, (seconds * 100, 2))
    assert used_path != source
    assert runtime.sources[1][0] == used_path
    retained_frames = min(2000, seconds * 100)
    np.testing.assert_array_equal(audio[:retained_frames], original_audio[:retained_frames])
    assert not np.any(audio[retained_frames:])
    assert source.read_bytes() == original_bytes
    assert not used_path.exists() and not used_path.parent.exists()


def test_ace_edit_temporary_source_is_removed_on_inference_error(runtime, tmp_path, monkeypatch):
    source = tmp_path / "original.wav"
    sf.write(source, np.ones((2000, 1), np.float32) * .1, 100, subtype="FLOAT")
    used_sources = []

    def failed_inference(handler, llm, params, config, save_dir=None):
        used_sources.append(Path(params.src_audio))
        assert sf.info(params.src_audio).frames == 1000
        raise RuntimeError("inference error")

    monkeypatch.setattr(sys.modules["acestep.inference"], "generate_music", failed_inference)
    with pytest.raises(RuntimeError, match="inference error"):
        asyncio.run(runtime.backend.generate([42], GenerationRequest("shorter", 10, init_audio=str(source))))
    assert used_sources and used_sources[0] != source
    assert not used_sources[0].exists() and not used_sources[0].parent.exists()
    assert sf.info(source).frames == 2000
