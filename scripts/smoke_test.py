r"""実モデルをMCP越しに動かし、ACE→MOSS→ACEの切替と4ツールを確認する。

Windows: .venv-hybrid\Scripts\python scripts\smoke_test.py --out outputs\smoke
--serverで登録済みのMCP実行ファイルを指定できる。重みは事前に取得する。
ACE / MOSS構成専用。出力指定を省くと一時保存する。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import Client, StdioServerParameters
import numpy as np
import soundfile as sf


async def call(client: Client, name: str, args: dict) -> dict:
    r = await client.call_tool(name, args)
    text = "".join(getattr(c, "text", "") for c in r.content)
    print(f"\n{name} {json.dumps(args, ensure_ascii=False)}\n  → {text}")
    if r.is_error:
        sys.exit(f"FAILED: {name}")
    return json.loads(text)


def check_audio(result: dict, count: int, seconds: float, *, channels: int | None = None) -> str:
    """実ファイルの尺・健全性と、モデル情報のsidecar引継ぎを確認する。"""
    assert len(result["candidates"]) == count, result
    for candidate in result["candidates"]:
        audio, sr = sf.read(candidate["path"], dtype="float32", always_2d=True)
        assert abs(len(audio) / sr - seconds) < 0.1, (audio.shape, sr)
        assert np.isfinite(audio).all() and np.abs(audio).max() > 1e-5
        if channels is not None:
            assert sr == 48000 and audio.shape[1] == channels
        meta = json.loads(Path(candidate["metadata"]).read_text(encoding="utf-8"))
        assert meta["model"] == result["model"] and meta["sample_rate"] == sr
        assert meta["seed"] == candidate["seed"]
    return result["candidates"][0]["path"]


async def run(out_dir: Path, server_command: Path | None = None) -> None:
    params = StdioServerParameters(
        command=str(server_command) if server_command else sys.executable,
        args=[] if server_command else ["-m", "audio_studio_mcp"],
        env={**os.environ, "AUDIO_STUDIO_OUTPUT_DIR": str(out_dir)})
    async with Client(params, read_timeout_seconds=1800) as c:
        tools = sorted(t.name for t in (await c.list_tools()).tools)
        print("tools:", tools)
        assert tools == ["edit_audio", "generate_audio", "process_audio", "unload_models"], tools

        music = await call(c, "generate_audio", {
            "prompt": "Synthwave track with warm analog pads, punchy gated drums and a shimmering "
                      "arpeggiated lead, nostalgic and driving",
            "bpm": 100, "duration_seconds": 20, "variations": 2,
            "seed": 42, "title": "smoke_synthwave"})
        assert music["prompt_used"].endswith("BPM: 100.")
        hybrid = music["model"].startswith("ACE-Step")
        mpath = check_audio(music, 2, 20, channels=2 if hybrid else None)
        original_hash = hashlib.sha256(Path(mpath).read_bytes()).digest()
        assert [x["seed"] for x in music["candidates"]] == [42, 43]

        sfx = await call(c, "generate_audio", {
            "audio_type": "sfx", "duration_seconds": 3, "seed": 42,
            "sound_source": "old wooden door", "sound_action": "creaking open slowly, then a soft thud",
            "sound_material": "dry aged oak, rusty iron hinges", "sound_space": "echoing empty hallway, close mic",
            "title": "smoke_door", "original_idea": "an old wooden door creaking open"})
        check_audio(sfx, 1, 3, channels=1 if hybrid else None)
        if hybrid:
            assert sfx["model"] == "MOSS-SoundEffect v2" and sfx["included_model_load"]
            meta = json.loads(Path(sfx["candidates"][0]["metadata"]).read_text(encoding="utf-8"))
            assert (meta["steps"], meta["cfg"]) == (100, 4)
            refused = await c.call_tool("edit_audio", {"input_audio": sfx["candidates"][0]["path"],
                "audio_type": "music", "mode": "variation", "prompt": "soft pad"})
            assert refused.is_error and "MOSS" in str(refused.content)

        ogg = await call(c, "process_audio", {
            "input_audio": sfx["candidates"][0]["path"], "trim_silence": True, "mono": True,
            "normalize": "peak", "peak_ceiling_db": -1, "format": "ogg", "sample_rate": 48000})
        assert ogg["channels"] == 1 and ogg["sample_rate"] == 48000 and Path(ogg["path"]).is_file()

        loop = await call(c, "process_audio", {
            "input_audio": mpath, "normalize": "loudness", "target_lufs": -16,
            "loop": True, "loop_crossfade_ms": 200, "format": "wav", "sample_rate": 48000})
        assert abs(loop["output_lufs"] - -16) < 1.5, loop

        ext = await call(c, "edit_audio", {
            "input_audio": mpath, "mode": "continue", "duration_seconds": 30,
            "prompt": "Synthwave outro with fading pads", "seed": 1, "title": "smoke_synthwave_ext"})
        assert ext["seconds"] == 30
        check_audio(ext, 1, 30, channels=2 if hybrid else None)
        if hybrid:
            assert ext["included_model_load"] and ext["model"].startswith("ACE-Step")

        inpaint = await call(c, "edit_audio", {
            "input_audio": mpath, "mode": "inpaint", "start_seconds": 8, "end_seconds": 14,
            "prompt": "Synthwave breakdown with only pads and arpeggio, no drums", "seed": 2})
        check_audio(inpaint, 1, 20, channels=2 if hybrid else None)
        variation = await call(c, "edit_audio", {
            "input_audio": mpath, "mode": "variation", "strength": 0.5,
            "duration_seconds": 10,
            "prompt": "Lo-fi hip-hop version with dusty drums and mellow electric piano", "seed": 3})
        check_audio(variation, 1, 10, channels=2 if hybrid else None)
        assert hashlib.sha256(Path(mpath).read_bytes()).digest() == original_hash

        assert (await call(c, "unload_models", {}))["unloaded"] is True

    files = sorted(p.name for p in out_dir.iterdir())
    print(f"\nOK — {len(files)} files in {out_dir}")


def main() -> None:
    # Windows consoles and pipes may use a legacy code page; never crash on ✓ / →.
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, help="keep outputs in this folder")
    ap.add_argument("--server", type=Path, help="registered MCP executable (default: this Python module)")
    args = ap.parse_args()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        asyncio.run(run(args.out.resolve(), args.server.resolve() if args.server else None))
    else:
        with tempfile.TemporaryDirectory(prefix="audio-studio-smoke-") as td:
            asyncio.run(run(Path(td), args.server.resolve() if args.server else None))


if __name__ == "__main__":
    main()
