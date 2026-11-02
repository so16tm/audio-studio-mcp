"""用途別profileを使うMCP境界の契約を確認する。"""
import json
from pathlib import Path

import pytest
from mcp import Client

from audio_studio_mcp import server
from audio_studio_mcp.backends.ace_moss import AceMossBackend
from conftest import FakeBackend

pytestmark = pytest.mark.anyio


class RoutedFake(FakeBackend):
    sample_rate = 48000

    def profile(self, audio_type):
        # 推論自体は既存fixtureを使い、モデルの公開capabilityは実実装を使う。
        return AceMossBackend.profile(self, audio_type)


@pytest.fixture
def routed(tmp_path, monkeypatch):
    backend = RoutedFake()
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path / "out")
    server.set_backend(backend)
    yield backend
    server.set_backend(None)


async def call(name, arguments):
    async with Client(server.server) as client:
        response = await client.call_tool(name, arguments)
    text = "".join(getattr(item, "text", "") for item in response.content)
    if response.is_error:
        raise AssertionError(text)
    return json.loads(text)


async def test_sfx_defaults_prompt_and_provenance(routed):
    result = await call("generate_audio", {"audio_type": "sfx", "sound_source": "door",
        "sound_action": "creaking", "duration_seconds": 1, "seed": 7})
    request = routed.requests[-1][1]
    assert (request.audio_type, request.steps, request.cfg) == ("sfx", 100, 4)
    assert request.prompt == "door, creaking."
    assert result["model"] == "MOSS-SoundEffect v2"
    meta = json.loads(Path(result["candidates"][0]["metadata"]).read_text(encoding="utf-8"))
    assert meta["backend"] == "moss-soundeffect" and meta["sample_rate"] == 48000
    assert meta["steps"] == 100 and meta["cfg"] == 4 and meta["model_revision"]


async def test_music_defaults_and_bpm_reach_the_backend(routed):
    result = await call("generate_audio", {"prompt": "warm synthwave", "bpm": 100,
        "duration_seconds": 20, "seed": 1})
    request = routed.requests[-1][1]
    assert (request.audio_type, request.steps, request.cfg, request.bpm) == ("music", 8, 1, 100)
    assert "TrackType:" not in request.prompt
    assert result["model"] == "ACE-Step 1.5 turbo"


async def test_sfx_uses_its_own_duration_cap(routed):
    result = await call("generate_audio", {"audio_type": "sfx", "prompt": "rain",
        "duration_seconds": 60, "seed": 1})
    assert result["seconds"] == 30 and routed.requests[-1][1].seconds == 30


@pytest.mark.parametrize("arguments, message", [
    ({"duration_seconds": 3}, "10"),
    ({"cfg": 3}, "cfg"),
    ({"negative_prompt": "voices"}, "negative_prompt"),
    ({"steps": 20}, "steps"),
])
async def test_unsupported_ace_parameters_fail_before_inference(routed, arguments, message):
    with pytest.raises(AssertionError, match=message):
        await call("generate_audio", {"prompt": "calm pads", **arguments})
    assert not routed.requests


async def test_sfx_edit_is_rejected_explicitly_and_via_source_metadata(routed):
    generated = await call("generate_audio", {"audio_type": "sfx", "prompt": "door",
        "duration_seconds": 10, "seed": 1})
    source = generated["candidates"][0]["path"]
    processed = await call("process_audio", {"input_audio": source, "sample_rate": 48000})
    count = len(routed.requests)
    for arguments in ({"audio_type": "sfx"}, {}, {"input_audio": processed["path"]}, {"audio_type": "music"}):
        with pytest.raises(AssertionError, match="MOSS"):
            await call("edit_audio", {"input_audio": source, "mode": "variation", **arguments})
    assert len(routed.requests) == count


async def test_mcp_schema_accepts_model_defaults_and_moss_steps(routed):
    async with Client(server.server) as client:
        tool = next(t for t in (await client.list_tools()).tools if t.name == "generate_audio")
    properties = tool.input_schema["properties"]
    assert properties["steps"]["default"] is None
    assert properties["cfg"]["default"] is None
    result = await call("generate_audio", {"audio_type": "sfx", "prompt": "rain",
        "duration_seconds": 1, "steps": 100, "seed": 1})
    assert result["model"] == "MOSS-SoundEffect v2"
