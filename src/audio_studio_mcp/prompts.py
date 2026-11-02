"""用途別のプロンプトと、MCPクライアントへ渡す生成ガイド。

明示的なStable Audioバックエンドでは、従来のTrackType形式も維持する。
"""
from __future__ import annotations

import re

TRACK_TYPE_PREFIXES = {
    "music": "TrackType: Music, VocalType: Instrumental, ",
    "instrument": "TrackType: Instrument, ",
    "sfx": "TrackType: SFX, ",
    "raw": "",
}
DEFAULT_SECONDS = {"music": 60.0, "instrument": 12.0, "sfx": 5.0, "raw": 30.0}

_LENGTH_RE = re.compile(r"\s*\.?\s*Length:\s*(\d+(?:\.\d+)?)\s*seconds?\.?\s*$", re.I)
_BPM_RE = re.compile(r"\bBPM\b", re.I)
_PREFIX_RE = re.compile(r"^TrackType:[^,]*,\s*(VocalType:[^,]*,\s*)?")


def prepare_prompt(prompt: str, audio_type: str, seconds: float | None,
                   bpm: int | None = None, sfx_fields: tuple[str, ...] = (), *, add_prefix: bool = True
                   ) -> tuple[str, float | None]:
    """Build the prompt the model sees:
      prompt [, source, action, material, space].  [BPM: N.]
    with the official TrackType prefix, and pull a trailing 'Length: N seconds'
    (the prompt assistant's format) out of the text into the duration."""
    text = prompt.strip()
    m = _LENGTH_RE.search(text)
    if m:
        text = text[:m.start()]
        if seconds is None:
            seconds = float(m.group(1))
    parts = [p.strip().rstrip(" .,") for p in (text, *sfx_fields) if p and p.strip()]
    if not parts:
        raise ValueError("empty prompt: give prompt and/or the sound_* fields")
    text = ", ".join(parts) + "."
    if bpm and not _BPM_RE.search(text):
        text += f" BPM: {bpm}."
    if not add_prefix:
        text = strip_prefix(text)
    elif audio_type != "raw" and not text.lower().startswith("tracktype:"):
        text = TRACK_TYPE_PREFIXES[audio_type] + text
    return text, seconds


def strip_prefix(prompt: str) -> str:
    """Drop the TrackType/VocalType prefix (used for auto file names)."""
    return _PREFIX_RE.sub("", prompt)


INSTRUCTIONS = """\
Local Audio Studio for instrumental BGM and sound effects, available to any MCP client.
On the Windows hybrid installation, music/instrument/raw use ACE-Step 1.5 turbo;
sfx uses MOSS-SoundEffect v2. Models run one at a time and are unloaded on switching.

- generate_audio writes 48 kHz 24-bit WAV: stereo for ACE, mono for MOSS.
  Each candidate has a .json sidecar containing prompt, seed, model and settings.
- Use music for BGM, instrument for a solo instrument/stem/loop, sfx for effects
  or ambience. raw selects ACE with a verbatim natural-language description.
- Write English prompts and preserve the user's request in original_idea.
  Music: genre, instruments, rhythm and mood; pass tempo as bpm. For SFX, fill
  sound_source, sound_action, sound_material and sound_space. The server joins them.
  Do not add TrackType or VocalType prefixes; the server prepares each model's prompt.
- Omit steps/cfg to use the selected model's defaults. ACE turbo uses 8 steps,
  cfg=1 (fixed), no negative_prompt, and 10–600 seconds. MOSS uses 100 steps,
  cfg=4 and 0.5–30 seconds; its solver generates a 30-second latent then crops it.
  MOSS supports English/Chinese descriptions; translate Japanese requests to English.
- edit_audio supports music variation/style transfer, inpaint and continue via ACE.
  If audio_type is omitted, source metadata is used. MOSS effects cannot be AI-edited;
  generate another effect or use process_audio for trimming, fades, loops and export.
- Use variations=1–4 for candidate selection. Seeds are seed, seed+1, ... .
  MOSS is memory intensive, so all candidates and GPU jobs run sequentially.
- Generated files get a safety peak ceiling, not mastering. Use process_audio for
  LUFS/peak normalization, seamless loops and 48 kHz WAV/OGG/FLAC game/video assets.
- Explicit sa3-mlx/sa3-torch are legacy backends with their original 44.1 kHz rate,
  step/cfg defaults, limits and editing support; the server adds their SA3 prefixes.
"""
