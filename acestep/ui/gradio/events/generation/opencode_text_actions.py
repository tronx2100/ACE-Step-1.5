"""Writes/rewrites song caption and lyrics via the external `opencode` CLI.

The bundled 5Hz LM is small and tuned for caption/metadata/audio-code
generation, not free creative writing - its lyrics in particular tend to
be incoherent ("Die 2000, 2000, 2000 / Der Mensch ist super schoen"). This
hands the Caption/Lyrics textbox content to a general-purpose coding-agent
CLI instead, which writes noticeably better text. Either field is
dual-purpose for this call: it can hold a short instruction ("ein
frischer Song ueber Blumen und Meer") or an existing draft to
rewrite/improve - either way it's passed straight through as the user's
request.
"""

import json
import os
import shutil
import subprocess

import gradio as gr

from acestep.ui.gradio.i18n import t

_OPENCODE_TIMEOUT_SECONDS = 180

# Common global-install locations for npm/bun/etc CLIs. The gradio server
# process doesn't always inherit an interactive shell's PATH (e.g. started
# from a login shell, a desktop autostart entry, or right after a reboot
# before any shell has sourced ~/.bashrc), so `opencode` on PATH can't be
# assumed even when it's installed and works fine in a terminal.
_OPENCODE_FALLBACK_DIRS = [
    "~/.npm-global/bin",
    "~/.bun/bin",
    "~/.local/bin",
    "~/go/bin",
    "/usr/local/bin",
]


def _find_opencode() -> str | None:
    found = shutil.which("opencode")
    if found:
        return found
    for directory in _OPENCODE_FALLBACK_DIRS:
        candidate = os.path.join(os.path.expanduser(directory), "opencode")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None

_LYRICS_STRUCTURE_HINT = (
    "Format the lyrics using ACE-Step structure tags on their own line, "
    "e.g. [Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge], "
    "[Instrumental Break], [Outro] - matching the tags implied by the "
    "song description. Output ONLY the lyrics (with tags), no commentary, "
    "no code fences, no explanation."
)

_CAPTION_STYLE_HINT = (
    "Write ONE dense paragraph (no line breaks, no lists, no headings) "
    "describing the song's genre, instrumentation, mood, tempo feel, and "
    "vocal style - the way a music-generation prompt is written, e.g. "
    "'dreamy shoegaze with reverb-drenched guitars and whispered vocals'. "
    "Output ONLY that paragraph, no commentary, no code fences, no quotes."
)


def _build_lyrics_prompt(caption: str, lyrics: str) -> str:
    caption = (caption or "").strip()
    request = (lyrics or "").strip()
    parts = ["Write song lyrics."]
    if caption:
        parts.append(f"Song description / style: {caption}")
    if request:
        parts.append(
            "User's request (may be a short instruction like a theme, or an "
            f"existing draft to rewrite/improve): {request}"
        )
    else:
        parts.append("No specific request was given - write lyrics that fit the song description above.")
    parts.append(_LYRICS_STRUCTURE_HINT)
    return "\n\n".join(parts)


def _build_caption_prompt(caption: str, lyrics: str) -> str:
    request = (caption or "").strip()
    lyrics = (lyrics or "").strip()
    parts = ["Write a music-generation caption (style/genre description) for a song."]
    if request:
        parts.append(
            "User's request (may be a short instruction like a genre/mood, or an "
            f"existing draft to rewrite/improve): {request}"
        )
    else:
        parts.append("No specific request was given - infer a fitting style from the lyrics below.")
    if lyrics:
        parts.append(f"The song's lyrics, for context: {lyrics}")
    parts.append(_CAPTION_STYLE_HINT)
    return "\n\n".join(parts)


def _extract_text_from_opencode_json(stdout: str) -> str:
    """Concatenate the assistant's text parts from `opencode run --format json` output."""
    chunks = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part", {})
        if part.get("type") == "text" and event.get("type") == "text":
            text = part.get("text", "")
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def _run_opencode(prompt: str):
    """Run `opencode run` with *prompt* and return (text_or_none, status_message_or_none)."""
    opencode_path = _find_opencode()
    if opencode_path is None:
        status_message = t("messages.opencode_not_found")
        gr.Warning(status_message)
        return None, status_message

    try:
        result = subprocess.run(
            [opencode_path, "run", "--format", "json", prompt],
            capture_output=True,
            text=True,
            timeout=_OPENCODE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        status_message = t("messages.opencode_timeout")
        gr.Warning(status_message)
        return None, status_message
    except Exception as exc:
        status_message = t("messages.opencode_failed", error=str(exc))
        gr.Warning(status_message)
        return None, status_message

    if result.returncode != 0:
        status_message = t("messages.opencode_failed", error=result.stderr.strip()[-300:])
        gr.Warning(status_message)
        return None, status_message

    text = _extract_text_from_opencode_json(result.stdout)
    if not text:
        status_message = t("messages.opencode_empty_response")
        gr.Warning(status_message)
        return None, status_message

    return text, None


def write_lyrics_with_opencode(caption: str, lyrics: str):
    """Send caption + current Lyrics field content to `opencode` and return new lyrics.

    Returns:
        Tuple of (lyrics_update, status_message).
    """
    text, status_message = _run_opencode(_build_lyrics_prompt(caption, lyrics))
    if text is None:
        return gr.update(), status_message
    return gr.update(value=text), t("messages.opencode_lyrics_success")


def write_caption_with_opencode(caption: str, lyrics: str):
    """Send caption request + current Lyrics field (for context) to `opencode` and return new caption.

    Returns:
        Tuple of (caption_update, status_message).
    """
    text, status_message = _run_opencode(_build_caption_prompt(caption, lyrics))
    if text is None:
        return gr.update(), status_message
    return gr.update(value=text), t("messages.opencode_caption_success")
