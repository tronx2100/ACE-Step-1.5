"""Local, offline lyrics transcription via faster-whisper.

Used by the "Analyze" action to produce lyrics that actually match the
words sung in the source audio, instead of the 5Hz LM's reconstruction
from audio codes (which captures style/structure well but frequently
gets the actual words wrong). Runs fully locally, no external API calls.
"""

import gc
from typing import Optional

import torch
from loguru import logger

_WHISPER_MODEL_SIZE = "large-v3-turbo"


def transcribe_lyrics_locally(
    audio_path: str,
    language: Optional[str] = None,
    device: str = "cuda",
) -> Optional[str]:
    """Transcribe sung lyrics from an audio file using a local Whisper model.

    Args:
        audio_path: Path to the source audio file.
        language: Optional ISO language code hint (e.g. "de", "en"). Pass
            None or "unknown" to let Whisper auto-detect.
        device: "cuda" or "cpu". The model is loaded, used, and freed again
            within this call, so GPU use is transient.

    Returns:
        Plain-text lyrics (one line per detected phrase, no timestamps or
        structure tags), or None if transcription failed.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("[local_transcription] faster-whisper not installed; skipping local transcription")
        return None

    use_cuda = device == "cuda" and torch.cuda.is_available()
    compute_type = "float16" if use_cuda else "int8"
    resolved_device = "cuda" if use_cuda else "cpu"

    model = None
    try:
        model = WhisperModel(_WHISPER_MODEL_SIZE, device=resolved_device, compute_type=compute_type)
        lang_hint = language if language and language != "unknown" else None
        # vad_filter is tuned for spoken speech and frequently misclassifies
        # sung vocals (with instrumentation) as non-speech, filtering out
        # entire songs. Leave it off for music; instead drop individual
        # high no_speech_prob segments below, which catches the "Thank you."
        # / "Thanks for watching." hallucinations Whisper produces on
        # purely instrumental stretches.
        segments, _info = model.transcribe(audio_path, language=lang_hint, vad_filter=False)
        lines = [
            seg.text.strip()
            for seg in segments
            if seg.text.strip() and seg.no_speech_prob < 0.6
        ]
        if not lines:
            return None

        # Whisper has a well-known failure mode on quiet/instrumental audio:
        # it confidently (no_speech_prob near 0) hallucinates short stock
        # phrases like "Thank you." or "Thanks for watching." repeated over
        # and over, inherited from its YouTube training data. Real lyrics
        # rarely consist of the same one or two short lines repeated
        # throughout an entire song, so treat that pattern as hallucination.
        unique_lines = set(lines)
        if len(lines) >= 4 and len(unique_lines) <= 2:
            logger.warning(
                f"[local_transcription] Discarding likely hallucinated transcript "
                f"(only {len(unique_lines)} unique line(s) across {len(lines)} segments): {unique_lines}"
            )
            return None

        return "\n".join(lines)
    except Exception as exc:
        logger.warning(f"[local_transcription] Whisper transcription failed: {exc}")
        return None
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
