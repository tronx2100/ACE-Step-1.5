"""Local, offline lyrics transcription via faster-whisper.

Used by the "Analyze" action to produce lyrics that actually match the
words sung in the source audio, instead of the 5Hz LM's reconstruction
from audio codes (which captures style/structure well but frequently
gets the actual words wrong). Runs fully locally, no external API calls.

The actual transcription runs in a subprocess (see `_worker_main`) rather
than in-process. Reason: ctranslate2 (faster-whisper's backend) dlopens
its CUDA libraries (libcublas, libcudnn, ...) against whatever
LD_LIBRARY_PATH the process had at startup, and the glibc loader caches
that at process init - setting os.environ["LD_LIBRARY_PATH"] later in
the same process has no effect. On this host, the process is normally
started with a system CUDA dir on LD_LIBRARY_PATH ahead of the
pip-installed nvidia-cublas-cu12/nvidia-cudnn-cu12 wheels, which pulls in
a libcublasLt from one CUDA build and libcublas from another and reliably
crashes with CUBLAS_STATUS_INVALID_VALUE. A fresh subprocess started with
the venv's own nvidia-*-cu12 lib dirs placed first on LD_LIBRARY_PATH
picks up a consistent, matching set of libraries.
"""

import gc
import json
import os
import subprocess
import sys
from typing import Optional

from loguru import logger

from acestep._cuda_lib_env import venv_cuda_lib_dirs

_WHISPER_MODEL_SIZE = "large-v3-turbo"


def transcribe_lyrics_locally(
    audio_path: str,
    language: Optional[str] = None,
    device: str = "cuda",
) -> Optional[str]:
    """Transcribe sung lyrics from an audio file using a local Whisper model.

    Runs in a subprocess (see module docstring for why) with the model
    loaded, used, and freed within that subprocess, so GPU use is transient.

    Args:
        audio_path: Path to the source audio file.
        language: Optional ISO language code hint (e.g. "de", "en"). Pass
            None or "unknown" to let Whisper auto-detect.
        device: "cuda" or "cpu".

    Returns:
        Plain-text lyrics (one line per detected phrase, no timestamps or
        structure tags), or None if transcription failed.
    """
    env = dict(os.environ)
    cuda_lib_dirs = venv_cuda_lib_dirs()
    if cuda_lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(cuda_lib_dirs + [env.get("LD_LIBRARY_PATH", "")])

    cmd = [sys.executable, "-m", "acestep.local_transcription", "--worker", audio_path, device]
    if language:
        cmd.append(language)

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    except Exception as exc:
        logger.warning(f"[local_transcription] Whisper subprocess failed to run: {exc}")
        return None

    if result.stderr:
        for line in result.stderr.splitlines():
            logger.warning(f"[local_transcription/worker] {line}")

    if result.returncode != 0:
        logger.warning(f"[local_transcription] Whisper subprocess exited with code {result.returncode}")
        return None

    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        logger.warning("[local_transcription] Could not parse subprocess output")
        return None

    return payload.get("lyrics")


def _worker_main(audio_path: str, device: str, language: Optional[str]) -> None:
    import torch
    from faster_whisper import WhisperModel

    use_cuda = device == "cuda" and torch.cuda.is_available()
    compute_type = "float16" if use_cuda else "int8"
    resolved_device = "cuda" if use_cuda else "cpu"

    model = None
    lyrics = None
    try:
        model = WhisperModel(_WHISPER_MODEL_SIZE, device=resolved_device, compute_type=compute_type)
        lang_hint = language if language and language != "unknown" else None
        # vad_filter is tempting here (it would skip instrumental stretches
        # entirely) but empirically truncates transcription on this kind of
        # audio: when Silero VAD merges a mostly-continuous vocal into one
        # large speech region (songs rarely have >~500ms silences), the
        # decoder stops after the first chunk and the rest of the song is
        # silently dropped. Leave it off; no_speech_prob/avg_logprob below
        # catch the segment-level hallucinations it would have prevented.
        segments, _info = model.transcribe(audio_path, language=lang_hint, vad_filter=False)
        lines = [
            seg.text.strip()
            for seg in segments
            if seg.text.strip() and seg.no_speech_prob < 0.6 and seg.avg_logprob > -1.0
        ]

        # Whisper has a well-known failure mode on quiet/instrumental audio:
        # it confidently (no_speech_prob near 0) hallucinates short stock
        # phrases like "Thank you." or "Thanks for watching." repeated over
        # and over, inherited from its YouTube training data. Real lyrics
        # rarely consist of the same one or two short lines repeated
        # throughout an entire song, so treat that pattern as hallucination.
        unique_lines = set(lines)
        if lines and len(lines) >= 4 and len(unique_lines) <= 2:
            logger.warning(
                f"[local_transcription] Discarding likely hallucinated transcript "
                f"(only {len(unique_lines)} unique line(s) across {len(lines)} segments): {unique_lines}"
            )
        elif lines:
            lyrics = "\n".join(lines)
    except Exception as exc:
        logger.warning(f"[local_transcription] Whisper transcription failed: {exc}")
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(json.dumps({"lyrics": lyrics}))


if __name__ == "__main__":
    # Invoked as: python -m acestep.local_transcription --worker <audio_path> <device> [language]
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        _audio_path = sys.argv[2]
        _device = sys.argv[3]
        _language = sys.argv[4] if len(sys.argv) > 4 else None
        _worker_main(_audio_path, _device, _language)
    else:
        print("Usage: python -m acestep.local_transcription --worker <audio_path> <device> [language]", file=sys.stderr)
        sys.exit(1)
