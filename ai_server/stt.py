"""
Speech-to-Text (STT) Service Module for Translation Call Demo.

Optimized for Apple Silicon using mlx-whisper with Metal acceleration.
Supports multilingual transcription including Urdu (ur) and English (en).
"""

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
import time
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf

from audio_utils import read_audio, mono_resample
from contracts import TranscriptionResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class STTError(Exception):
    """Base exception for Speech-to-Text errors."""
    pass


class SpeechToTextService(ABC):
    """Abstract interface for Speech-to-Text (STT) services."""

    @abstractmethod
    def transcribe(
        self,
        audio_source: Union[str, np.ndarray],
        language: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Transcribe speech audio into text.

        Args:
            audio_source: File path to WAV/audio or numpy array of audio samples.
            language: Optional language code ('ur', 'en', etc.). If None, auto-detected.

        Returns:
            Tuple of (transcribed_text, elapsed_seconds).
        """
        pass


class MLXWhisperSTTService(SpeechToTextService):
    """
    Apple Silicon MLX implementation of OpenAI Whisper.
    Runs on the M-series GPU via Metal with zero cloud dependencies.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/whisper-small-mlx",
    ):
        self.model_name = os.environ.get("STT_MODEL", model_name)
        if self.model_name.endswith(".en") or ".en-" in self.model_name:
            raise STTError("English-only models cannot serve Urdu calls")
        logger.info("STT model: %s", self.model_name)

    def transcribe(self, audio_source, language=None):
        result = self.transcribe_detailed(audio_source, language)
        return result.text, result.elapsed_seconds

    def transcribe_detailed(self, audio_source, language=None):
        import mlx_whisper
        t0 = time.perf_counter()
        try:
            audio = (read_audio(audio_source, 16000) if isinstance(audio_source, (str, Path))
                     else mono_resample(audio_source, 16000, 16000))
            if len(audio) / 16000 > 30:
                raise STTError("Utterance exceeds 30 seconds; finish the sentence and send")
            mode = 'configured' if language else 'detected'
            if len(audio) < 3200 or np.sqrt(np.mean(audio ** 2)) < 0.001:
                return TranscriptionResult('', language, mode, None, None, time.perf_counter()-t0)
            result = mlx_whisper.transcribe(
                audio, path_or_hf_repo=self.model_name, language=language,
                task='transcribe', condition_on_previous_text=False,
                initial_prompt=None, temperature=0.0, no_speech_threshold=0.6,
            )
            segments = result.get('segments', [])
            scores = [s['avg_logprob'] for s in segments if 'avg_logprob' in s]
            silence = [s['no_speech_prob'] for s in segments if 'no_speech_prob' in s]
            score = sum(scores)/len(scores) if scores else None
            no_speech = sum(silence)/len(silence) if silence else None
            raw = result.get('text', '')
            text = raw.strip()
            if score is not None and score < -1.0:
                text = ''
            if no_speech is not None and no_speech > 0.8:
                text = ''
            return TranscriptionResult(text, result.get('language', language), mode,
                                       score, no_speech, time.perf_counter()-t0, raw)
        except STTError:
            raise
        except Exception as exc:
            raise STTError(f"Transcription failed: {exc}") from exc


def record_microphone_audio(
    output_path: str,
    duration: float = 4.0,
    sample_rate: int = 16000,
) -> str:
    """
    Helper function to record speech from the local microphone.
    Handles device permissions and creates a mono 16kHz WAV file.
    """
    try:
        import sounddevice as sd
    except ImportError as e:
        raise STTError("sounddevice package is required for microphone recording.") from e

    logger.info(f"Recording from microphone for {duration:.1f}s at {sample_rate}Hz...")
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as e:
        raise STTError(f"Microphone recording failed (check permissions in System Settings): {e}") from e

    # Check for empty / zero signal
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-4:
        logger.warning("Recorded audio has very low energy (silence or muted mic).")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    logger.info(f"Recorded audio saved to: {output_path}")
    return output_path
