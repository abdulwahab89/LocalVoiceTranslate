"""
Voice Cloning Service Module for Translation Call Demo.

Optimized for Apple Silicon (macOS M1/M2/M3/M4) using Apple's MLX framework.
"""

from abc import ABC, abstractmethod
import datetime
import logging
import os
from pathlib import Path
import time
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np
import soundfile as sf

from f5_tts_mlx.cfm import F5TTS
from f5_tts_mlx.generate import (
    convert_char_to_pinyin,
    split_sentences,
    estimated_duration,
    TARGET_RMS,
    SAMPLE_RATE,
    FRAMES_PER_SEC,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class VoiceCloningService(ABC):
    """Abstract interface for voice cloning TTS models."""

    @abstractmethod
    def clone_voice(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: Optional[str] = None,
        output_path: Optional[str] = None,
        steps: int = 6,
    ) -> Tuple[str, float]:
        """
        Synthesize speech in the target voice of ref_audio_path.

        Args:
            text: The text to be spoken.
            ref_audio_path: Path to the reference audio WAV file (enrolled user voice).
            ref_text: Transcription of the reference audio (improves flow matching alignment).
            output_path: Destination WAV path. If None, generates an automated path.
            steps: Number of Euler/RK4 flow matching ODE steps (fewer = faster).

        Returns:
            Tuple of (output_file_path, synthesis_duration_seconds).
        """
        pass


class F5TTSMLXVoiceCloningService(VoiceCloningService):
    """
    MLX-accelerated implementation of F5-TTS for Apple Silicon.
    Directly utilizes Apple Silicon GPU via Metal Performance Shaders / MLX.
    """

    def __init__(
        self,
        model_name: str = "lucasnewman/f5-tts-mlx",
        default_steps: int = 6,
    ):
        self.model_name = model_name
        self.default_steps = default_steps
        self.target_sample_rate = SAMPLE_RATE  # 24000 Hz

        logger.info(f"Initializing F5TTS MLX model: {model_name} on device: {mx.default_device()}")
        t0 = time.time()
        self.model = F5TTS.from_pretrained(model_name)
        logger.info(f"Model loaded successfully in {time.time() - t0:.2f}s")

    def _prepare_reference_audio(self, ref_audio_path: str) -> mx.array:
        """Loads and normalizes the reference audio, ensuring mono 24kHz."""
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference voice audio file not found: {ref_audio_path}")

        audio, sr = sf.read(ref_audio_path)
        # Convert stereo to mono if necessary
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if sample rate doesn't match 24kHz
        if sr != self.target_sample_rate:
            logger.warning(
                f"Reference audio sample rate {sr}Hz != {self.target_sample_rate}Hz. Resampling."
            )
            # Resample using linear interpolation or numpy
            duration = len(audio) / sr
            target_length = int(duration * self.target_sample_rate)
            audio = np.interp(
                np.linspace(0, len(audio), target_length, endpoint=False),
                np.arange(len(audio)),
                audio,
            )

        audio_mx = mx.array(audio, dtype=mx.float32)

        # Normalize RMS volume to target
        rms = mx.sqrt(mx.mean(mx.square(audio_mx)))
        if rms > 0 and rms < TARGET_RMS:
            audio_mx = audio_mx * TARGET_RMS / rms

        return audio_mx

    def clone_voice(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: Optional[str] = None,
        output_path: Optional[str] = None,
        steps: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Generates cloned speech for the given text using reference voice audio."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        num_steps = steps if steps is not None else self.default_steps

        # Resolve output file path
        if output_path is None:
            timestamp = int(time.time() * 1000)
            output_dir = Path("ai_server/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"cloned_{timestamp}.wav")
        else:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # If ref_text is not given, attempt reading companion .txt if present
        if not ref_text:
            companion_txt = Path(ref_audio_path).with_suffix(".txt")
            if companion_txt.exists():
                ref_text = companion_txt.read_text(encoding="utf-8").strip()
            else:
                ref_text = "Some call me nature, others call me mother nature."

        logger.info(f"Synthesizing '{text}' (steps={num_steps}) with ref voice '{ref_audio_path}'")
        t_start = time.time()

        ref_audio = self._prepare_reference_audio(ref_audio_path)
        ref_audio_samples = ref_audio.shape[0]

        sentences = split_sentences(text)
        outputs = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            full_prompt = f"{ref_text} {sentence}"
            pinyin_text = convert_char_to_pinyin([full_prompt])

            # Generate waveform with MLX
            wave, _ = self.model.sample(
                mx.expand_dims(ref_audio, axis=0),
                text=pinyin_text,
                steps=num_steps,
                method="rk4",
                speed=1.0,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
            )

            # Trim out the prompt reference portion of the audio
            gen_wave = wave[ref_audio_samples:]
            mx.eval(gen_wave)
            outputs.append(gen_wave)

        if not outputs:
            raise RuntimeError("Voice synthesis produced no output audio.")

        final_wave = mx.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]
        mx.eval(final_wave)

        # Save to WAV file
        audio_np = np.array(final_wave)
        sf.write(output_path, audio_np, self.target_sample_rate)

        elapsed = time.time() - t_start
        audio_duration = len(audio_np) / self.target_sample_rate
        logger.info(
            f"Voice clone complete: {audio_duration:.2f}s audio generated in {elapsed:.2f}s "
            f"(saved to {output_path})"
        )

        return output_path, elapsed
