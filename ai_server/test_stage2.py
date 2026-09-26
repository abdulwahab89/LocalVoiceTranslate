#!/usr/bin/env python3
"""
Stage 2 Test Script: Speech Recognition (STT)

Input:
    Urdu audio (sample file or live microphone)
Output:
    Urdu text transcription printed to console
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import soundfile as sf

from stt import MLXWhisperSTTService, record_microphone_audio, STTError


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Speech Recognition Test")
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Record a live 4-second utterance from microphone instead of using sample file",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="ai_server/samples/urdu_sentence.wav",
        help="Path to Urdu audio file to transcribe",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/whisper-tiny",
        help="MLX Whisper model to use",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("STAGE 2 — SPEECH RECOGNITION (URDU STT)")
    print("=" * 60)

    # 1. Environment & Hardware
    device = mx.default_device()
    py_version = sys.version.split()[0]
    print(f"Python Version    : {py_version}")
    print(f"MLX Device        : {device} (Apple Silicon GPU)")
    print(f"Target Language   : Urdu (ur)")

    # 2. Prepare Audio Source
    audio_path = args.audio
    if args.mic:
        mic_output = "ai_server/outputs/mic_urdu_recording.wav"
        print("\n[Microphone Mode Activated]")
        print("Please speak an Urdu sentence into your microphone in 3 seconds...")
        for count in [3, 2, 1]:
            print(f"  Starting in {count}...")
            time.sleep(1)
        print("  RECORDING NOW (speak for 4 seconds)...")
        try:
            audio_path = record_microphone_audio(mic_output, duration=4.0)
            print("  Recording finished.")
        except STTError as e:
            print(f"\nMicrophone Error: {e}")
            print("Falling back to sample audio file.")
            audio_path = args.audio

    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at {audio_path}")
        sys.exit(1)

    audio_info = sf.info(audio_path)
    print(f"\n[Audio Source]")
    print(f"  File Path       : {audio_path}")
    print(f"  Duration        : {audio_info.duration:.2f} s")
    print(f"  Sample Rate     : {audio_info.samplerate} Hz")
    print(f"  Channels        : {audio_info.channels}")

    # 3. Initialize STT Service
    print(f"\n[Model Initialization]")
    t_init = time.time()
    stt_service = MLXWhisperSTTService(model_name=args.model)
    print(f"  Model Initialized: {args.model} ({time.time() - t_init:.2f}s)")

    # 4. Transcribe Urdu Audio -> Urdu Text
    print(f"\n[Executing Transcription]")
    t_start = time.time()
    try:
        transcript, elapsed = stt_service.transcribe(audio_path, language="ur")
    except STTError as e:
        print(f"ERROR: Transcription failed: {e}")
        sys.exit(1)

    t_total = time.time() - t_start

    # 5. Output Results
    print(f"\n[Transcription Results]")
    print(f"  ORIGINAL URDU SPEECH TRANSCRIPTION:")
    print(f"  ------------------------------------")
    print(f"  >>> {transcript} <<<")
    print(f"  ------------------------------------")
    print(f"  Inference Latency: {elapsed * 1000:.1f} ms ({elapsed:.2f} s)")
    print(f"  Audio RTF (Real-Time Factor): {elapsed / audio_info.duration:.2f}x")

    print("\n" + "=" * 60)
    print("STAGE 2 TEST PASSED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
