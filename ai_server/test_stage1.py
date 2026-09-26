#!/usr/bin/env python3
"""
Stage 1 Test Script: Voice Clone Proof

Input:
    samples/user_a.wav
Text:
    "Hello, this is a test of my translated voice."
Output:
    outputs/output.wav
"""

import os
import sys
import time
from pathlib import Path
import mlx.core as mx
import soundfile as sf

from voice_clone import F5TTSMLXVoiceCloningService


def main():
    print("=" * 60)
    print("STAGE 1 — VOICE CLONE PROOF")
    print("=" * 60)

    # 1. Environment & Hardware Information
    device = mx.default_device()
    py_version = sys.version.split()[0]
    print(f"Python Version    : {py_version}")
    print(f"MLX Device        : {device} (Apple Silicon GPU)")
    print(f"Platform          : macOS arm64")

    # 2. Input Setup
    ref_audio_path = "ai_server/samples/user_a.wav"
    ref_text = "Some call me nature, others call me mother nature."
    input_text = "Hello, this is a test of my translated voice."
    output_path = "ai_server/outputs/output.wav"

    if not os.path.exists(ref_audio_path):
        print(f"ERROR: Reference audio not found at {ref_audio_path}")
        sys.exit(1)

    ref_info = sf.info(ref_audio_path)
    print(f"\n[Input Reference Voice]")
    print(f"  Path            : {ref_audio_path}")
    print(f"  Sample Rate     : {ref_info.samplerate} Hz")
    print(f"  Channels        : {ref_info.channels}")
    print(f"  Duration        : {ref_info.duration:.2f} s")
    print(f"  Reference Text  : \"{ref_text}\"")

    print(f"\n[Synthesis Request]")
    print(f"  Text to Clone   : \"{input_text}\"")
    print(f"  Output Path     : {output_path}")

    # 3. Model Initialization
    print(f"\n[Model Initialization]")
    t_load_start = time.time()
    service = F5TTSMLXVoiceCloningService(model_name="lucasnewman/f5-tts-mlx", default_steps=6)
    t_load_end = time.time()
    print(f"  Load Time       : {t_load_end - t_load_start:.2f} s")

    # 4. Voice Cloning Inference
    print(f"\n[Executing Voice Cloning]")
    t_start = time.time()
    generated_file, synthesis_time = service.clone_voice(
        text=input_text,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        output_path=output_path,
        steps=6,
    )
    t_total = time.time() - t_start

    # 5. Output Verification
    if not os.path.exists(output_path):
        print(f"ERROR: Output file {output_path} was not created!")
        sys.exit(1)

    out_info = sf.info(output_path)
    file_size_kb = os.path.getsize(output_path) / 1024

    print(f"\n[Output Verification]")
    print(f"  File Status     : Created successfully")
    print(f"  File Location   : {os.path.abspath(output_path)}")
    print(f"  File Size       : {file_size_kb:.1f} KB")
    print(f"  Sample Rate     : {out_info.samplerate} Hz")
    print(f"  Channels        : {out_info.channels}")
    print(f"  Duration        : {out_info.duration:.2f} s")
    print(f"  Execution Time  : {t_total:.2f} s ({t_total * 1000:.0f} ms)")

    print("\n" + "=" * 60)
    print("STAGE 1 TEST PASSED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
