#!/usr/bin/env python3
"""
Stage 4 Test Script: Complete AI Pipeline

Tests end-to-end voice calling pipeline in both directions:
1. User A (Urdu -> English):
   User A Urdu Audio -> STT -> Urdu-to-English Translation -> TTS cloned with User A voice -> Output WAV
2. User B (English -> Urdu):
   User B English Audio -> STT -> English-to-Urdu Translation -> TTS cloned with User B voice -> Output WAV
"""

import argparse
import os
import sys
import time

from pipeline import TranslationCallPipeline, PipelineError
from stt import record_microphone_audio, STTError


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Complete AI Pipeline Test")
    parser.add_argument("--mic", action="store_true", help="Record live Urdu input from microphone")
    args = parser.parse_args()

    print("=" * 65)
    print("STAGE 4 — COMPLETE AI TRANSLATION CALL PIPELINE")
    print("=" * 65)

    # 1. Initialize complete pipeline (loads STT, Translation, and Voice Clone models)
    print("\n[Initializing Full AI Pipeline (STT + NMT + Cloned TTS)]")
    t0 = time.time()
    pipeline = TranslationCallPipeline(default_steps=4)
    print(f"Pipeline initialized and warmed up in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # Test 1: User A speaks Urdu -> User B hears English in User A's cloned voice
    # -------------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("DIRECTION 1: USER A (Urdu) -> USER B (English)")
    print("-" * 65)

    audio_source_a = "ai_server/samples/urdu_sentence.wav"
    if args.mic:
        audio_source_a = "ai_server/outputs/mic_user_a_input.wav"
        print("Speak an Urdu sentence into your microphone in 3 seconds...")
        for count in [3, 2, 1]:
            print(f"  {count}...")
            time.sleep(1)
        print("  RECORDING (speak now for 4s)...")
        record_microphone_audio(audio_source_a, duration=4.0)
        print("  Finished recording.")

    print(f"Input Audio       : {audio_source_a}")
    print(f"Speaker ID        : user_a")
    print(f"Enrolled Voice    : ai_server/samples/user_a.wav")

    res_a = pipeline.process_utterance(
        audio_source=audio_source_a,
        speaker_id="user_a",
        source_language="ur",
        target_language="en",
        output_audio_path="ai_server/outputs/stage4_call_user_b_hears.wav",
        tts_steps=4,
    )

    assert res_a["status"] == "success", res_a.get("error", res_a["status"])

    print("\n[Result for User B (Hearing User A in English)]:")
    print(f"  Original Speech (Urdu) : {res_a['transcript']}")
    print(f"  Translated Text (Eng)  : {res_a['translated_text']}")
    print(f"  Synthesized Audio File : {res_a['output_audio_path']}")
    print(f"  Latency Metrics:")
    print(f"    STT         : {res_a['metrics']['stt_ms']:.1f} ms")
    print(f"    Translation : {res_a['metrics']['translation_ms']:.1f} ms")
    print(f"    TTS         : {res_a['metrics']['tts_ms']:.1f} ms")
    print(f"    Total       : {res_a['metrics']['total_ms']:.1f} ms")

    # -------------------------------------------------------------------------
    # Test 2: User B speaks English -> User A hears Urdu in User B's cloned voice
    # -------------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("DIRECTION 2: USER B (English) -> USER A (Urdu)")
    print("-" * 65)

    audio_source_b = "ai_server/samples/english_sentence.wav"
    print(f"Input Audio       : {audio_source_b}")
    print(f"Speaker ID        : user_b")
    print(f"Enrolled Voice    : ai_server/samples/user_b.wav")

    res_b = pipeline.process_utterance(
        audio_source=audio_source_b,
        speaker_id="user_b",
        source_language="en",
        target_language="ur",
        output_audio_path="ai_server/outputs/stage4_call_user_a_hears.wav",
        tts_steps=4,
    )

    assert res_b["status"] == "success", res_b.get("error", res_b["status"])

    print("\n[Result for User A (Hearing User B in Urdu)]:")
    print(f"  Original Speech (Eng)  : {res_b['transcript']}")
    print(f"  Translated Text (Urdu) : {res_b['translated_text']}")
    if res_b.get("phonetic_text"):
        print(f"  Phonetic Spoken (Urdu) : {res_b['phonetic_text']}")
    print(f"  Synthesized Audio File : {res_b['output_audio_path']}")
    print(f"  Latency Metrics:")
    print(f"    STT         : {res_b['metrics']['stt_ms']:.1f} ms")
    print(f"    Translation : {res_b['metrics']['translation_ms']:.1f} ms")
    print(f"    TTS         : {res_b['metrics']['tts_ms']:.1f} ms")
    print(f"    Total       : {res_b['metrics']['total_ms']:.1f} ms")

    print("\n" + "=" * 65)
    print("STAGE 4 SYNTHESIS SMOKE TEST PASSED (quality requires review)")
    print("=" * 65)


if __name__ == "__main__":
    main()
