#!/usr/bin/env python3
"""
Stage 3 Test Script: Translation

Tests bidirectional translation:
1. Urdu -> English
2. English -> Urdu
"""

import sys
import time

from translator import MarianTranslationService, TranslationError


def main():
    print("=" * 60)
    print("STAGE 3 — TRANSLATION (URDU <-> ENGLISH)")
    print("=" * 60)

    # 1. Initialize translation service
    print("\n[Initializing Local Translation Service]")
    t0 = time.time()
    try:
        translator = MarianTranslationService(device="cpu")
    except TranslationError as e:
        print(f"ERROR: Failed to initialize translation service: {e}")
        sys.exit(1)
    print(f"Service initialized in {time.time() - t0:.2f}s")

    # 2. Test Urdu -> English (from Stage 2 STT transcript)
    test_urdu = "تم کیا کر رہے ہو؟"
    print("\n[Test 1: Urdu -> English]")
    translated_en, lat_en = translator.translate(test_urdu, source_lang="ur", target_lang="en")

    print("ORIGINAL:")
    print(f"  {test_urdu}")
    print("TRANSLATED:")
    print(f"  {translated_en}")
    print(f"Latency: {lat_en * 1000:.1f} ms")

    # 3. Test English -> Urdu (Reverse direction for User B)
    test_english = "Where are you going?"
    print("\n[Test 2: English -> Urdu]")
    translated_ur, lat_ur = translator.translate(test_english, source_lang="en", target_lang="ur")

    print("ORIGINAL:")
    print(f"  {test_english}")
    print("TRANSLATED:")
    print(f"  {translated_ur}")
    print(f"Latency: {lat_ur * 1000:.1f} ms")

    # 4. Verify outputs
    assert translated_en.strip() != "", "Urdu to English translation returned empty text"
    assert translated_ur.strip() != "", "English to Urdu translation returned empty text"

    print("\n" + "=" * 60)
    print("STAGE 3 TEST PASSED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
