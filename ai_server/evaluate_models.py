"""Reproducible model evidence, not nonempty-output 'accuracy' assertions.

Run from repo root: PYTHONPATH=ai_server ai_server/venv/bin/python ai_server/evaluate_models.py
Use --english-audio for the exact Amjad English recording and --urdu-audio for Urdu.
Without it the existing short English demo is evaluated separately.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
from stt import MLXWhisperSTTService
from translator import create_translation_service

URDU = 'میرا نام امجد ہے اور میں سلائی کا کام کرتا ہوں۔ آپ مجھے پہچانتے ہیں؟'
ENGLISH = 'My name is Amjad and I work as a tailor. Do you recognize me?'


def word_error_rate(reference, hypothesis):
    def words(text): return re.sub(r'[^\w\s]', '', text.lower()).split()
    a, b = words(reference), words(hypothesis)
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        next_row = [i]
        for j, y in enumerate(b, 1):
            next_row.append(min(next_row[-1]+1, row[j]+1, row[j-1]+(x != y)))
        row = next_row
    return row[-1]/max(1,len(a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mlx-community/whisper-small-mlx')
    parser.add_argument('--urdu-audio', default='ai_server/outputs/ws_in_user_a_1790455066307.wav')
    parser.add_argument('--english-audio')
    parser.add_argument('--output', default='docs/model-evaluation.json')
    args = parser.parse_args()
    stt, translator = MLXWhisperSTTService(args.model), create_translation_service()
    report = {'english_fixture_note': 'The full English evaluation fixture may be synthetic; it is not proof of live microphone accuracy', 'stt_model': stt.model_name, 'translation_model': type(translator).__name__,
              'stt': [], 'translation': [], 'notice': 'WER is diagnostic; semantic/voice quality requires review'}
    tests = [(args.urdu_audio, 'ur', URDU),
             (args.english_audio or 'ai_server/samples/english_sentence.wav', 'en',
              ENGLISH if args.english_audio else 'Where are you going?')]
    # Alternate participant languages and repeat the first to detect accidental context carryover.
    for path, lang, expected in tests + tests[:1]:
        result = stt.transcribe_detailed(path, lang)
        item = dict(audio=path, expected=expected, wer=word_error_rate(expected, result.text), **asdict(result))
        report['stt'].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        if result.text:
            target = 'en' if lang == 'ur' else 'ur'
            text, elapsed = translator.translate(result.text, lang, target)
            report['translation'].append(dict(source=lang, target=target, input=result.text, output=text, seconds=elapsed, from_audio=True))
    for text, src, dst in [(URDU,'ur','en'), (ENGLISH,'en','ur'), ('The weather is pleasant today.','en','ur'),
                           (URDU,'ur','en'), (URDU,'ur','ur'), (ENGLISH,'en','en')]:
        output, elapsed = translator.translate(text,src,dst)
        item = dict(source=src,target=dst,input=text,output=output,seconds=elapsed,from_audio=False)
        report['translation'].append(item)
        print(json.dumps(item, ensure_ascii=False),flush=True)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
