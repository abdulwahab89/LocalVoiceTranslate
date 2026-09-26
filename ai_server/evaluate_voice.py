"""First-request synthesis smoke/ intelligibility check using explicit demo fixtures.
This does not establish perceptual speaker similarity or Urdu support.
"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from pipeline import TranslationCallPipeline
from profiles import VoiceProfiles
from stt import MLXWhisperSTTService


class IdentityTranslation:
    def translate(self,text,source_lang,target_lang):
        assert source_lang == target_lang == 'en'
        return text, 0.


def main():
    root = Path(__file__).resolve().parent
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    report = []
    with tempfile.TemporaryDirectory() as temp:
        profiles = VoiceProfiles(temp)
        stt = MLXWhisperSTTService()
        pipeline = TranslationCallPipeline(stt, IdentityTranslation(), profiles=profiles, default_steps=args.steps)
        for number, fixture in enumerate(['user_a','user_b'],1):
            profile = profiles.enroll('fixture_'+fixture, root/'samples'/f'{fixture}.wav',
                (root/'samples'/f'{fixture}.txt').read_text(), 'en')
            result = pipeline.process_utterance(str(root/'samples/english_sentence.wav'),
                profile.speaker_id, 'en','en',call_id='voice_evaluation',segment_id=str(number),
                output_audio_path=str(root/'outputs'/f'evaluation_{fixture}.wav'))
            if result['output_audio_path']:
                result['output_stt'] = asdict(stt.transcribe_detailed(result['output_audio_path'],'en'))
            report.append(result)
    Path(f'docs/voice-evaluation-{args.steps}-steps.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
