"""Deterministic regression tests: no downloaded models or GPU required."""
import asyncio
import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from audio_utils import read_audio
from contracts import TranscriptionResult, VoiceProfile, VoiceState
from pipeline import TranslationCallPipeline, PipelineError
from profiles import VoiceProfiles
from stt import MLXWhisperSTTService


class FakeSTT:
    def transcribe_detailed(self, path, language):
        text = Path(path).stem
        return TranscriptionResult(text, language or 'ur', 'configured', -.1, .01, .01, text)


class FakeTranslator:
    def __init__(self):
        self.inputs = []
    def translate(self, text, source_lang, target_lang):
        self.inputs.append((text, source_lang, target_lang))
        return f'{target_lang}:{text}', .01


class FakeVoice:
    supported_languages = {'en', 'ur'}
    supported_reference_languages = {'en', 'ur'}
    def __init__(self):
        self.requests = []
    def clone_voice(self, **kw):
        self.requests.append(kw)
        sf.write(kw['output_path'], np.ones(1600)*.1, 16000)
        return kw['output_path'], .01


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / 'sentence.wav'
        # A 440 Hz reference makes pitch and duration changes measurable.
        sf.write(self.audio, .1*np.sin(2*np.pi*440*np.arange(48000*4)/48000), 48000)
        self.profiles = VoiceProfiles(self.root / 'profiles')
        self.translator, self.voice = FakeTranslator(), FakeVoice()
        self.pipeline = TranslationCallPipeline(FakeSTT(), self.translator, self.voice, profiles=self.profiles)

    def run_segment(self, speaker='alice', **kw):
        return self.pipeline.process_utterance(str(self.audio), speaker, call_id='call1',
            segment_id=kw.pop('segment_id', '1'), output_audio_path=str(self.root/'out.wav'), **kw)

    def test_resampling_preserves_duration_pitch(self):
        audio = read_audio(self.audio)
        self.assertEqual(len(audio), 64000)
        peak = np.argmax(abs(np.fft.rfft(audio))) / 4
        self.assertAlmostEqual(peak, 440, delta=1)

    def test_enrollment_resamples_and_versions(self):
        p = self.profiles.enroll('alice', self.audio, 'The exact words', 'en')
        self.assertAlmostEqual(sf.info(p.audio_path).duration, 4)
        q = self.profiles.enroll('alice', self.audio, 'Other words', 'en')
        self.assertNotEqual(p.profile_id, q.profile_id)
        self.assertEqual(self.profiles.get('alice').profile_id, q.profile_id)
        self.assertTrue(Path(p.audio_path).exists())

    def test_missing_clone_never_synthesizes(self):
        result = self.run_segment()
        self.assertEqual(result['status'], 'clone_not_ready')
        self.assertFalse(self.voice.requests)

    def test_missing_reference_transcript_rejected(self):
        with self.assertRaises(ValueError):
            self.profiles.enroll('alice', self.audio, '', 'en')
        self.assertEqual(self.profiles.states['alice'], VoiceState.ERROR)

    def test_participants_and_languages_do_not_cross(self):
        for speaker, src, dst in [('alice','ur','en'), ('bob','en','ur'), ('alice','ur','en')]:
            if not self.profiles.get(speaker):
                self.profiles.enroll(speaker, self.audio, speaker+' reference', 'en')
            result = self.run_segment(speaker, source_language=src, target_language=dst)
            self.assertEqual(result['status'], 'success')
            self.assertEqual(self.voice.requests[-1]['language'], dst)
            self.assertEqual(self.voice.requests[-1]['ref_text'], speaker+' reference')
        self.assertEqual(len(self.translator.inputs), 3)
        self.assertEqual(self.translator.inputs[0], self.translator.inputs[2])

    def test_profile_mismatch_rejected(self):
        p = self.profiles.enroll('bob', self.audio, 'bob reference', 'en')
        with self.assertRaises(PipelineError):
            self.run_segment(voice_profile=p)
        self.assertFalse(self.voice.requests)

    def test_unsupported_language_preserves_text_without_audio(self):
        self.profiles.enroll('alice', self.audio, 'reference', 'en')
        self.voice.supported_languages = {'en'}
        result = self.run_segment(source_language='en', target_language='ur')
        self.assertEqual(result['status'], 'unsupported_synthesis_language')
        self.assertTrue(result['translated_text'])
        self.assertIsNone(result['output_audio_path'])
        self.assertFalse(self.voice.requests)

    def test_empty_stt_stops_translation(self):
        self.pipeline.stt.transcribe_detailed = lambda *a: TranscriptionResult('', 'ur', 'configured', -2, .9, .1, 'bad')
        result = self.run_segment()
        self.assertEqual(result['status'], 'low_confidence')
        self.assertFalse(self.translator.inputs)

    def test_empty_translation_stops_synthesis(self):
        self.translator.translate = lambda *a, **k: ('  ', .01)
        with self.assertRaises(PipelineError):
            self.run_segment()
        self.assertFalse(self.voice.requests)

    def test_stt_audio_and_context_contract(self):
        class Whisper:
            @staticmethod
            def transcribe(audio, **kw):
                self.assertEqual(audio.shape, (64000,))
                self.assertEqual(audio.dtype, np.float32)
                self.assertEqual(kw['task'], 'transcribe')
                self.assertFalse(kw['condition_on_previous_text'])
                self.assertIsNone(kw['initial_prompt'])
                self.assertEqual(kw['language'], 'ur')
                return dict(text='اردو', language='ur', segments=[dict(avg_logprob=-.1, no_speech_prob=.01)])
        with patch.dict(sys.modules, mlx_whisper=Whisper):
            result = MLXWhisperSTTService().transcribe_detailed(str(self.audio), 'ur')
        self.assertEqual(result.text, 'اردو')
        self.assertEqual(result.avg_logprob, -.1)

    def test_path_identifiers_rejected(self):
        with self.assertRaises(ValueError):
            self.profiles.get('../bob')


class AsyncIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ended_call_discards_inflight_result(self):
        import server
        class Socket:
            def __init__(self): self.messages = []
            async def send_json(self, value): self.messages.append(value)
        a, b = server.Connection(Socket()), server.Connection(Socket())
        call = server.Call('testcall', 'alice', 'bob', True)
        a.call_id = b.call_id = call.call_id
        a.source, b.receive = 'ur', 'en'
        server.connections.update(alice=a, bob=b)
        server.calls[call.call_id] = call
        async def fake_infer(**kw):
            self.assertEqual(kw['target_language'], 'en')
            server.calls.pop(call.call_id)
            return {'status':'success'}
        try:
            with patch.object(server, 'infer', fake_infer):
                await server.process_segment('alice', a, call, 1, b'wav', None)
            self.assertEqual([m['type'] for m in a.ws.messages], ['processing_started'])
            self.assertEqual(b.ws.messages, [])
            self.assertFalse(a.busy)
        finally:
            server.connections.clear()
            server.calls.clear()


if __name__ == '__main__':
    unittest.main()
