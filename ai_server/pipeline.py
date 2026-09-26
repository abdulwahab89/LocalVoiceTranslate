"""Request-local, typed utterance pipeline with explicit synthesis failures."""
import json
import logging
import os
from pathlib import Path
import time
from uuid import uuid4
import soundfile as sf
from contracts import (SegmentIdentity, CapturedAudioSegment, TranslationResult,
                       SpeechSynthesisRequest, TranslatedAudioResult, VoiceState)
from profiles import VoiceProfiles, valid_id

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


class PipelineError(Exception):
    pass


class TranslationCallPipeline:
    def __init__(self, stt_service=None, translation_service=None,
                 voice_cloning_service=None, default_steps=8, profiles=None):
        if stt_service is None:
            from stt import MLXWhisperSTTService
            stt_service = MLXWhisperSTTService()
        if translation_service is None:
            from translator import create_translation_service
            translation_service = create_translation_service()
        # Load F5 lazily: text diagnostics must work even if TTS cannot initialize.
        self.stt, self.translator = stt_service, translation_service
        self.voice_clone = voice_cloning_service
        self.tts_backend = os.environ.get("TTS_BACKEND", "f5")
        if self.tts_backend not in {"f5", "omnivoice"}:
            raise PipelineError("Unknown TTS_BACKEND")
        self.default_steps = default_steps
        self.profiles = profiles or VoiceProfiles(ROOT / 'profiles')

    def process_utterance(self, audio_source, speaker_id, source_language='ur',
                          target_language='en', reference_voice_path=None,
                          output_audio_path=None, tts_steps=None, *, call_id=None,
                          segment_id=None, voice_profile=None):
        identity = SegmentIdentity(valid_id(call_id or uuid4().hex), valid_id(speaker_id),
                                   valid_id(str(segment_id or uuid4().hex)))
        started = time.perf_counter()
        metrics = {'stt_ms': 0., 'translation_ms': 0., 'tts_ms': 0.}
        diagnostic = dict(identity.__dict__, event='segment', source_language=source_language,
                          target_language=target_language, tts_language=target_language)
        def log(event, **fields):
            diagnostic.update(fields)
            logger.info(json.dumps(dict(diagnostic, event=event), ensure_ascii=False))
        try:
            if reference_voice_path is not None:
                raise PipelineError('Enroll references through VoiceProfiles; ad-hoc reference paths are not caller identities')
            if source_language not in {'en', 'de', 'es', 'fr', 'ur', None, 'auto'} or target_language not in {'en', 'de', 'es', 'fr', 'ur'}:
                raise PipelineError('Unsupported source/target language')
            info = sf.info(audio_source)
            segment = CapturedAudioSegment(identity, audio_source, info.samplerate, info.duration)
            log('captured', audio_duration=segment.duration, sample_rate=segment.sample_rate,
                channels=info.channels)
            tick = time.perf_counter()
            stt = self.stt.transcribe_detailed(audio_source, None if source_language == 'auto' else source_language)
            metrics['stt_ms'] = round((time.perf_counter()-tick)*1000, 1)
            source = stt.detected_language
            log('transcribed', detected_language=source, language_mode=stt.language_mode,
                stt_raw=stt.raw_text, stt_text=stt.text, stt_avg_logprob=stt.avg_logprob,
                stt_no_speech_probability=stt.no_speech_probability, metrics=metrics.copy())
            result = dict(identity.__dict__, source_language=source, target_language=target_language,
                          transcript=stt.text, translated_text='', output_audio_path=None,
                          metrics=metrics, status='no_speech', voice_state=VoiceState.UNINITIALIZED.value)
            if not stt.text:
                result['status'] = 'low_confidence' if stt.raw_text else 'no_speech'
                return result
            if source not in {'en', 'de', 'es', 'fr', 'ur'}:
                raise PipelineError(f'Unsupported detected language: {source}')
            tick = time.perf_counter()
            log('translation_request', translation_input=stt.text)
            raw, _ = self.translator.translate(stt.text, source_lang=source, target_lang=target_language)
            translated = raw.strip()
            metrics['translation_ms'] = round((time.perf_counter()-tick)*1000, 1)
            log('translated', raw_translation=raw, translated_text=translated, metrics=metrics.copy())
            if not translated:
                raise PipelineError('Translation returned empty text')
            translation = TranslationResult(identity, source, target_language, stt.text, translated)
            result['translated_text'] = translated
            # Reference-path shortcuts used by old demos are deliberately not enrollment.
            profile = voice_profile or self.profiles.get(speaker_id)
            if profile is None:
                result.update(status='clone_not_ready', error='Enroll your own voice before synthesis')
                log('synthesis_blocked', reason=result['error'], voice_state=result['voice_state'])
                return result
            if profile.speaker_id != speaker_id or profile.state != VoiceState.READY:
                raise PipelineError('Voice profile identity/state mismatch')
            request = SpeechSynthesisRequest(translation, target_language, profile)
            result.update(voice_state=profile.state.value if isinstance(profile.state, VoiceState) else profile.state,
                          voice_profile_id=profile.profile_id)
            log('synthesis_request', voice_profile_id=profile.profile_id,
                voice_state=result['voice_state'], reference_language=profile.reference_language)
            capabilities = {'en', 'de', 'es', 'fr', 'ur'} if self.tts_backend == 'omnivoice' else {'en', 'de', 'es', 'fr', 'zh'}
            supported = getattr(self.voice_clone, 'supported_languages', capabilities)
            ref_supported = getattr(self.voice_clone, 'supported_reference_languages', capabilities)
            if target_language not in supported or profile.reference_language not in ref_supported:
                result.update(status='unsupported_synthesis_language',
                    error=f'Configured TTS cannot synthesize {target_language} with a {profile.reference_language} reference; install an explicitly compatible multilingual cloning backend')
                log('synthesis_blocked', reason=result['error'])
                return result
            tick = time.perf_counter()
            try:
                if self.voice_clone is None:
                    if self.tts_backend == "omnivoice":
                        from multilingual_voice import OmniVoiceCloningService
                        self.voice_clone = OmniVoiceCloningService()
                    else:
                        from voice_clone import F5TTSMLXVoiceCloningService
                        self.voice_clone = F5TTSMLXVoiceCloningService(default_steps=self.default_steps)
                output = output_audio_path or str(ROOT / 'outputs' / f'{uuid4().hex}.wav')
                path, _ = self.voice_clone.clone_voice(text=request.translation.translated_text,
                    ref_audio_path=profile.audio_path, ref_text=profile.reference_text,
                    language=request.tts_language, output_path=output, steps=tts_steps or self.default_steps)
                from audio_utils import read_audio
                read_audio(path)  # Reject NaN/empty/invalid waveforms before delivery.
                info = sf.info(path)
                if info.frames == 0:
                    raise PipelineError('TTS returned empty audio')
                generated = TranslatedAudioResult(identity, target_language, path, info.duration)
                result.update(status='success', output_audio_path=generated.audio_path)
                log('synthesized', generated_audio_duration=generated.duration)
            except Exception as exc:
                result.update(status='synthesis_error', error=str(exc), voice_state=VoiceState.ERROR.value)
                log('synthesis_failed', reason=str(exc), voice_state=VoiceState.ERROR.value)
            finally:
                metrics['tts_ms'] = round((time.perf_counter()-tick)*1000, 1)
            return result
        except Exception as exc:
            log('failed', reason=str(exc))
            raise PipelineError(str(exc)) from exc
        finally:
            metrics['total_ms'] = round((time.perf_counter()-started)*1000, 1)
            log('finished', metrics=metrics)
