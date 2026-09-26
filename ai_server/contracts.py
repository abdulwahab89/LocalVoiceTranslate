"""Immutable identities and stage values; never stored as global 'last result'."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class VoiceState(str, Enum):
    UNINITIALIZED = 'UNINITIALIZED'
    COLLECTING_REFERENCE_AUDIO = 'COLLECTING_REFERENCE_AUDIO'
    INITIALIZING = 'INITIALIZING'
    READY = 'READY'
    ERROR = 'ERROR'


@dataclass(frozen=True)
class SegmentIdentity:
    call_id: str
    speaker_id: str
    segment_id: str


@dataclass(frozen=True)
class CapturedAudioSegment:
    identity: SegmentIdentity
    audio_path: str
    sample_rate: int
    duration: float


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    detected_language: Optional[str]
    language_mode: str
    avg_logprob: Optional[float]
    no_speech_probability: Optional[float]
    elapsed_seconds: float
    raw_text: str = ''


@dataclass(frozen=True)
class TranslationResult:
    identity: SegmentIdentity
    source_language: str
    target_language: str
    source_text: str
    translated_text: str


@dataclass(frozen=True)
class VoiceProfile:
    speaker_id: str
    profile_id: str
    audio_path: str
    reference_text: str
    reference_language: str
    state: VoiceState = VoiceState.READY


@dataclass(frozen=True)
class SpeechSynthesisRequest:
    translation: TranslationResult
    tts_language: str
    voice_profile: VoiceProfile


@dataclass(frozen=True)
class TranslatedAudioResult:
    identity: SegmentIdentity
    language: str
    audio_path: str
    duration: float
