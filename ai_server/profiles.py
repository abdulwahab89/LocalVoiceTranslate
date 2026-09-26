"""Explicit enrollment; demo WAV files are never silently treated as live voices."""
import json
import logging
import re
from pathlib import Path
from uuid import uuid4
import numpy as np
import soundfile as sf
from audio_utils import read_audio
from contracts import VoiceProfile, VoiceState


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', value):
        raise ValueError('IDs must contain 1–80 letters, digits, underscores or hyphens')
    return value


class VoiceProfiles:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.states = {}

    def _state(self, speaker, state):
        self.states[speaker] = state
        logging.getLogger(__name__).info(json.dumps(dict(event='voice_state',
            speaker_id=speaker, call_id=None, segment_id=None, state=state.value)))

    def get(self, speaker):
        speaker = valid_id(speaker)
        manifest = self.root / f'{speaker}.json'
        if not manifest.exists():
            return None
        data = json.loads(manifest.read_text())
        if data['speaker_id'] != speaker:
            raise ValueError('Voice profile identity mismatch')
        data['state'] = VoiceState(data['state'])
        return VoiceProfile(**data)

    def enroll(self, speaker, audio_path, text, language):
        speaker = valid_id(speaker)
        self._state(speaker, VoiceState.COLLECTING_REFERENCE_AUDIO)
        try:
            if not text or not text.strip():
                raise ValueError('Provide the exact words spoken in the reference recording')
            if language not in {'en', 'de', 'es', 'fr', 'ur', 'zh'}:
                raise ValueError('Unsupported reference language')
            audio = read_audio(audio_path, 24000)
            duration = len(audio)/24000
            if not 3 <= duration <= 15:
                raise ValueError('Record 3–15 seconds of clean reference speech (6–10 recommended)')
            frames = audio[:len(audio)//480*480].reshape(-1, 480)
            active_seconds = float((np.sqrt(np.mean(frames**2, axis=1)) > .005).sum()) * .02
            if active_seconds < 2:
                raise ValueError('Reference needs at least 2 seconds of audible speech, not padded silence')
            self._state(speaker, VoiceState.INITIALIZING)
            version = uuid4().hex
            path = self.root / f'{speaker}_{version}.wav'
            sf.write(path, audio, 24000, subtype='PCM_16')
            # F5 conditions on the waveform and exact transcript, not a stored embedding.
            profile = VoiceProfile(speaker, version, str(path.resolve()), text.strip(), language)
            data = dict(profile.__dict__)
            data['state'] = profile.state.value
            pending = self.root / f'{speaker}_{version}.json.tmp'
            pending.write_text(json.dumps(data, ensure_ascii=False))
            pending.replace(self.root / f'{speaker}.json')
            self._state(speaker, VoiceState.READY)
            return profile
        except Exception:
            self._state(speaker, VoiceState.ERROR)
            raise
