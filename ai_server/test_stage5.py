"""Optional real-model REST integration check, separate from deterministic tests."""
import os
from pathlib import Path
from unittest import SkipTest
from fastapi.testclient import TestClient
from server import app, profiles


def test_api():
    if os.environ.get('RUN_MODEL_TESTS') != '1':
        raise SkipTest('Set RUN_MODEL_TESTS=1 after explicitly enrolling user_a')
    assert profiles.get('user_a'), 'Enroll user_a through the app first'
    with TestClient(app) as client:
        assert client.get('/api/health').json()['pipeline_ready']
        assert client.get('/api/speakers').status_code == 200
        with (Path(__file__).resolve().parent/'samples/urdu_sentence.wav').open('rb') as audio:
            response = client.post('/api/translate-voice',files={'audio':('input.wav',audio,'audio/wav')},
                data=dict(speaker_id='user_a',source_language='ur',target_language='en'))
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['status'] == 'success', result.get('error', result['status'])
        assert result['audio_base64']
        print('REST synthesis smoke test passed; this is not an accuracy assertion.')


if __name__ == '__main__': test_api()
