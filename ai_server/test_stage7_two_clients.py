"""Live model integration demo for the versioned two-client protocol.

Requires explicitly enrolled user_a/user_b and a working Urdu-capable backend.
A text-only/unsupported result is a failure, never a 'passed' voice test.
For deterministic protocol coverage run test_call_protocol.py instead.
"""
import base64
import json
from pathlib import Path
from fastapi.testclient import TestClient
from server import app, profiles


def receive(ws, kind):
    for _ in range(20):
        message = ws.receive_json()
        if message['type'] == 'error':
            raise AssertionError(message)
        if message['type'] == kind:
            return message
    raise AssertionError(f'Expected {kind}')


def run_two_client_call():
    for speaker in ('user_a','user_b'):
        if profiles.get(speaker) is None:
            raise RuntimeError(f'Enroll {speaker} through the app before this integration test')
    with TestClient(app) as client:
        with client.websocket_connect('/ws/call/user_a') as a, client.websocket_connect('/ws/call/user_b') as b:
            receive(a,'connected'); receive(b,'connected')
            a.send_json(dict(type='configure',source_language='ur',receive_language='ur'))
            b.send_json(dict(type='configure',source_language='en',receive_language='en'))
            a.send_json(dict(type='call_start',target_client='user_b'))
            call = receive(b,'incoming_call')['call_id']
            b.send_json(dict(type='call_accept',call_id=call))
            receive(a,'call_connected'); receive(b,'call_connected')
            root = Path(__file__).resolve().parent
            for sender, receiver, sample in [(a,b,'urdu_sentence.wav'),(b,a,'english_sentence.wav')]:
                sender.send_json(dict(type='utterance',call_id=call,segment_id=1,
                    audio_base64=base64.b64encode((root/'samples'/sample).read_bytes()).decode()))
                receive(sender,'processing_started')
                confirmation = receive(sender,'utterance_sent_confirmation')
                result = receive(receiver,'translated_utterance')
                print(json.dumps({k:v for k,v in result.items() if k != 'audio_base64'},ensure_ascii=False,indent=2))
                assert confirmation['status'] == result['status'] == 'success', result.get('error')
                assert result['audio_base64'], 'No generated speech'
            a.send_json(dict(type='call_end',call_id=call))
            receive(b,'call_terminated')
    print('Transport and synthesis smoke test passed; semantic and speaker similarity review still required.')


if __name__ == '__main__': run_two_client_call()
