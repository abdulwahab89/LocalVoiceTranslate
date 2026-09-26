"""Exercise the real WebSocket protocol without model weights."""
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import server
from contracts import VoiceProfile


def until(ws, kind):
    for _ in range(12):
        data = ws.receive_json()
        if data['type'] == kind:
            return data
    raise AssertionError(f'Missing message {kind}')


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        server.connections.clear()
        server.calls.clear()
        self.seen = []
        async def fake_infer(**kw):
            self.seen.append(kw)
            return dict(call_id=kw['call_id'], speaker_id=kw['speaker_id'], segment_id=kw['segment_id'],
                        source_language=kw['source_language'], target_language=kw['target_language'],
                        transcript='source', translated_text='translated', status='success', metrics={}, output_audio_path=None)
        self.patches = [patch.object(server, 'get_pipeline', return_value=object()),
                        patch.object(server, 'infer', fake_infer),
                        patch.object(server.profiles, 'get', side_effect=lambda name: VoiceProfile(name,name+'_clone','/unused','ref','en'))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_alternating_participants_duplicate_and_stale_segments(self):
        with TestClient(server.app) as client:
            with client.websocket_connect('/ws/call/alice') as a, client.websocket_connect('/ws/call/bob') as b:
                until(a, 'connected'); until(b, 'connected')
                a.send_json(dict(type='configure',source_language='ur',receive_language='ur'))
                b.send_json(dict(type='configure',source_language='en',receive_language='en'))
                a.send_json(dict(type='call_start',target_client='bob'))
                call_id = until(b, 'incoming_call')['call_id']
                b.send_json(dict(type='call_accept',call_id=call_id))
                until(a, 'call_connected'); until(b, 'call_connected')
                for ws, peer, participant, number in [(a,b,'alice',1),(b,a,'bob',1),(a,b,'alice',2)]:
                    ws.send_json(dict(type='utterance',call_id=call_id,segment_id=number,
                                      audio_base64='d2F2',target_language='invalid',target_client='mallory'))
                    delivered = until(peer, 'translated_utterance')
                    until(ws, 'utterance_sent_confirmation')
                    self.assertEqual(delivered['sender_id'],participant)
                    self.assertEqual(delivered['target_language'],'en' if participant=='alice' else 'ur')
                    self.assertEqual(self.seen[-1]['voice_profile'].speaker_id,participant)
                a.send_json(dict(type='utterance',call_id=call_id,segment_id=2,audio_base64='d2F2'))
                self.assertIn('Duplicate',until(a,'error')['message'])
                a.send_json(dict(type='utterance',call_id='old_call',segment_id=3,audio_base64='d2F2'))
                self.assertIn('active call',until(a,'error')['message'])
                self.assertEqual(len(self.seen),3)
                a.send_json(dict(type='call_end',call_id=call_id))
                until(b,'call_terminated')

    def test_duplicate_socket_cannot_replace_participant(self):
        with TestClient(server.app) as client:
            with client.websocket_connect('/ws/call/alice') as a:
                until(a,'connected')
                with client.websocket_connect('/ws/call/alice') as duplicate:
                    self.assertIn('already connected',until(duplicate,'error')['message'])
                self.assertIn('alice',server.connections)


if __name__ == '__main__': unittest.main()
