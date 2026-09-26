"""Local two-party call server: explicit enrollment, bounded work, stale-result rejection."""
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pipeline import TranslationCallPipeline
from profiles import VoiceProfiles, valid_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('server')
ROOT = Path(__file__).resolve().parent
SAMPLES_DIR, OUTPUTS_DIR = ROOT / 'samples', ROOT / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
profiles = VoiceProfiles(ROOT / 'profiles')
app = FastAPI(title='Local translated calls')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
# All MLX/model access occurs on one dedicated thread, including construction.
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='local-inference')
_pipeline: Optional[TranslationCallPipeline] = None
capacity = asyncio.Semaphore(4)
MAX_UPLOAD = 4 * 1024 * 1024


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = TranslationCallPipeline(profiles=profiles)
    return _pipeline


async def infer(**kwargs):
    queued = time.perf_counter()
    def execute():
        queue_ms = round((time.perf_counter()-queued)*1000, 1)
        result = get_pipeline().process_utterance(**kwargs)
        result['metrics']['queue_ms'] = queue_ms
        result['metrics']['end_to_end_ms'] = round((time.perf_counter()-queued)*1000, 1)
        logger.info(json.dumps(dict(event='delivery_ready', call_id=result['call_id'],
            speaker_id=result['speaker_id'], segment_id=result['segment_id'], metrics=result['metrics'])))
        return result
    async with capacity:
        return await asyncio.get_running_loop().run_in_executor(worker, execute)


@app.on_event('startup')
async def startup():
    try:
        await asyncio.get_running_loop().run_in_executor(worker, get_pipeline)
    except Exception:
        logger.exception('Model initialization failed; health will report not ready')


@app.get('/api/health')
async def health():
    languages = ['en', 'ur'] if os.environ.get('TTS_BACKEND') == 'omnivoice' else ['en', 'zh']
    return dict(status='online', device='local', platform='Apple Silicon',
                pipeline_ready=_pipeline is not None, tts_languages=languages,
                tts_reference_languages=languages, enrollment_required=True)


@app.get('/api/speakers')
async def speakers():
    return {'speakers': [dict(speaker_id=p.stem, state='READY',
                             profile_id=profiles.get(p.stem).profile_id)
                         for p in profiles.root.glob('*.json')]}


@app.post('/api/enroll-voice')
async def enroll_voice(speaker_id: str = Form(...), audio: UploadFile = File(...),
                       ref_text: str = Form(...), reference_language: str = Form('en')):
    path = OUTPUTS_DIR / f'enroll_{uuid4().hex}.wav'
    try:
        valid_id(speaker_id)
        data = await audio.read(MAX_UPLOAD+1)
        if len(data) > MAX_UPLOAD:
            raise ValueError('Reference upload too large')
        path.write_bytes(data)
        # Serialize publication with inference; manifests atomically select immutable versions.
        profile = await asyncio.get_running_loop().run_in_executor(worker,
            partial(profiles.enroll, speaker_id, str(path), ref_text, reference_language))
        return dict(status='success', speaker_id=speaker_id, state=profile.state,
                    profile_id=profile.profile_id, sample_rate=24000)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


def with_audio(result):
    result = dict(result)
    path = result.get('output_audio_path')
    result['audio_base64'] = base64.b64encode(Path(path).read_bytes()).decode() if path else None
    result['audio_file'] = Path(path).name if path else None
    return result


@app.post('/api/translate-voice')
async def translate_voice(audio: UploadFile = File(...), speaker_id: str = Form(...),
                          source_language: str = Form('de'), target_language: str = Form('en')):
    path = OUTPUTS_DIR / f'input_{uuid4().hex}.wav'
    try:
        valid_id(speaker_id)
        data = await audio.read(MAX_UPLOAD+1)
        if not data or len(data) > MAX_UPLOAD:
            raise ValueError('Empty or oversized audio upload')
        path.write_bytes(data)
        return with_audio(await infer(audio_source=str(path), speaker_id=speaker_id,
                            source_language=source_language, target_language=target_language))
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.get('/api/audio/{filename}')
async def audio_file(filename: str):
    return serve(OUTPUTS_DIR, filename)


@app.get('/api/sample-audio/{filename}')
async def sample_file(filename: str):
    return serve(SAMPLES_DIR, filename)


def serve(directory, filename):
    if Path(filename).name != filename or not filename.endswith('.wav'):
        raise HTTPException(400, 'Invalid audio filename')
    path = directory / filename
    if not path.is_file():
        raise HTTPException(404, 'Audio not found')
    return FileResponse(path, media_type='audio/wav')


@dataclass
class Connection:
    ws: WebSocket
    source: str = 'de'
    receive: str = 'en'
    call_id: Optional[str] = None
    busy: bool = False
    last_segment: int = 0
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send(self, payload):
        async with self.send_lock:
            await self.ws.send_json(payload)


@dataclass
class Call:
    call_id: str
    caller: str
    receiver: str
    accepted: bool = False

    def peer(self, participant):
        if participant not in (self.caller, self.receiver):
            raise ValueError('Participant is not in this call')
        return self.receiver if participant == self.caller else self.caller


connections = {}
calls = {}


async def presence():
    for conn in list(connections.values()):
        try:
            await conn.send(dict(type='presence_update', active_clients=list(connections)))
        except Exception:
            logger.exception('Presence delivery failed')


async def end_call(conn):
    call = calls.pop(conn.call_id, None)
    conn.call_id = None
    if call:
        for name in (call.caller, call.receiver):
            other = connections.get(name)
            if other and other.call_id == call.call_id:
                other.call_id = None
                try:
                    await other.send(dict(type='call_terminated', call_id=call.call_id))
                except Exception:
                    logger.exception('Call termination delivery failed')


async def process_segment(name, conn, call, segment_id, audio, profile):
    path = OUTPUTS_DIR / f'ws_{uuid4().hex}.wav'
    identity = dict(call_id=call.call_id, speaker_id=name, segment_id=str(segment_id))
    peer_name = call.peer(name)
    peer = connections[peer_name]
    source, target = conn.source, peer.receive
    try:
        path.write_bytes(audio)
        await conn.send(dict(type='processing_started', **identity))
        result = await infer(audio_source=str(path), source_language=source,
                             target_language=target, voice_profile=profile, **identity)
        # Ending/replacing a call or socket invalidates all in-flight output.
        if (calls.get(call.call_id) is not call or connections.get(name) is not conn
                or connections.get(peer_name) is not peer or peer.call_id != call.call_id):
            logger.info(json.dumps(dict(event='stale_result_discarded', **identity)))
            return
        payload = with_audio(result)
        await peer.send(dict(payload, type='translated_utterance', sender_id=name))
        await conn.send(dict(payload, type='utterance_sent_confirmation'))
    except Exception as exc:
        logger.exception('Segment failed: %s', identity)
        if connections.get(name) is conn and conn.call_id == call.call_id:
            await conn.send(dict(type='error', message=str(exc), **identity))
    finally:
        conn.busy = False
        path.unlink(missing_ok=True)


@app.websocket('/ws/call/{client_id}')
async def websocket_call_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        valid_id(client_id)
        if client_id in connections:
            raise ValueError('This participant is already connected')
    except ValueError as exc:
        await websocket.send_json(dict(type='error', message=str(exc)))
        await websocket.close(code=1008)
        return
    conn = Connection(websocket)
    connections[client_id] = conn
    tasks = set()
    await conn.send(dict(type='connected', client_id=client_id, active_clients=list(connections)))
    await presence()
    try:
        while True:
            raw = await websocket.receive_text()
            identity = dict(speaker_id=client_id, call_id=conn.call_id, segment_id=None)
            try:
                if len(raw) > MAX_UPLOAD * 2:
                    raise ValueError('Message too large')
                data = json.loads(raw)
                kind = data.get('type')
                identity['segment_id'] = data.get('segment_id')
                if kind == 'configure':
                    if conn.call_id and calls[conn.call_id].accepted:
                        raise ValueError('Language settings cannot change during a call')
                    source, receive = data.get('source_language'), data.get('receive_language')
                    if source not in {'en', 'de', 'es', 'fr', 'ur', 'auto'} or receive not in {'en', 'de', 'es', 'fr', 'ur'}:
                        raise ValueError('Unsupported language configuration')
                    conn.source, conn.receive = source, receive
                elif kind == 'call_start':
                    target = valid_id(data.get('target_client'))
                    peer = connections.get(target)
                    if not peer or peer is conn:
                        raise ValueError('Peer is not connected')
                    if conn.call_id:
                        continue  # Simultaneous starts join the already offered call.
                    if peer.call_id:
                        raise ValueError('Peer is already in a call')
                    call = Call(uuid4().hex, client_id, target)
                    calls[call.call_id] = call
                    conn.call_id = peer.call_id = call.call_id
                    conn.last_segment = peer.last_segment = 0
                    await conn.send(dict(type='call_pending', call_id=call.call_id))
                    await peer.send(dict(type='incoming_call', from_client=client_id, call_id=call.call_id))
                elif kind == 'call_accept':
                    call = calls.get(data.get('call_id'))
                    if not call or call.receiver != client_id or conn.call_id != call.call_id:
                        raise ValueError('Invalid call acceptance')
                    call.accepted = True
                    for name in (call.caller, call.receiver):
                        await connections[name].send(dict(type='call_connected', call_id=call.call_id,
                                                        peer_client=call.peer(name)))
                elif kind == 'call_end':
                    if data.get('call_id') != conn.call_id:
                        raise ValueError('Stale call end')
                    await end_call(conn)
                elif kind == 'utterance':
                    call = calls.get(data.get('call_id'))
                    if not call or not call.accepted or conn.call_id != call.call_id:
                        raise ValueError('Utterance does not belong to an active call')
                    segment_id = data.get('segment_id')
                    if type(segment_id) is not int or segment_id <= conn.last_segment:
                        raise ValueError('Duplicate/out-of-order segment ID')
                    if conn.busy:
                        raise ValueError('Backpressure: wait for the previous utterance')
                    audio = base64.b64decode(data.get('audio_base64', ''), validate=True)
                    if not audio or len(audio) > MAX_UPLOAD:
                        raise ValueError('Empty or oversized audio')
                    profile = profiles.get(client_id)
                    if profile is None:
                        raise ValueError('Clone UNINITIALIZED: enroll your voice before sending speech')
                    conn.last_segment = segment_id
                    conn.busy = True
                    task = asyncio.create_task(process_segment(client_id, conn, call, segment_id, audio, profile))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                else:
                    raise ValueError('Unknown message type')
            except Exception as exc:
                logger.warning(json.dumps(dict(event='request_rejected', reason=str(exc), **identity)))
                await conn.send(dict(type='error', message=str(exc), **identity))
    except WebSocketDisconnect:
        pass
    finally:
        await end_call(conn)
        if connections.get(client_id) is conn:
            del connections[client_id]
        # Inference is bounded and may finish; identity checks discard stale output.
        await presence()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
