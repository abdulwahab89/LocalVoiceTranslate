"""Optional local OmniVoice adapter, isolated from the legacy Python/Transformers env.

OMNIVOICE_PYTHON points to a Python >=3.10 environment with requirements-omnivoice.txt.
No network API, random voice, transliteration, or fallback is used.
"""
import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import time
from uuid import uuid4


class OmniVoiceCloningService:
    supported_languages = frozenset({'ur', 'en'})
    supported_reference_languages = supported_languages

    def __init__(self):
        python = os.environ.get('OMNIVOICE_PYTHON')
        if not python or not Path(python).is_file():
            raise RuntimeError('Set OMNIVOICE_PYTHON to the separate Python >=3.10 OmniVoice environment; see docs/pipeline-investigation.md')
        self.process = subprocess.Popen([python, '-u', str(Path(__file__).with_name('omnivoice_worker.py'))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            env=dict(os.environ, HF_HUB_OFFLINE='1'))
        atexit.register(self.close)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()

    def clone_voice(self, text, ref_audio_path, ref_text=None, output_path=None, steps=None, language='en'):
        if language not in self.supported_languages or not text.strip() or not ref_text:
            raise ValueError('OmniVoice requires supported output language, text, and exact reference transcript')
        if self.process.poll() is not None:
            raise RuntimeError('OmniVoice worker exited; restart backend after checking worker diagnostics')
        request_id = uuid4().hex
        request = dict(request_id=request_id, language=language, text=text,
                       reference_audio=str(Path(ref_audio_path).resolve()), reference_text=ref_text,
                       output_path=str(Path(output_path).resolve()))
        start = time.perf_counter()
        self.process.stdin.write(json.dumps(request, ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 180)[0]:
            self.close()  # Never allow a late response to be associated with the next request.
            raise RuntimeError('OmniVoice synthesis timed out; no fallback was generated')
        response = json.loads(self.process.stdout.readline())
        if response.get('request_id') != request_id:
            self.close()
            raise RuntimeError('OmniVoice response identity mismatch')
        if response.get('error'):
            raise RuntimeError(response['error'])
        return request['output_path'], time.perf_counter()-start
