"""Persistent local synthesis worker. Run with a separate Python >=3.10 environment."""
from collections import OrderedDict
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys


def synthesize(model, prompts, request):
    import numpy as np
    import soundfile as sf
    if request['language'] not in {'ur','en'}:
        raise ValueError('Unsupported output language')
    text, ref_text = request['text'].strip(), request['reference_text'].strip()
    if not text or not ref_text:
        raise ValueError('Target and reference texts are required')
    ref = Path(request['reference_audio'])
    key = hashlib.sha256(str(ref.resolve()).encode() + ref.read_bytes() + ref_text.encode()).hexdigest()
    if key not in prompts:
        prompts[key] = model.create_voice_clone_prompt(ref_audio=str(ref), ref_text=ref_text)
    prompts.move_to_end(key)
    while len(prompts) > 16:
        prompts.popitem(last=False)
    waves = model.generate(text=text, language=request['language'], voice_clone_prompt=prompts[key],
                           num_step=int(os.environ.get('OMNIVOICE_STEPS', '32')))
    audio = np.asarray(waves[0])
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError('Invalid OmniVoice output')
    Path(request['output_path']).parent.mkdir(parents=True, exist_ok=True)
    sf.write(request['output_path'], audio, model.sampling_rate)


def main():
    model, prompts = None, OrderedDict()
    for line in sys.stdin:
        request, response = {}, {}
        try:
            request = json.loads(line)
            # Keep library progress/debug output off the JSON transport channel.
            with redirect_stdout(sys.stderr):
                if model is None:
                    import torch
                    from omnivoice import OmniVoice
                    device = os.environ.get('OMNIVOICE_DEVICE', 'mps' if torch.backends.mps.is_available() else 'cpu')
                    model = OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map=device,
                        dtype=torch.float16 if device != 'cpu' else torch.float32, local_files_only=True)
                synthesize(model, prompts, request)
            response = {'ok': True}
        except Exception as exc:
            response = {'error': str(exc)}
        response['request_id'] = request.get('request_id')
        print(json.dumps(response), flush=True)


if __name__ == '__main__':
    main()
