import unittest
from collections import OrderedDict
from pathlib import Path
import tempfile
import numpy as np
from omnivoice_worker import synthesize


class MultilingualAdapterTests(unittest.TestCase):
    def test_target_language_and_content_keyed_reference_cache(self):
        class Model:
            sampling_rate = 24000
            def __init__(self): self.prompts, self.requests = [], []
            def create_voice_clone_prompt(self, **kw):
                prompt = object()
                self.prompts.append((kw,prompt))
                return prompt
            def generate(self, **kw):
                self.requests.append(kw)
                return [np.ones(2400)*.1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root/'a.wav', root/'b.wav'
            a.write_bytes(b'speaker a audio'); b.write_bytes(b'speaker b audio')
            model, cache = Model(), OrderedDict()
            request = dict(language='ur',text='میرا نام امجد ہے',reference_audio=str(a),
                           reference_text='original A',output_path=str(root/'out.wav'))
            synthesize(model,cache,request)
            synthesize(model,cache,dict(request,language='en',text='My name is Amjad',reference_audio=str(b),reference_text='original B'))
            synthesize(model,cache,request)
            self.assertEqual(len(model.prompts),2)
            self.assertEqual([r['language'] for r in model.requests],['ur','en','ur'])
            self.assertIs(model.requests[0]['voice_clone_prompt'],model.requests[2]['voice_clone_prompt'])
            self.assertIsNot(model.requests[0]['voice_clone_prompt'],model.requests[1]['voice_clone_prompt'])
            self.assertEqual(model.requests[0]['text'],'میرا نام امجد ہے')
            a.write_bytes(b'new reference same path')
            synthesize(model,cache,request)
            self.assertEqual(len(model.prompts),3)


if __name__ == '__main__': unittest.main()
