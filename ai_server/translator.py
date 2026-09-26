"""
Translation Service Module for Translation Call Demo.

Supports bidirectional translation:
- Urdu (ur) -> English (en)
- English (en) -> Urdu (ur)

Uses local MarianMT neural machine translation models running offline on Apple Silicon.
"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, Optional, Tuple

import torch
from transformers import MarianMTModel, MarianTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class TranslationError(Exception):
    """Base exception for translation failures."""
    pass


class TranslationService(ABC):
    """Abstract interface for Translation services."""

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str = "ur",
        target_lang: str = "en",
    ) -> Tuple[str, float]:
        """
        Translate text from source language to target language.

        Args:
            text: Text string to translate.
            source_lang: Language code ('ur' or 'en').
            target_lang: Language code ('en' or 'ur').

        Returns:
            Tuple of (translated_text, latency_seconds).
        """
        pass


class MarianTranslationService(TranslationService):
    """
    Local neural translation service using Helsinki-NLP/opus-mt models.
    Runs on CPU for ultra-low latency (~130ms) without Metal graph compilation overhead.
    """

    MODEL_NAMES = {
        ("ur", "en"): "Helsinki-NLP/opus-mt-ur-en",
        ("en", "ur"): "Helsinki-NLP/opus-mt-en-ur",
    }

    def __init__(self, device: Optional[str] = "cpu"):
        self.device = device or "cpu"
        self._tokenizers: Dict[Tuple[str, str], MarianTokenizer] = {}
        self._models: Dict[Tuple[str, str], MarianMTModel] = {}

        logger.info(f"Initializing MarianTranslationService on {self.device}...")
        for pair, model_id in self.MODEL_NAMES.items():
            t0 = time.time()
            try:
                # Attempt offline load first; if not found, let it download once
                try:
                    tok = MarianTokenizer.from_pretrained(model_id, local_files_only=True)
                    mod = MarianMTModel.from_pretrained(model_id, local_files_only=True).to(self.device)
                except Exception:
                    tok = MarianTokenizer.from_pretrained(model_id)
                    mod = MarianMTModel.from_pretrained(model_id).to(self.device)

                mod.eval()
                self._tokenizers[pair] = tok
                self._models[pair] = mod
                logger.info(f"Loaded translation model for {pair[0]} -> {pair[1]} in {time.time() - t0:.2f}s")
            except Exception as e:
                logger.error(f"Failed to load translation model {model_id}: {e}")
                raise TranslationError(f"Failed to load model {model_id}: {e}") from e

        # Warm up both models with dummy inference
        self._warmup()

    def _warmup(self):
        """Warm up model inference caches."""
        try:
            for pair in self.MODEL_NAMES:
                tok = self._tokenizers[pair]
                mod = self._models[pair]
                sample_text = "سلام" if pair[0] == "ur" else "Hello"
                batch = tok([sample_text], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    mod.generate(**batch)
            logger.info("Translation models warmed up successfully.")
        except Exception as e:
            logger.warning(f"Translation warmup warning: {e}")

    def translate(
        self,
        text: str,
        source_lang: str = "ur",
        target_lang: str = "en",
    ) -> Tuple[str, float]:
        """Translates text between supported language pairs."""
        if not text or not text.strip():
            return "", 0.0

        pair = (source_lang.lower(), target_lang.lower())
        if pair[0] == pair[1] and pair[0] in {"en", "ur"}:
            return text.strip(), 0.0
        if pair not in self.MODEL_NAMES:
            raise TranslationError(
                f"Unsupported language pair: '{source_lang}' -> '{target_lang}'. "
                f"Supported pairs: {list(self.MODEL_NAMES.keys())}"
            )

        tok = self._tokenizers[pair]
        mod = self._models[pair]

        t0 = time.time()
        try:
            batch = tok([text.strip()], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                generated_tokens = mod.generate(**batch)
            translated = tok.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            elapsed = time.time() - t0

            logger.info(
                f"Translated ({pair[0]}->{pair[1]}) in {elapsed * 1000:.1f}ms: "
                f"'{text}' -> '{translated}'"
            )
            return translated, elapsed
        except Exception as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            raise TranslationError(f"Translation failed: {e}") from e


class NLLBTranslationService(TranslationService):
    """Sentence-level translation with explicit language tokens and no history."""
    CODES = {
        'en': 'eng_Latn',
        'de': 'deu_Latn',
        'ur': 'urd_Arab',
        'es': 'spa_Latn',
        'fr': 'fra_Latn',
    }
    SNAPSHOT_PATH = "/Users/mac/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"

    def __init__(self, model_name=None):
        import os
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        target = model_name or (self.SNAPSHOT_PATH if os.path.exists(self.SNAPSHOT_PATH) else 'facebook/nllb-200-distilled-600M')
        # Separate tokenizers avoid mutable src_lang leaking between participants.
        self.tokenizers = {lang: AutoTokenizer.from_pretrained(target, src_lang=code) for lang, code in self.CODES.items()}
        self.model = AutoModelForSeq2SeqLM.from_pretrained(target).eval()

    def translate(self, text, source_lang='de', target_lang='en'):
        import re
        start = time.perf_counter()
        if source_lang not in self.CODES or target_lang not in self.CODES:
            raise TranslationError(f'Unsupported NLLB language pair: {source_lang} -> {target_lang}')
        if not text.strip():
            raise TranslationError('Empty translation input')
        if source_lang == target_lang:
            return text.strip(), 0.
        tokenizer = self.tokenizers[source_lang]
        # NLLB is a sentence translation model. Preserve sentence order explicitly.
        sentences = re.split(r'(?<=[.!?۔؟])\s+', text.strip())
        outputs = []
        for sentence in sentences:
            tokens = tokenizer(sentence, return_tensors='pt', truncation=False)
            if tokens['input_ids'].shape[1] > 512:
                raise TranslationError('Sentence exceeds model context; input was not truncated')
            with torch.inference_mode():
                ids = self.model.generate(
                    **tokens,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(self.CODES[target_lang]),
                    max_new_tokens=256,
                    num_beams=1,
                    do_sample=False,
                )
            if ids.shape[1] >= 512:
                raise TranslationError('Translation reached output limit')
            output = tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if not output:
                raise TranslationError('Translation returned an empty sentence')
            outputs.append(output)
        return ' '.join(outputs), time.perf_counter()-start


def create_translation_service():
    import os
    backend = os.environ.get('TRANSLATION_BACKEND', 'nllb')
    if backend == 'nllb':
        return NLLBTranslationService()
    if backend == 'marian':
        logger.warning('Marian is a diagnostic baseline; known Urdu/English semantic failures')
        return MarianTranslationService()
    raise TranslationError(f'Unknown translation backend: {backend}')
