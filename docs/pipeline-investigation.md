# Investigation — 2026-09-27

## Existing flow and reproduced failures (before edits)

`lib/main.dart`: fixed user_a/Urdu and user_b/English demo identities; button-down starts a WAV recorder (16 kHz mono), button-up stops it and uploads base64. No VAD, pre-roll, enrollment UI, call ID, segment ID, or playback queue. Recorder startup and button release can race; mute/processing/call state do not guard capture. Static UI claims enrollment without checking it.

`ai_server/server.py`: global participant-ID websocket map; synchronous model calls block the event loop; sender controls destination language; no call membership, duplicate connection protection, or stale-call validation. Enrollment writes original samples with a 24 kHz header, changing pitch/speed instead of resampling. Profiles are fixed sample filenames, not live callers.

`stt.py`: multilingual Whisper small, supplied language, previous-text conditioning already disabled. Reads any file sample rate into an array which Whisper interprets as 16 kHz; no mono conversion. Drops detected language and confidence metadata. No evidence of previous-utterance prompt leakage.

`translator.py`: independent Marian requests with no conversation history. Cached weights/tokenizers are not conversational state. Live reproduction with exact user text:
- Urdu → English: `My name is Cycle and I'm acting. Do you know me?` (445 ms).
- English → Urdu: `کیا آپ مجھے پہچان رہے ہیں ؟` (192 ms); first sentence omitted.

`pipeline.py`: source language supplied to STT and translation, reference selected by static speaker filename; Urdu transliterated by lossy character dictionary before F5. No output language parameter in TTS.

`voice_clone.py`: F5 conditioned on audio+transcript per synthesis, not a persistent speaker embedding. Missing transcript becomes unrelated Nature sample text. Four inference steps forced by server. No clone lifecycle or language capability check. The bundled user_a transcript is the Nature demo, so selecting user_a does not enroll the live microphone speaker.

Existing test_stage1/2/3/4/5/7 scripts are smoke demos; nonempty output is treated as success, not semantic/voice accuracy.

Baseline cached Whisper on real saved Amjad recording `outputs/ws_in_user_a_1790455066307.wav` (16 kHz mono, 8.992 s):
`میرا نام امجد ہے اور میں سلاحی کا کام کرتا ہوں کیا آپ مجھے پیچھانتے ہیں`
338 ms warm. Mostly recognizable, with consequential spelling errors. This file does not exhibit the sample-rate bug; that bug affects other input rates. Exact original bizarre English output has not yet been reproduced.

Existing English demo: `Where are you going?` (243 ms); Urdu demo: `آپ کیا کر رہے ہو؟` (1159 ms cold), differs from the test's claimed `تم کیا کر رہے ہو؟`.

## Design decisions

Keep utterance-based push-to-talk; no aggressive VAD is present to tune. Add continuous PCM pre-roll and bounded utterances without fragmenting on short pauses. Require explicit reference enrollment before synthesis; never promote arbitrary first-call speech or bundled fixtures to a caller's identity. A 3–15 second reference is an application validation policy, not proof of clone fidelity; recommend 6–10 clean seconds and its exact transcript.

Use immutable stage records, explicit target language, versioned enrolled profiles, per-call IDs and monotonic segment IDs, serialized model work off the event loop, bounded in-flight utterances, and stale-result checks. F5 language capability is separate from voice identity. Unsupported languages must produce text plus an explicit synthesis error, never transliteration/default voice.

## Implemented pipeline

Microphone stream → 300 ms PCM16/16 kHz mono pre-roll → complete push-to-talk utterance + 200 ms tail (25 s cap) → call/participant/monotonic segment validation → real resampling to Whisper mono 16 kHz → explicit transcription task, no prior-text conditioning or prompt → transcript/language/log-probability diagnostics → language-token-directed, sentence-wise translation → immutable enrolled voice version + explicit output language → supported cloning backend → same-call validation → ordered playback.

The receiver now selects their listening language. The server owns routing and does not trust a sender's target-language or target-client fields. Only one utterance per participant can be in flight. Model work runs on one dedicated executor thread, keeping websocket signaling responsive. Disconnect/end invalidates pending delivery, even if inference finishes later. Duplicate participant sockets and duplicate/out-of-order segment IDs are rejected.

Enrollment is explicit before a call; the app records an 8-second English phrase. Reference files are resampled correctly, versioned, and published atomically with their exact transcript. Old fixture WAVs are not auto-enrolled. Reference preparation states are UNINITIALIZED → COLLECTING_REFERENCE_AUDIO → INITIALIZING → READY or ERROR. READY means validated reference conditioning is available, not a claim of perceptual speaker similarity. F5 has no independent speaker embedding initialization; waveform and transcript condition each generation. Failed synthesis is explicit and never falls back to another voice.

Useful JSON logs retain IDs, input duration/rate/channels, configured-versus-detected language provenance, raw/accepted STT text, average log probability (not a calibrated confidence percentage), no-speech probability, translation input/raw/normalized output, target TTS language, immutable profile ID, generated duration, per-stage and total timings, and failure reasons. Transcripts contain call content; this is a local diagnostic demo.

## Run and reproduce

From repository root:

```sh
# Core environment (existing Python 3.9 MLX environment)
ai_server/venv/bin/pip install -r ai_server/requirements.txt
# Fetch the replacement translator once. Inference loads it offline.
ai_server/venv/bin/python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; m='facebook/nllb-200-distilled-600M'; AutoTokenizer.from_pretrained(m); AutoModelForSeq2SeqLM.from_pretrained(m)"
PYTHONPATH=ai_server ai_server/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
# Use the app's Enroll my voice button separately for each participant.

PYTHONPATH=ai_server ai_server/venv/bin/python -m unittest ai_server/test_pipeline_contracts.py ai_server/test_call_protocol.py ai_server/test_multilingual_voice.py
flutter analyze
flutter test

PYTHONPATH=ai_server ai_server/venv/bin/python ai_server/evaluate_models.py
PYTHONPATH=ai_server ai_server/venv/bin/python ai_server/evaluate_voice.py --steps 8
```

`STT_MODEL` selects a multilingual MLX Whisper checkpoint. `TRANSLATION_BACKEND=nllb` is the default; `marian` is retained only for reproduction of the known poor baseline. Identical-language requests preserve text without translation. Stage 4/5/7 remain smoke demos, not semantic acceptance tests. Their success checks and websocket handshake have been updated; they require explicit enrollment. Stage 5 additionally requires RUN_MODEL_TESTS=1.

API protocol: connect `/ws/call/{participant}`, send `configure` with `source_language` and `receive_language`, then `call_start` with `target_client`. Accept `incoming_call` with its server-generated `call_id`. Send utterances only after `call_connected`, with that `call_id`, a strictly increasing integer `segment_id`, and WAV `audio_base64`. End with the same `call_id`. Server payloads carry all three identity fields and an explicit `status`. `clone_not_ready`, `low_confidence`, `no_speech`, `unsupported_synthesis_language`, and `synthesis_error` must not be displayed as successful spoken output.

## Multilingual cloning component change

Default F5 supports English/Chinese conditioning, not Urdu. The lossy Urdu-to-Roman dictionary and canned reference transcript are removed. Unsupported target/reference language returns translated text plus an explicit synthesis status, with no audio.

Optional **OmniVoice** integration runs in a separate persistent local process. Its documented language list includes Urdu (`ur`); it accepts `language`, native-script text, and a reusable voice prompt. The adapter caches by immutable reference path, audio content, and exact reference text, limits cache size, and validates response IDs. A timeout terminates the worker to prevent a late reply contaminating the next segment.

This component requires a separate Python >=3.10 environment and Transformers >=5.3, while the existing backend is Python 3.9 / Transformers 4.57.6. Do not upgrade the existing environment in place. Setup on a machine with Python 3.10+:

```sh
python3.11 -m venv ai_server/.venv-omnivoice
ai_server/.venv-omnivoice/bin/pip install -r ai_server/requirements-omnivoice.txt
# Download both the model and its audio tokenizer once in this environment.
ai_server/.venv-omnivoice/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('k2-fsa/OmniVoice'); snapshot_download('eustlb/higgs-audio-v2-tokenizer')"
export OMNIVOICE_PYTHON="$PWD/ai_server/.venv-omnivoice/bin/python"
export TTS_BACKEND=omnivoice
export HF_HUB_OFFLINE=1
PYTHONPATH=ai_server ai_server/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
```

The OmniVoice integration is contract-tested with a fake model; actual Urdu audio generation, pronunciation and identity quality are **not validated** in this environment. Upstream explicitly notes cross-language reference accents can carry over, and its audio tokenizer runs on CPU for MPS. Language coverage does not guarantee native pronunciation or real-time latency. Do not present this optional path as a verified cure.

Primary component references:
- [F5-TTS upstream](https://github.com/SWivid/F5-TTS)
- [NLLB model card](https://huggingface.co/facebook/nllb-200-distilled-600M): sentence-level research model, 512-token context, CC-BY-NC; not a production deployment endorsement.
- [MLX Whisper turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo)
- [OmniVoice API and Apple Silicon guidance](https://github.com/k2-fsa/OmniVoice)
- [OmniVoice language coverage](https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md)
- [OmniVoice dependencies](https://github.com/k2-fsa/OmniVoice/blob/master/pyproject.toml)

## Remaining platform limits

This is still an utterance-based, half-duplex demo; no streaming STT/TTS or acoustic full-duplex guarantee. Holding longer than 25 seconds forces a boundary; release at sentence boundaries. Headset/device tests are still needed to verify actual PCM rate, recording permission UX, playback echo handling and voice similarity. IDs isolate accidental state reuse, but this demo has no authenticated identity/authorization and is not a public multi-tenant service.

## Final measured results

| Check | Result |
|---|---|
| Deterministic backend regression suite | 15 tests passed: resampling/pitch, enrollment/versioning, missing clones/transcripts, alternating profile/language routing, low-confidence and empty stages, Whisper context options, path IDs, stale completion, real WebSocket handshake/duplicates, optional multilingual prompt cache |
| Flutter | `flutter analyze`: no issues; `flutter test`: all 3 passed |
| Android | `flutter build apk --debug`: passed (including final incremental build) |
| Existing English short demo | Whisper-small: `Where are you going?`, exact match |
| Full English requested sentence | Both small and turbo: exact transcript, normalized WER 0 on a **synthetic macOS Samantha** fixture, not a live caller recording |
| Saved real Urdu Amjad sentence | Small: normalized WER 0.20; turbo: 0.267 against supplied text; both retain Urdu but have consequential orthographic errors. Additional `کیا` and token spacing also contribute to WER; this is not a native-speaker audited reference |
| Correct Urdu text → Marian | `My name is Cycle and I'm acting. Do you know me?` — fails |
| Correct English text → Marian | Only `کیا آپ مجھے پہچان رہے ہیں ؟` — drops the first sentence, fails |
| Correct Urdu text → NLLB | `My name is Amjad and I work as a seamstress. Do you recognize me?` — preserves broad meaning but introduces gender/occupation wording error |
| Correct English text → NLLB | `میرا نام امجد ہے اور میں ایک ٹیلر کے طور پر کام کرتا ہوں. کیا آپ مجھے پہچانتے ہیں؟` — retains both sentences/name, uses transliterated “tailor” rather than native `درزی` |
| Real Urdu → small STT → NLLB | `My name is Amjad and I work as an armourer. Do you know me?` — **still fails occupation**, caused by upstream `سلاحی` transcription |
| Real Urdu → turbo STT → NLLB | `My name is Amjad and I work as a tailor.` — **still omits the question** from the unpunctuated STT output |
| Repeated/alternating model requests | Repeated Urdu STT and translation outputs identical after English/weather requests; no observed prior-utterance contamination |
| Same-language text handling | Urdu→Urdu and English→English preserve input text exactly |
| First English cloned synthesis, two distinct fixture profiles | Both first requests returned audio transcribed as `Where are you going?`; correct profile IDs and reference inputs verified. **Perceptual identity/timbre not established** |
| F5 8 RK4 steps | About 8.9 s cold / 6.0 s warm synthesis for ~1.6 s output; not natural real-time calling |
| F5 32 RK4 steps experiment | About 34.7 s cold / 24.1 s warm; rejected as the default due to latency. Default now 8 (upstream library setting), no server-forced 4-step override |
| Urdu cloned speech/pronunciation | F5 rejected explicitly. Optional OmniVoice adapter contract tests pass; real model/dependency runtime and listening validation **not run** |

Evidence: [small + Marian](model-evaluation-small-marian-full.json), [small + NLLB (final default)](model-evaluation-small-nllb.json), [turbo + NLLB](model-evaluation-turbo-nllb.json), [8-step voice checks](voice-evaluation-8-steps.json), [32-step experiment](voice-evaluation.json).

Whisper-turbo was evaluated but **not made the default**: it was slower and did not resolve this Urdu example. Small remains configurable, and the Urdu accuracy acceptance requirement is unmet. NLLB is a better local diagnostic/default translator on clean input, not a guarantee of correct semantics. Its warm CPU translation times here were roughly 0.9–3 seconds; neither this nor F5 meets the desired whole-call latency.

Recommended next model evaluation is an Urdu-specialized ASR checkpoint (or full multilingual Whisper-large-v3) on a native-speaker transcribed corpus, followed by a stronger Urdu/English translator tested on unpunctuated speech. Do not add a sentence-specific spelling dictionary or infer “tailor” from the example. OmniVoice is the implemented optional Urdu-capable synthesis candidate; its accent/timbre and CPU-tokenizer latency must be measured before enabling it for callers.

## Files changed in this work

- Core: `ai_server/pipeline.py`, `stt.py`, `translator.py`, `voice_clone.py`, `server.py`.
- New contracts/utilities: `ai_server/contracts.py`, `audio_utils.py`, `profiles.py`.
- Optional multilingual backend: `ai_server/multilingual_voice.py`, `omnivoice_worker.py`, `requirements-omnivoice.txt`.
- Client: `lib/main.dart`, new `lib/audio_capture.dart`.
- Regression tests: `ai_server/test_pipeline_contracts.py`, `test_call_protocol.py`, `test_multilingual_voice.py`, `test/audio_capture_test.dart`.
- Reproducible model checks: `ai_server/evaluate_models.py`, `evaluate_voice.py`; compatibility/success-check updates to `test_stage4.py`, `test_stage5.py`, `test_stage7_two_clients.py`.
- Setup/reporting: `ai_server/requirements.txt`, `.gitignore`, `README.md`, this report and `docs/*evaluation*.json`.
- Generated evaluation audio in `ai_server/outputs/evaluation_*.wav` (synthetic English fixture and two explicit demo-reference outputs).

Pre-existing platform/config edits and saved caller audio were preserved. No commit, deployment, or restart of the user's existing server/client processes was performed.
