# Real-Time Translated Voice Calling with Consent-Based Zero-Shot Voice Cloning 🎙️

<img width="2606" height="1604" alt="Real-Time Translated Voice Calling Demo" src="https://github.com/user-attachments/assets/10b9c7c1-dd2e-4490-8d04-54e7575128d6" />

A fully local proof-of-concept demonstrating **bidirectional translated voice calling with consent-based zero-shot voice cloning**, optimized for **Apple Silicon (M1/M2/M3/M4)**.

The system allows two users who speak different languages to communicate while preserving the identity of the original speaker's voice:

- **User A speaks Urdu** → User B hears **English in User A's cloned voice**.
- **User B speaks English** → User A hears **Urdu in User B's cloned voice**.
- **100% Local** → No cloud AI or paid APIs are required.
- **Zero-Shot Voice Cloning** → A short reference recording is enough; no speaker-specific model training is required.
- **Apple Silicon Acceleration** → MLX + Metal acceleration; no CUDA/NVIDIA GPU required.
- **Utterance-Based / Near-Real-Time** → Speak → Release/Pause → STT → Translation → Voice-Cloned TTS → Remote Playback.

> **Note:** The current implementation is utterance-based rather than continuous streaming translation. The receiving user hears the translated speech after the current utterance has been captured and processed.

---

## How It Works

The application separates translated calling into three independent AI stages:

```text
Speech
  │
  ▼
┌──────────────────────┐
│ 1. Speech-to-Text    │
│    MLX Whisper       │
└──────────┬───────────┘
           │
           │ Source-language text
           ▼
┌──────────────────────┐
│ 2. Translation       │
│    MarianMT          │
└──────────┬───────────┘
           │
           │ Translated text
           ▼
┌──────────────────────┐
│ 3. Voice-Cloned TTS  │
│    F5-TTS + Vocos    │
└──────────┬───────────┘
           │
           ▼
Translated speech
in the original
speaker's cloned voice
```

Each stage has a single responsibility:

1. **Whisper** determines what the user said.
2. **MarianMT** translates the recognized text.
3. **F5-TTS** synthesizes the translation using the speaker's enrolled reference voice.
4. **Vocos** produces the final audio waveform for playback.

This separation also makes debugging easier. A bad final result may originate from STT, translation, or TTS independently.

---

# Architecture Overview

## User A → User B

```text
User A (Flutter)
Urdu Speaker
      │
      │ "تم کیا کر رہے ہو؟"
      │
      │ WebSocket Audio
      ▼
┌─────────────────────────────────────────┐
│          Local AI Backend               │
│                                         │
│  1. STT — MLX Whisper                  │
│                                         │
│     Audio                               │
│       ↓                                 │
│     "تم کیا کر رہے ہو؟"                 │
│                                         │
│  2. Translation — MarianMT             │
│                                         │
│     Urdu                                │
│       ↓                                 │
│     English                             │
│       ↓                                 │
│     "What are you doing?"               │
│                                         │
│  3. Zero-Shot TTS — F5-TTS MLX        │
│                                         │
│     "What are you doing?"               │
│              +                          │
│     User A Reference Voice              │
│              ↓                          │
│     Generated English Speech            │
│     resembling User A's voice           │
└──────────────────┬──────────────────────┘
                   │
                   │ WebSocket Audio
                   ▼
              User B (Flutter)

        Hears in English:

        "What are you doing?"

        in User A's cloned voice
```

## User B → User A

```text
User B (Flutter)
English Speaker
      │
      │ "Where are you going?"
      │
      │ WebSocket Audio
      ▼
┌─────────────────────────────────────────┐
│          Local AI Backend               │
│                                         │
│  1. STT — MLX Whisper                  │
│                                         │
│     "Where are you going?"              │
│              ↓                          │
│                                         │
│  2. Translation — MarianMT             │
│                                         │
│     English → Urdu                      │
│              ↓                          │
│     "تم کہاں جا رہے ہو؟"                │
│              ↓                          │
│                                         │
│  3. Zero-Shot TTS — F5-TTS MLX        │
│                                         │
│     Urdu translation                    │
│              +                          │
│     User B Reference Voice              │
│              ↓                          │
│     Generated Urdu Speech               │
│     resembling User B's voice           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
              User A (Flutter)

        Hears Urdu in
        User B's cloned voice
```

---

# Zero-Shot Voice Cloning

The project uses **F5-TTS MLX** for zero-shot voice cloning.

"Zero-shot" means that the application does **not train or fine-tune a new TTS model for every user**.

Instead, each user provides a short reference recording and an accurate transcript of that recording.

For example:

```text
Reference Audio:
user_a.wav

Reference Transcript:
"Hello, my name is Wahab and this is my reference recording."
```

Later, the translation system might produce completely different text:

```text
"What are you doing?"
```

F5-TTS receives:

```text
Reference Audio
       +
Reference Transcript
       +
New Translated Text
       │
       ▼
    F5-TTS
       │
       ▼
Generated Speech
       │
       ▼
"What are you doing?"

spoken using characteristics
of the reference voice
```

No speaker-specific model such as:

```text
user_a_model.pt
```

needs to be trained.

The same pretrained F5-TTS model can therefore synthesize speech for multiple enrolled speakers by changing the reference audio and transcript.

---

# Consent-Based Voice Enrollment

Voice cloning is **strictly consent-based** in this project.

Each participant must explicitly enroll their own voice before cloned speech can be generated for them.

A recommended enrollment recording is:

- **5–30 seconds** of clean speech
- One speaker only
- Minimal background noise
- No music
- Minimal echo
- Clear pronunciation
- 24 kHz mono WAV
- An accurate transcript matching the recording

For example:

```text
user_a.wav

Audio:
"Hello, my name is Wahab and this is my reference recording."
```

The corresponding transcript should contain exactly what was spoken:

```text
user_a.txt

Hello, my name is Wahab and this is my reference recording.
```

A mismatched reference transcript can reduce synthesis quality.

---

# Complete Processing Pipeline

When a user finishes speaking:

```text
Microphone
    │
    ▼
Recorded Utterance
    │
    ▼
Audio Conversion / Resampling
    │
    ▼
MLX Whisper
    │
    ▼
Source-Language Text
    │
    ▼
MarianMT
    │
    ▼
Translated Text
    │
    ▼
F5-TTS MLX
    │
    ├── Speaker Reference Audio
    │
    └── Speaker Reference Transcript
    │
    ▼
Vocos
    │
    ▼
Generated Waveform
    │
    ▼
WebSocket
    │
    ▼
Remote Client
    │
    ▼
Audio Playback
```

---

# AI Models

| Task | Model / Framework | Purpose |
| :--- | :--- | :--- |
| **Speech Recognition** | `mlx-community/whisper-tiny` | Speech → text |
| **Urdu → English Translation** | `Helsinki-NLP/opus-mt-ur-en` | Urdu text → English text |
| **English → Urdu Translation** | `Helsinki-NLP/opus-mt-en-ur` | English text → Urdu text |
| **Zero-Shot Voice-Cloned TTS** | `lucasnewman/f5-tts-mlx` | Translated text → cloned speech |
| **Vocoder** | `vocos-mlx` | Acoustic representation → waveform |

---

# Measured Performance & Latency

Measured locally on an **M1 MacBook Pro** after model warm-up:

| Pipeline Stage | Model & Framework | Compute Device | Typical Warm Latency |
| :--- | :--- | :--- | :--- |
| **STT (Urdu / English)** | `mlx-community/whisper-tiny` | Apple Silicon GPU / MLX | **180 ms – 1,450 ms** |
| **Translation** | MarianMT | CPU / Torch / SentencePiece | **130 ms – 180 ms** |
| **Voice-Cloned TTS** | `lucasnewman/f5-tts-mlx` + `vocos-mlx` | Apple Silicon GPU / Metal | **2,570 ms – 3,200 ms** |
| **Total AI Turnaround** | Full utterance pipeline | Apple Silicon Unified Memory | **~3.2 s – 4.0 s** |

The total above represents AI processing after the utterance has been captured.

Actual perceived conversational delay also includes:

```text
Speaking Duration
       +
Utterance Finalization
       +
STT
       +
Translation
       +
Voice-Cloned TTS
       +
Network Transfer
       +
Playback Startup
```

At present, **F5-TTS is the largest contributor to AI processing latency**.

---

# Near-Real-Time vs. Fully Streaming

The current implementation is **utterance-based**:

```text
Hold Push-to-Talk
       ↓
Speak
       ↓
Release
       ↓
Speech-to-Text
       ↓
Translation
       ↓
Voice-Cloned TTS
       ↓
Remote Playback
```

The receiving participant therefore begins hearing the translation **after the sender finishes the utterance**.

A future fully streaming implementation would instead look approximately like:

```text
Continuous Microphone Audio
          ↓
Streaming STT
          ↓
Partial Transcripts
          ↓
Incremental Translation
          ↓
Streaming TTS
          ↓
Continuous Remote Playback
```

Continuous streaming translation is outside the scope of the current proof-of-concept.

---

# Project Structure

```text
translation_call_demo/
├── lib/
│   └── main.dart
│       # Flutter application (Home Screen & Call Screen)
│
├── macos/
│   # macOS desktop runner and audio/network entitlements
│
├── ai_server/
│   ├── server.py
│   │   # FastAPI REST API & WebSocket call hub
│   │
│   ├── stt.py
│   │   # Speech-to-Text service & MLX Whisper implementation
│   │
│   ├── translator.py
│   │   # Urdu ↔ English MarianMT translation
│   │
│   ├── voice_clone.py
│   │   # F5-TTS MLX zero-shot voice cloning
│   │
│   ├── pipeline.py
│   │   # Pipeline orchestration & latency logging
│   │
│   ├── samples/
│   │   ├── user_a.wav
│   │   ├── user_a.txt
│   │   ├── user_b.wav
│   │   └── user_b.txt
│   │
│   ├── outputs/
│   │   # Generated synthesized call audio
│   │
│   ├── requirements.txt
│   │
│   ├── test_stage1.py
│   │   # Voice cloning proof
│   │
│   ├── test_stage2.py
│   │   # Speech recognition test
│   │
│   ├── test_stage3.py
│   │   # Urdu ↔ English translation test
│   │
│   ├── test_stage4.py
│   │   # Complete STT → NMT → TTS pipeline
│   │
│   ├── test_stage5.py
│   │   # REST API & WebSocket test
│   │
│   └── test_stage7_two_clients.py
│       # Two-client translated call simulation
│
├── pubspec.yaml
└── README.md
```

---

# Setup

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS
- Flutter
- Python 3.9+
- Homebrew
- `ffmpeg`

---

## 1. Install FFmpeg

```bash
brew install ffmpeg
```

FFmpeg is used for audio conversion and resampling.

---

## 2. Create Python Virtual Environment

```bash
cd /Users/mac/StudioProjects/translation_call_demo

python3 -m venv ai_server/venv

source ai_server/venv/bin/activate
```

---

## 3. Install Python Dependencies

```bash
pip install -r ai_server/requirements.txt
```

---

## 4. AI Model Weights

Model weights are downloaded automatically on first use and cached locally under:

```text
~/.cache/huggingface
```

Models used by the project:

```text
Voice Cloning:
lucasnewman/f5-tts-mlx
vocos-mlx

Speech Recognition:
mlx-community/whisper-tiny

Translation:
Helsinki-NLP/opus-mt-ur-en
Helsinki-NLP/opus-mt-en-ur
```

The initial launch can therefore take longer while model weights are downloaded and initialized.

---

## 5. Install Flutter Dependencies

```bash
flutter pub get
```

---

# Verify Each Pipeline Stage

Activate the Python environment:

```bash
source ai_server/venv/bin/activate
export PYTHONPATH=ai_server
```

### Stage 1 — Zero-Shot Voice Cloning

```bash
python3 ai_server/test_stage1.py
```

Expected output:

```text
ai_server/outputs/output.wav
```

This verifies:

```text
Reference Voice + New Text
            ↓
          F5-TTS
            ↓
Generated Cloned Speech
```

---

### Stage 2 — Speech Recognition

```bash
python3 ai_server/test_stage2.py
```

Verifies:

```text
Speech
  ↓
MLX Whisper
  ↓
Transcription
```

---

### Stage 3 — Bidirectional Translation

```bash
python3 ai_server/test_stage3.py
```

Verifies:

```text
Urdu → English

and

English → Urdu
```

---

### Stage 4 — Complete AI Pipeline

```bash
python3 ai_server/test_stage4.py
```

Verifies:

```text
Speech
   ↓
STT
   ↓
Translation
   ↓
Voice-Cloned TTS
   ↓
Generated Audio
```

---

### Stage 5 — API & WebSocket

```bash
python3 ai_server/test_stage5.py
```

---

### Stage 7 — Two-Client Call Simulation

```bash
python3 ai_server/test_stage7_two_clients.py
```

This verifies the complete bidirectional translated-call flow.

---

# Enrolling a Voice

## Option A — File System

Create:

```text
ai_server/samples/<speaker_id>.wav
ai_server/samples/<speaker_id>.txt
```

For example:

```text
ai_server/samples/user_a.wav
ai_server/samples/user_a.txt
```

Convert an existing recording to the required format:

```bash
ffmpeg \
  -i my_recording.wav \
  -ac 1 \
  -ar 24000 \
  ai_server/samples/user_a.wav
```

Then ensure:

```text
user_a.txt
```

contains an accurate transcript of what is spoken in:

```text
user_a.wav
```

---

## Option B — REST API

```bash
curl -X POST http://localhost:8000/api/enroll-voice \
  -F "speaker_id=user_a" \
  -F "audio=@/path/to/my_recording.wav" \
  -F "ref_text=Hello, this is my reference recording for voice enrollment."
```

---

# Running the Application

## Step 1 — Start the AI Backend

```bash
cd /Users/mac/StudioProjects/translation_call_demo

source ai_server/venv/bin/activate

export PYTHONPATH=ai_server

python3 ai_server/server.py
```

The server binds to:

```text
http://0.0.0.0:8000
```

The AI models are pre-warmed to reduce first-request latency.

Verify the backend:

```bash
curl http://localhost:8000/api/health
```

---

# Step 2 — Start User A

Run the macOS Flutter client:

```bash
flutter run -d macos
```

Then:

1. Select **User A (Urdu Speaker)**.
2. Verify **AI Backend Online** is displayed.
3. Click **CALL USER B**.

---

# Step 3 — Start User B

Open another terminal:

```bash
flutter run -d chrome
```

Then:

1. Select **User B (English Speaker)**.
2. Verify **AI Backend Online** is displayed.
3. Accept User A's call or initiate the call from User B.

---

# Example Call

### User A speaks

```text
تم کیا کر رہے ہو؟
```

The server processes:

```text
STT
↓
تم کیا کر رہے ہو؟

Translation
↓
What are you doing?

F5-TTS + User A Reference Voice
↓
Generated English Audio
```

User B hears:

```text
"What are you doing?"
```

in User A's cloned voice.

### User B responds

```text
Where are you going?
```

The server processes:

```text
STT
↓
Where are you going?

Translation
↓
تم کہاں جا رہے ہو؟

F5-TTS + User B Reference Voice
↓
Generated Urdu Audio
```

User A hears the Urdu translation in User B's cloned voice.

---

# Audio Feedback & Loop Prevention

Running both clients on the same computer can create microphone/speaker feedback.

The proof-of-concept uses several protections.

### Push-to-Talk

The microphone records only while the user holds the speak button.

```text
Hold
 ↓
Record
 ↓
Speak
 ↓
Release
 ↓
Process utterance
```

### Playback Muting

Microphone capture is temporarily paused while translated audio is being played.

This prevents generated speech from being captured again and sent through the translation pipeline.

### Demo Utterances

The clients also provide predefined test utterances so the complete pipeline can be tested without microphone feedback.

User A:

```text
Send Demo Urdu:
"تم کیا کر رہے ہو؟"
```

User B:

```text
Send Demo English:
"Where are you going?"
```

---

# Debugging the Pipeline

Because the system contains several independent AI models, debug each stage separately.

For every utterance, inspect:

```text
1. Original Audio
       ↓
2. STT Result
       ↓
3. Translation Result
       ↓
4. Text Sent to F5-TTS
       ↓
5. Generated Audio
```

For example, if the final audio says the wrong sentence:

```text
Wrong STT?
    ↓
Check Whisper

Correct STT but wrong translation?
    ↓
Check MarianMT

Correct translation but wrong words spoken?
    ↓
Check F5-TTS input/output

Correct words but poor voice similarity?
    ↓
Check reference audio,
reference transcript,
noise and synthesis settings
```

This distinction is important because **F5-TTS does not perform translation**. It synthesizes the text produced by the translation stage.

---

# Current Limitations

This repository is a **proof-of-concept**, not a production calling system.

Current limitations include:

- Utterance-based rather than continuous streaming translation
- Approximately **3–4 seconds of AI processing latency** after an utterance
- Voice-cloning quality depends heavily on reference audio quality
- Translation quality depends on STT and NMT accuracy
- Cross-language voice characteristics may not perfectly match the original speaker
- Push-to-talk is currently used instead of automatic turn detection
- Running both clients on one machine can cause acoustic feedback without muting/headphones
- The current implementation is optimized specifically for Apple Silicon

---

# Future Improvements

Potential next steps include:

```text
Streaming STT
     ↓
Incremental Translation
     ↓
Streaming Voice-Cloned TTS
     ↓
Sub-second / low-latency chunks
```

Other improvements:

- Voice Activity Detection (VAD)
- Automatic utterance segmentation
- Better multilingual STT models
- Better multilingual translation models
- Streaming F5-TTS generation
- TTS reference embedding/cache optimization
- Parallel pipeline execution
- Audio jitter buffering
- WebRTC transport
- Interruption / barge-in handling
- Conversation turn management
- Noise suppression
- Echo cancellation
- Additional languages
- Mobile Flutter clients
- Dynamic language selection

---

# Privacy & Consent

Voice cloning introduces important privacy and identity considerations.

This proof-of-concept is designed around **explicit voice enrollment**:

```text
User records their own voice
          ↓
User provides/approves transcript
          ↓
Reference stored locally
          ↓
Voice cloning enabled for that user
```

The project is intended for experimentation with **consensual voice cloning**.

Do not enroll or clone another person's voice without their permission.

Because the AI pipeline runs locally, reference recordings, transcripts, speech recognition, translations, and generated audio do not need to be sent to a third-party AI service.

---

# Technology Stack

```text
Frontend
├── Flutter
├── Dart
└── WebSocket

Backend
├── Python
├── FastAPI
└── WebSocket

Speech Recognition
├── Whisper
└── MLX

Translation
├── MarianMT
├── PyTorch
└── SentencePiece

Voice Cloning
├── F5-TTS
├── MLX
└── Vocos

Hardware
├── Apple Silicon
├── Unified Memory
└── Metal GPU
```

---

# Summary

This project demonstrates that a complete translated voice-calling pipeline can run locally on consumer Apple Silicon hardware:

```text
User A speaks Urdu
        ↓
MLX Whisper
        ↓
Urdu Text
        ↓
MarianMT
        ↓
English Text
        ↓
F5-TTS + User A Voice
        ↓
User B hears English
in User A's cloned voice
```

and in reverse:

```text
User B speaks English
        ↓
MLX Whisper
        ↓
English Text
        ↓
MarianMT
        ↓
Urdu Text
        ↓
F5-TTS + User B Voice
        ↓
User A hears Urdu
in User B's cloned voice
```

The current implementation demonstrates the complete **STT → Translation → Zero-Shot Voice-Cloned TTS → Remote Playback** pipeline locally, without relying on cloud AI services.
