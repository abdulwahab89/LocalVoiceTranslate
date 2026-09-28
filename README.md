# Real-Time Translated Voice Calling App with Consent-Based Voice Cloning 🎙️

- <img width="2606" height="1604" alt="image" src="https://github.com/user-attachments/assets/10b9c7c1-dd2e-4490-8d04-54e7575128d6" />

A local, offline proof-of-concept demonstrating real-time bidirectional translated voice calls with zero-shot voice cloning, fully accelerated on **Apple Silicon (M1/M2/M3/M4 MacBook Pro)**.

- **Zero Cloud / Paid APIs**: Runs 100% locally.
- **Hardware Acceleration**: Apple MLX + Metal Performance Shaders (GPU). No CUDA/NVIDIA GPU required.
- **Bidirectional Calling**: User A speaks **German** ➔ User B hears **English** in User A's cloned voice; User B speaks **English** ➔ User A hears **German** in User B's cloned voice.
- **Utterance-Based Translation**: Speak ➔ Pause/Release ➔ STT ➔ NMT ➔ Voice Clone TTS ➔ Remote Playback.




---

## Architecture Overview

```text
User A (Flutter Client)                      AI Backend Server (Apple Silicon MLX)                     User B (Flutter Client)
┌──────────────────────┐                     ┌───────────────────────────────────┐                     ┌──────────────────────┐
│ Speaks German:         │  WebSocket Audio    │ 1. STT (MLX Whisper):             │  WebSocket Audio    │ Hears English in     │
│ "تم کیا کر رہے ہو؟"  │ ──────────────────> │    "کیا کر رہی ہو؟" (1.4s)        │ ──────────────────> │ User A's Cloned      │
│                      │                     │ 2. NMT (MarianMT):                │                     │ Voice:               │
│                      │                     │    "What are you doing?" (130ms)  │                     │ "What are you doing?"│
│                      │                     │ 3. Voice Clone TTS (F5-TTS MLX):  │                     │                      │
│                      │                     │    Using User A Reference (2.9s)  │                     │                      │
└──────────────────────┘                     └───────────────────────────────────┘                     └──────────────────────┘
                                                               ▲
                                                               │ Reverse Direction
                                                               ▼
┌──────────────────────┐                     ┌───────────────────────────────────┐                     ┌──────────────────────┐
│ Hears German in        │  WebSocket Audio    │ 1. STT (MLX Whisper):             │  WebSocket Audio    │ Speaks English:      │
│ User B's Cloned      │ <────────────────── │    "Where are you going?" (180ms) │ <────────────────── │ "Where are you going"│
│ Voice:               │                     │ 2. NMT (MarianMT):                │                     │                      │
│ "تم کہاں جا رہے ہو؟" │                     │    "تم کہاں جا رہے ہو؟" (130ms)   │                     │                      │
│                      │                     │ 3. Voice Clone TTS (F5-TTS MLX):  │                     │                      │
│                      │                     │    Using User B Reference (2.8s)  │                     │                      │
└──────────────────────┘                     └───────────────────────────────────┘                     └──────────────────────┘
```

---

## Measured Performance & Latency (M1 MacBook Pro)

| Pipeline Stage | Model & Framework | Compute Device | Typical Warm Latency |
| :--- | :--- | :--- | :--- |
| **STT (German / English)** | `mlx-community/whisper-tiny` | Apple Silicon GPU (MLX) | **180 ms – 1,450 ms** |
| **Translation (NMT)** | `Helsinki-NLP/opus-mt-ur-en` & `en-ur` | CPU (Torch / SentencePiece) | **130 ms – 180 ms** |
| **Voice Cloning (TTS)** | `lucasnewman/f5-tts-mlx` + `vocos-mlx` | Apple Silicon GPU (Metal) | **2,570 ms – 3,200 ms** (4 steps) |
| **Total Turnaround** | Full Utterance Processing Pipeline | Apple Silicon Unified Memory | **~3.2 s – 4.0 s** |

---

## Project Structure

```text
translation_call_demo/
├── lib/
│   └── main.dart                     # Flutter application (Home Screen & Call Screen)
├── macos/                            # macOS desktop runner with audio/network entitlements
├── ai_server/
│   ├── server.py                     # FastAPI REST API & WebSocket Real-time Call Hub
│   ├── stt.py                        # Speech-to-Text service interface & MLX Whisper implementation
│   ├── translator.py                 # Bidirectional Translation service & MarianMT implementation
│   ├── voice_clone.py                # Voice Cloning service interface & F5-TTS MLX implementation
│   ├── pipeline.py                   # TranslationCallPipeline orchestrator & latency logger
│   ├── samples/
│   │   ├── user_a.wav                # User A voice enrollment reference (24kHz mono WAV)
│   │   ├── user_a.txt                # User A reference transcript
│   │   ├── user_b.wav                # User B voice enrollment reference (24kHz mono WAV)
│   │   └── user_b.txt                # User B reference transcript
│   ├── outputs/                      # Generated synthesized call audio
│   ├── requirements.txt              # Python dependencies
│   ├── test_stage1.py                # Stage 1 test: Voice clone proof
│   ├── test_stage2.py                # Stage 2 test: German speech recognition
│   ├── test_stage3.py                # Stage 3 test: German <-> English translation
│   ├── test_stage4.py                # Stage 4 test: Complete AI pipeline
│   ├── test_stage5.py                # Stage 5 test: Local API & WebSocket
│   └── test_stage7_two_clients.py    # Stage 7 test: Two-client translated call simulation
├── pubspec.yaml                      # Flutter dependencies
└── README.md
```

---

## Setup Instructions (M1 MacBook Pro)

### 1. Homebrew Requirements

Install `ffmpeg` for high-performance audio conversion and resampling:

```bash
brew install ffmpeg
```

### 2. Python Version & Virtual Environment

Native macOS Apple Silicon Python 3.9+ is supported:

```bash
cd /Users/mac/StudioProjects/translation_call_demo

# Create virtual environment
python3 -m venv ai_server/venv
source ai_server/venv/bin/activate
```

### 3. Install Required Python Packages

```bash
pip install -r ai_server/requirements.txt
```

### 4. AI Model Weights

Model weights download automatically upon first run and are cached locally in `~/.cache/huggingface`:
- **Voice Cloning**: `lucasnewman/f5-tts-mlx` + `vocos-mlx`
- **Speech Recognition**: `mlx-community/whisper-tiny`
- **Translation**: `Helsinki-NLP/opus-mt-ur-en` and `Helsinki-NLP/opus-mt-en-ur`

### 5. Flutter Dependencies

Install Flutter dependencies:

```bash
flutter pub get
```

---

## Verifying Each Stage (Step-by-Step)

Each stage can be verified independently with automated verification scripts:

```bash
# Activate virtual environment
source ai_server/venv/bin/activate
export PYTHONPATH=ai_server

# Stage 1: Voice Clone Proof (Generates ai_server/outputs/output.wav)
python3 ai_server/test_stage1.py

# Stage 2: German Speech Recognition (Transcribes German audio -> German text)
python3 ai_server/test_stage2.py

# Stage 3: Bidirectional Translation (German <-> English)
python3 ai_server/test_stage3.py

# Stage 4: Complete AI Pipeline (STT -> NMT -> Cloned Voice TTS)
python3 ai_server/test_stage4.py

# Stage 5: Local API & WebSocket Server
python3 ai_server/test_stage5.py

# Stage 7: Two-Client Call Simulation
python3 ai_server/test_stage7_two_clients.py
```

---

## How to Enroll a User Voice

Voice cloning is **strictly consent-based**. Each user must enroll their own voice by recording approximately 5–30 seconds of clean speech.

### Option A: Via Command Line / File System
Save a clean mono 24 kHz WAV file to:
```text
ai_server/samples/<speaker_id>.wav
ai_server/samples/<speaker_id>.txt  (transcript of what was spoken)
```
Convert existing audio with `ffmpeg`:
```bash
ffmpeg -i my_recording.wav -ac 1 -ar 24000 ai_server/samples/user_a.wav
```

### Option B: Via REST API
```bash
curl -X POST http://localhost:8000/api/enroll-voice \
  -F "speaker_id=user_a" \
  -F "audio=@/path/to/my_recording.wav" \
  -F "ref_text=Some call me nature, others call me mother nature."
```

---

## Running the Application

### Step 1: Start the AI Backend Server

```bash
cd /Users/mac/StudioProjects/translation_call_demo
source ai_server/venv/bin/activate
export PYTHONPATH=ai_server
python3 ai_server/server.py
```
The server will bind to `http://0.0.0.0:8000` and pre-warm models into M1 unified memory.

Verify health:
```bash
curl http://localhost:8000/api/health
```

---

### Step 2: Run Two Flutter Clients for Testing

You can run two clients on the same M1 Mac simultaneously:

#### Client 1 (User A — macOS Desktop App):
```bash
flutter run -d macos
```
1. Select **User A (German Speaker)** on the home screen.
2. Verify the backend indicator shows **AI Backend Online**.
3. Click **CALL USER B**.

#### Client 2 (User B — Google Chrome Web App):
Open a new terminal window:
```bash
flutter run -d chrome
```
1. Select **User B (English Speaker)** on the home screen.
2. Verify the backend indicator shows **AI Backend Online**.
3. Accept the incoming call or initiate the call with User A.

---

## Audio Feedback & Loop Prevention

When both users are on the same machine or using open speakers:
1. **Push-to-Talk (Hold to Speak)**: The app records speech when held and immediately releases when done.
2. **Playback Muting**: The client automatically pauses microphone capture during translated speech playback.
3. **Demo Utterance Buttons**: Quick-action buttons allow testing both directions without wearing headphones:
   - User A: `[Send Demo German: "تم کیا کر رہے ہو؟"]`
   - User B: `[Send Demo English: "Where are you going?"]`
