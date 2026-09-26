# LocalVoiceTranslate 🎙️

Real-time voice translation using **locally running AI models** for speech recognition, translation, text-to-speech, and voice cloning.

The goal of this project is to experiment with running the complete speech translation pipeline locally instead of depending on paid cloud APIs.

## How It Works

```text
Microphone Input
      ↓
Speech-to-Text
      ↓
Language Detection
      ↓
Translation
      ↓
TTS / Voice Cloning
      ↓
Translated Audio Output
```

You speak in one language, the application captures the audio and sends it through the local processing pipeline.

The speech is first transcribed, translated into the selected target language, and then converted back into speech using TTS or a cloned voice.

## Technical Implementation

The project combines multiple AI/audio components into one pipeline:

**Speech-to-Text (STT)**  
Audio captured from the microphone is processed by a locally running speech recognition model to generate a transcription.

**Translation**  
The transcription is passed through the translation layer to produce text in the selected target language.

**Text-to-Speech (TTS)**  
Translated text is synthesized back into audio using a local TTS model.

**Voice Cloning**  
A reference voice sample can be used by the speech synthesis model to generate translated speech while preserving characteristics of the original speaker's voice.

**Audio Pipeline**  
The application manages recording, audio preprocessing, model inference, generated audio, playback, and state transitions between each stage.

## Why Local Models?

Running the models locally provides a useful environment for experimenting with:

- Offline AI inference
- Speech processing
- Machine translation
- Voice synthesis
- Voice cloning
- AI pipeline orchestration
- Latency optimization
- Reduced dependency on paid APIs
- Greater control over audio data

## Main Challenge

The interesting part of this project isn't the UI — it's connecting several AI models into a usable speech pipeline.

Each stage has different input/output requirements and inference times:

```text
Audio
  → STT inference
  → Transcribed text
  → Translation inference
  → Translated text
  → TTS / Voice Clone inference
  → Generated audio
  → Playback
```

Keeping this pipeline responsive while handling recording, model processing, errors, and playback is one of the main engineering challenges explored in the project.

## Features

- 🎙️ Microphone audio capture
- 📝 Local speech-to-text
- 🌍 Multi-language translation
- 🔊 Local text-to-speech
- 🗣️ Voice cloning
- 🔄 End-to-end speech translation pipeline
- ⚡ Focus on low-latency inference
- 💻 Local model execution
- ☁️ No paid AI API required for the core pipeline

## Architecture

The application layer handles the user interaction and audio lifecycle, while local AI services are responsible for model inference.

```text
┌──────────────────────┐
│     Application      │
│   UI + Audio Input   │
└──────────┬───────────┘
           │ Audio
           ▼
┌──────────────────────┐
│  Local AI Services   │
├──────────────────────┤
│ Speech Recognition   │
│ Translation          │
│ TTS / Voice Cloning  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Generated Speech    │
│      Playback        │
└──────────────────────┘
```

## Project Goal

This is an experimental/open-source project built to understand how modern speech AI models can be combined into a practical application.

Rather than treating STT, translation, and TTS as isolated demos, the project connects them into an end-to-end system:

**Speak → Understand → Translate → Recreate Voice → Listen**

## Status

🚧 **Experimental / Active Development**

The project is primarily intended for experimentation, learning, and demonstrating local speech AI integration. Latency and output quality depend heavily on the models and hardware being used.

## Disclaimer

Voice cloning should only be used with your own voice or with the explicit permission of the person whose voice is being used.

## License

Open source. See the repository's `LICENSE` file for details.
