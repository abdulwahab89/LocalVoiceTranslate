import 'dart:async';
import 'dart:convert';
import 'dart:io';


import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:audioplayers/audioplayers.dart';
import 'package:record/record.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const TranslationCallApp());
}

class TranslationCallApp extends StatelessWidget {
  const TranslationCallApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Translation Call Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1E88E5),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

// =============================================================================
// Home Screen
// =============================================================================

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _selectedUser = 'user_a'; // 'user_a' or 'user_b'
  late String _serverHost;
  late final TextEditingController _hostController;
  bool _serverOnline = false;
  String _serverDevice = 'Checking...';
  bool _isCheckingServer = false;
  final Set<String> _enrolled = {};
  bool _enrolling = false;
  String _receiveLang = 'de';

  @override
  void initState() {
    super.initState();
    if (kIsWeb) {
      _serverHost = 'localhost:8000';
    } else if (Platform.isAndroid) {
      _serverHost = '10.0.2.2:8000';
    } else {
      _serverHost = 'localhost:8000';
    }
    _hostController = TextEditingController(text: _serverHost);
    _checkServerHealth();
  }

  @override
  void dispose() {
    _hostController.dispose();
    super.dispose();
  }

  Future<void> _checkServerHealth() async {
    if (!mounted) return;
    setState(() => _isCheckingServer = true);
    try {
      final res = await http
          .get(Uri.parse('http://$_serverHost/api/health'))
          .timeout(const Duration(seconds: 3));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final speakers = await http.get(
          Uri.parse('http://$_serverHost/api/speakers'),
        );
        if (!mounted) return;
        _enrolled.clear();
        if (speakers.statusCode == 200) {
          for (final profile in jsonDecode(speakers.body)['speakers']) {
            _enrolled.add(profile['speaker_id'] as String);
          }
        }
        setState(() {
          _serverOnline = data['pipeline_ready'] == true;
          _serverDevice = '${data["platform"]} (${data["device"]})';
        });
      } else {
        setState(() {
          _serverOnline = false;
          _serverDevice = 'Unavailable (status ${res.statusCode})';
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _serverOnline = false;
        _serverDevice = 'Offline / Connection refused';
      });
    } finally {
      if (mounted) {
        setState(() => _isCheckingServer = false);
      }
    }
  }

  void _startCall() {
    final targetUser = _selectedUser == 'user_a' ? 'user_b' : 'user_a';
    final sourceLang = _selectedUser == 'user_a' ? 'de' : 'en';
    final targetLang = _selectedUser == 'user_a' ? 'en' : 'de';
    final receiveLang = _receiveLang;

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => CallScreen(
          myId: _selectedUser,
          targetId: targetUser,
          sourceLang: sourceLang,
          receiveLang: receiveLang,
          targetLang: targetLang,
          serverHost: _serverHost,
        ),
      ),
    );
  }

  Future<void> _enrollVoice() async {
    const phrase =
        'Hello, this is my own voice. I am recording a clear sample for translated calls.';
    final consent = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Enroll your voice'),
        content: const Text(
          'Record your own voice for 8 seconds. Read this phrase clearly:\n\n"$phrase"\n\nYour voice sample will be enrolled to synthesize speech in your own voice during German ↔ English translated calls.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Record my voice'),
          ),
        ],
      ),
    );
    if (consent != true || !mounted) return;
    final speaker = _selectedUser;
    final recorder = AudioRecorder();
    String? localFilePath;
    setState(() => _enrolling = true);
    try {
      if (!await recorder.hasPermission()) {
        throw StateError('Microphone permission denied');
      }
      const recordConfig = RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 24000,
        numChannels: 1,
      );
      if (kIsWeb) {
        await recorder.start(recordConfig, path: '');
      } else {
        localFilePath =
            '${Directory.systemTemp.path}/enroll_${DateTime.now().microsecondsSinceEpoch}.wav';
        await recorder.start(recordConfig, path: localFilePath);
      }
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            duration: Duration(seconds: 8),
            content: Text('Recording for 8 seconds: $phrase'),
          ),
        );
      }
      await Future<void>.delayed(const Duration(seconds: 8));
      final recordedPathOrUrl = await recorder.stop();

      Uint8List audioBytes;
      if (kIsWeb) {
        if (recordedPathOrUrl == null || recordedPathOrUrl.isEmpty) {
          throw StateError('Web audio recording produced no data');
        }
        final res = await http.get(Uri.parse(recordedPathOrUrl));
        audioBytes = res.bodyBytes;
      } else {
        final path = recordedPathOrUrl ?? localFilePath;
        if (path == null || !File(path).existsSync()) {
          throw StateError('Recorded audio file missing');
        }
        audioBytes = await File(path).readAsBytes();
      }

      final request =
          http.MultipartRequest(
              'POST',
              Uri.parse('http://$_serverHost/api/enroll-voice'),
            )
            ..fields.addAll({
              'speaker_id': speaker,
              'ref_text': phrase,
              'reference_language': 'en',
            })
            ..files.add(
              http.MultipartFile.fromBytes(
                'audio',
                audioBytes,
                filename: 'enroll_$speaker.wav',
              ),
            );
      final response = await http.Response.fromStream(await request.send());
      if (response.statusCode != 200) {
        throw StateError(response.body);
      }
      await _checkServerHealth();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Enrollment failed: $e')));
      }
    } finally {
      await recorder.dispose();
      if (!kIsWeb && localFilePath != null) {
        try {
          final file = File(localFilePath);
          if (file.existsSync()) {
            await file.delete();
          }
        } catch (_) {}
      }
      if (mounted) {
        setState(() => _enrolling = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isUserA = _selectedUser == 'user_a';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Translation Call Demo'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _checkServerHealth,
            tooltip: 'Refresh Server Status',
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Backend Health Card
            Card(
              elevation: 2,
              color: _serverOnline
                  ? const Color(0xFF1B5E20).withAlpha(77)
                  : const Color(0xFFB71C1C).withAlpha(77),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
                side: BorderSide(
                  color: _serverOnline ? Colors.green : Colors.red,
                  width: 1.5,
                ),
              ),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  children: [
                    Icon(
                      _serverOnline ? Icons.check_circle : Icons.error,
                      color: _serverOnline
                          ? Colors.greenAccent
                          : Colors.redAccent,
                      size: 28,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _serverOnline
                                ? 'AI Backend Online (Apple Silicon Metal)'
                                : 'AI Backend Offline',
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 15,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _serverDevice,
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey.shade400,
                            ),
                          ),
                        ],
                      ),
                    ),
                    if (_isCheckingServer)
                      const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // User Identity Selection
            const Text(
              'Select Client Profile for this Instance:',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
            const SizedBox(height: 12),
            SegmentedButton<String>(
              segments: const [
                ButtonSegment(
                  value: 'user_a',
                  label: Text('User A (German Speaker)'),
                  icon: Icon(Icons.person),
                ),
                ButtonSegment(
                  value: 'user_b',
                  label: Text('User B (English Speaker)'),
                  icon: Icon(Icons.person_outline),
                ),
              ],
              selected: {_selectedUser},
              onSelectionChanged: (set) {
                setState(() {
                  _selectedUser = set.first;
                  _receiveLang = _selectedUser == 'user_a' ? 'de' : 'en';
                });
              },
            ),
            const SizedBox(height: 20),

            Row(
              children: [
                const Text('I want to hear: '),
                DropdownButton<String>(
                  value: _receiveLang,
                  items: const [
                    DropdownMenuItem(value: 'de', child: Text('German')),
                    DropdownMenuItem(value: 'en', child: Text('English')),
                  ],
                  onChanged: (value) {
                    if (value != null) {
                      setState(() => _receiveLang = value);
                    }
                  },
                ),
              ],
            ),
            // Profile Info Card
            Card(
              elevation: 2,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    CircleAvatar(
                      radius: 36,
                      backgroundColor: isUserA ? Colors.teal : Colors.indigo,
                      child: Text(
                        isUserA ? 'A' : 'B',
                        style: const TextStyle(
                          fontSize: 32,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      isUserA ? 'User A' : 'User B',
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.blueGrey.shade800,
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Text(
                        'Speaks ${isUserA ? 'German' : 'English'} • Hears ${_receiveLang == 'de' ? 'German' : 'English'}',
                        style: const TextStyle(
                          fontSize: 13,
                          color: Colors.cyanAccent,
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(
                          Icons.verified,
                          size: 16,
                          color: Colors.greenAccent,
                        ),
                        const SizedBox(width: 6),
                        Text(
                          _enrolled.contains(_selectedUser)
                              ? 'Your voice reference is ready'
                              : 'Voice not enrolled',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey.shade400,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Server Host Field
            TextFormField(
              controller: _hostController,
              decoration: InputDecoration(
                labelText: 'Backend Server Address',
                border: const OutlineInputBorder(),
                prefixIcon: const Icon(Icons.dns),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.refresh),
                  onPressed: _checkServerHealth,
                  tooltip: 'Connect',
                ),
              ),
              onChanged: (val) {
                _serverHost = val.trim();
                _checkServerHealth();
              },
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              children: [
                ActionChip(
                  avatar: const Icon(Icons.phone_android, size: 16),
                  label: const Text(
                    '10.0.2.2:8000 (Emulator)',
                    style: TextStyle(fontSize: 11),
                  ),
                  onPressed: () {
                    setState(() {
                      _serverHost = '10.0.2.2:8000';
                      _hostController.text = _serverHost;
                    });
                    _checkServerHealth();
                  },
                ),
                ActionChip(
                  avatar: const Icon(Icons.computer, size: 16),
                  label: const Text(
                    'localhost:8000 (Mac/Desktop)',
                    style: TextStyle(fontSize: 11),
                  ),
                  onPressed: () {
                    setState(() {
                      _serverHost = 'localhost:8000';
                      _hostController.text = _serverHost;
                    });
                    _checkServerHealth();
                  },
                ),
                ActionChip(
                  avatar: const Icon(Icons.wifi, size: 16),
                  label: const Text(
                    '192.168.1.75:8000 (WiFi)',
                    style: TextStyle(fontSize: 11),
                  ),
                  onPressed: () {
                    setState(() {
                      _serverHost = '192.168.1.75:8000';
                      _hostController.text = _serverHost;
                    });
                    _checkServerHealth();
                  },
                ),
              ],
            ),
            const SizedBox(height: 24),

            FilledButton.icon(
              onPressed: _enrolling ? null : _enrollVoice,
              icon: const Icon(Icons.record_voice_over),
              label: Text(
                _enrolling ? 'Recording voice sample...' : 'Enroll my voice',
              ),
            ),
            const SizedBox(height: 12),
            // Call Button
            FilledButton.icon(
              onPressed:
                  _serverOnline &&
                      _enrolled.contains(_selectedUser) &&
                      !_enrolling
                  ? _startCall
                  : null,
              style: FilledButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 18),
                backgroundColor: Colors.green.shade700,
              ),
              icon: const Icon(Icons.call, size: 26),
              label: Text(
                'CALL ${isUserA ? "USER B" : "USER A"}',
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// Active Call Screen
// =============================================================================

class CallScreen extends StatefulWidget {
  final String myId;
  final String targetId;
  final String sourceLang;
  final String receiveLang;
  final String targetLang;
  final String serverHost;

  const CallScreen({
    super.key,
    required this.myId,
    required this.targetId,
    required this.sourceLang,
    required this.receiveLang,
    required this.targetLang,
    required this.serverHost,
  });

  @override
  State<CallScreen> createState() => _CallScreenState();
}

class _CallScreenState extends State<CallScreen> {
  WebSocketChannel? _channel;
  bool _isConnected = false;
  bool _callAccepted = false;
  bool _isMuted = false;
  bool _speakerOn = true;
  bool _isDisposed = false;

  // Utterance status
  String _statusText = 'Connecting to Call Server...';
  String _originalTranscript = '';
  String _translatedText = '';
  String? _displaySourceLang;
  String? _displayTargetLang;
  Map<String, dynamic>? _lastMetrics;

  // Recording & Playback
  final AudioRecorder _audioRecorder = AudioRecorder();
  final AudioPlayer _audioPlayer = AudioPlayer();
  bool _isRecording = false;
  bool _isProcessing = false;
  bool _isPlayingTranslatedAudio = false;
  String? _callId;
  int _nextSegment = 0;
  int _lastReceived = 0;
  bool _stopping = false;
  Future<void> _playbackQueue = Future<void>.value();

  @override
  void initState() {
    super.initState();
    _initWebSocket();
    _setupAudioPlayer();
  }

  void _setupAudioPlayer() {
    _audioPlayer.onPlayerStateChanged.listen((state) {
      if (!mounted) return;
      if (state == PlayerState.playing) {
        setState(() {
          _isPlayingTranslatedAudio = true;
          _statusText = 'Playing translated speech...';
        });
      } else if (state == PlayerState.completed ||
          state == PlayerState.stopped) {
        setState(() {
          _isPlayingTranslatedAudio = false;
          _statusText = 'Listening for utterances...';
        });
      }
    });
  }

  void _initWebSocket() {
    try {
      final wsUrl = Uri.parse(
        'ws://${widget.serverHost}/ws/call/${widget.myId}',
      );
      _channel = WebSocketChannel.connect(wsUrl);
      _channel!.stream.listen(
        _handleWebSocketMessage,
        onError: (err) {
          if (!mounted) return;
          setState(() => _statusText = 'WebSocket Error: $err');
        },
        onDone: () {
          if (!mounted) return;
          setState(() {
            _isConnected = false;
            _statusText = 'Disconnected from server';
          });
        },
      );
    } catch (e) {
      setState(() => _statusText = 'Connection failed: $e');
    }
  }

  void _handleWebSocketMessage(dynamic message) {
    if (!mounted) return;
    try {
      final data = jsonDecode(message.toString());
      final type = data['type'];
      if ([
            'processing_started',
            'utterance_sent_confirmation',
            'translated_utterance',
            'call_terminated',
          ].contains(type) &&
          data['call_id'] != _callId) {
        return;
      }
      if (type == 'error' &&
          data['call_id'] != null &&
          data['call_id'] != _callId) {
        return;
      }

      if (type == 'connected') {
        setState(() {
          _isConnected = true;
          _statusText = 'Connected. Calling ${widget.targetId}...';
        });
        _channel?.sink.add(
          jsonEncode({
            'type': 'configure',
            'source_language': widget.sourceLang,
            'receive_language': widget.receiveLang,
          }),
        );
        // Signal call start to remote peer
        _channel?.sink.add(
          jsonEncode({'type': 'call_start', 'target_client': widget.targetId}),
        );
      } else if (type == 'presence_update') {
        if (_callId == null &&
            (data['active_clients'] as List).contains(widget.targetId)) {
          _channel?.sink.add(
            jsonEncode({
              'type': 'call_start',
              'target_client': widget.targetId,
            }),
          );
        }
      } else if (type == 'call_pending') {
        _callId = data['call_id'];
      } else if (type == 'incoming_call') {
        _callId = data['call_id'];
        // Automatically accept for smooth testing
        _channel?.sink.add(
          jsonEncode({
            'type': 'call_accept',
            'call_id': _callId,
            'target_client': data['from_client'],
          }),
        );
        setState(() {
          _callAccepted = true;
          _statusText = 'Call connected with ${data["from_client"]}';
        });
      } else if (type == 'call_connected') {
        _callId = data['call_id'];
        setState(() {
          _callAccepted = true;
          _statusText = 'Call connected with ${data["peer_client"]}';
        });
      } else if (type == 'call_terminated') {
        _endCall(remoteEnded: true);
      } else if (type == 'processing_started') {
        setState(() {
          _isProcessing = true;
          _statusText = 'AI translating & cloning voice...';
        });
      } else if (type == 'utterance_sent_confirmation') {
        setState(() {
          _isProcessing = false;
          _displaySourceLang = data['source_language'];
          _displayTargetLang = data['target_language'];
          _originalTranscript = data['transcript'] ?? '';
          _translatedText = data['translated_text'] ?? '';
          _lastMetrics = data['metrics'];
          _statusText = data['status'] == 'success'
              ? 'Utterance translated and sent!'
              : (data['error'] ?? data['status']);
        });
      } else if (type == 'translated_utterance') {
        final segment = int.tryParse(data['segment_id'].toString()) ?? 0;
        if (segment <= _lastReceived) return;
        _lastReceived = segment;
        // Received translated utterance from peer
        final b64Audio = data['audio_base64'];
        final audioFile = data['audio_file'];
        setState(() {
          _displaySourceLang = data['source_language'];
          _displayTargetLang = data['target_language'];
          _originalTranscript = data['transcript'] ?? '';
          _translatedText = data['translated_text'] ?? '';
          _lastMetrics = data['metrics'];
          _statusText = data['status'] == 'success'
              ? 'Received translated utterance'
              : (data['error'] ?? data['status']);
        });

        if (_speakerOn) {
          _playAudio(b64Audio: b64Audio, audioFile: audioFile);
        }
      } else if (type == 'silence_detected') {
        setState(() {
          _isProcessing = false;
          _statusText = data['message'] ?? 'No speech detected';
        });
      } else if (type == 'error') {
        setState(() {
          _isProcessing = false;
          _statusText = 'Server Error: ${data["message"]}';
        });
      }
    } catch (e) {
      debugPrint('Error parsing ws message: $e');
    }
  }

  Future<void> _playAudio({String? b64Audio, String? audioFile}) {
    final call = _callId;
    _playbackQueue = _playbackQueue.then((_) async {
      if (!mounted || call != _callId || !_speakerOn) return;
      setState(() {
        _isRecording = false;
        _isPlayingTranslatedAudio = true;
      });
      File? tempFile;
      try {
        Source? source;
        if (audioFile != null && audioFile.isNotEmpty) {
          source = UrlSource('http://${widget.serverHost}/api/audio/$audioFile');
        } else if (b64Audio != null && b64Audio.isNotEmpty) {
          if (!kIsWeb) {
            final tempPath =
                '${Directory.systemTemp.path}/play_${DateTime.now().millisecondsSinceEpoch}.wav';
            tempFile = File(tempPath);
            await tempFile.writeAsBytes(base64Decode(b64Audio));
            source = DeviceFileSource(tempPath);
          } else {
            source = BytesSource(base64Decode(b64Audio));
          }
        }
        if (source != null) {
          final complete = _audioPlayer.onPlayerComplete.first;
          await _audioPlayer.play(source);
          await complete.timeout(const Duration(seconds: 60));
        }
      } catch (e) {
        debugPrint('Playback failed: $e');
      } finally {
        if (tempFile != null) {
          try {
            if (tempFile.existsSync()) await tempFile.delete();
          } catch (_) {}
        }
        if (mounted) setState(() => _isPlayingTranslatedAudio = false);
      }
    });
    return _playbackQueue;
  }

  String? _recordedUtterancePath;

  Future<void> _startRecording() async {
    if (!_callAccepted ||
        _callId == null ||
        _isMuted ||
        _isProcessing ||
        _isRecording ||
        _stopping ||
        _isPlayingTranslatedAudio) {
      return;
    }
    try {
      if (!await _audioRecorder.hasPermission()) {
        setState(() => _statusText = 'Microphone permission denied');
        return;
      }
      const config = RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
      );
      if (kIsWeb) {
        await _audioRecorder.start(config, path: '');
      } else {
        _recordedUtterancePath =
            '${Directory.systemTemp.path}/call_utt_${DateTime.now().millisecondsSinceEpoch}.wav';
        await _audioRecorder.start(config, path: _recordedUtterancePath!);
      }
      setState(() {
        _isRecording = true;
        _statusText = 'Recording — release to send';
      });
    } catch (e) {
      setState(() => _statusText = 'Recording error: $e');
    }
  }

  Future<void> _stopRecordingAndSend() async {
    if (!_isRecording || _stopping) return;
    _stopping = true;
    final call = _callId;
    await Future<void>.delayed(const Duration(milliseconds: 200));
    try {
      if (!mounted || call != _callId || !_isRecording) return;
      final recordResult = await _audioRecorder.stop();
      setState(() => _isRecording = false);

      Uint8List? bytes;
      if (kIsWeb) {
        if (recordResult != null && recordResult.isNotEmpty) {
          final res = await http.get(Uri.parse(recordResult));
          bytes = res.bodyBytes;
        }
      } else {
        final path = recordResult ?? _recordedUtterancePath;
        if (path != null) {
          final f = File(path);
          if (f.existsSync()) {
            bytes = await f.readAsBytes();
            try {
              await f.delete();
            } catch (_) {}
          }
        }
      }

      if (bytes == null || bytes.length < 44 + 4000) {
        setState(
          () => _statusText =
              'Utterance too short — hold while speaking a sentence',
        );
        return;
      }
      _sendAudio(bytes);
    } catch (e) {
      setState(() => _statusText = 'Send error: $e');
    } finally {
      _stopping = false;
    }
  }

  void _sendAudio(Uint8List bytes) {
    if (_callId == null || !_callAccepted || _isProcessing) return;
    setState(() {
      _isProcessing = true;
      _statusText = 'Processing utterance...';
    });
    _channel?.sink.add(
      jsonEncode({
        'type': 'utterance',
        'call_id': _callId,
        'segment_id': ++_nextSegment,
        'audio_base64': base64Encode(bytes),
      }),
    );
  }

  // Sends the pre-enrolled test utterance (e.g. "Tum kya kar rahe ho?" or "Where are you going?")
  Future<void> _sendDemoUtterance() async {
    setState(() {
      _isProcessing = true;
      _statusText = 'Sending demo utterance to AI server...';
    });

    try {
      final sampleName = widget.myId == 'user_a'
          ? 'german_sentence.wav'
          : 'english_sentence.wav';
      Uint8List? bytes;

      // 1. Try local file path (Desktop / Local dev only)
      if (!kIsWeb) {
        try {
          final localFile = File('ai_server/samples/$sampleName');
          if (localFile.existsSync()) {
            bytes = await localFile.readAsBytes();
          }
        } catch (_) {}
      }

      // 2. Fallback to server HTTP endpoint (Web / Mobile emulator / remote client)
      if (bytes == null || bytes.isEmpty) {
        final uri = Uri.parse(
          'http://${widget.serverHost}/api/sample-audio/$sampleName',
        );
        final res = await http.get(uri).timeout(const Duration(seconds: 4));
        if (res.statusCode == 200) {
          bytes = res.bodyBytes;
        }
      }

      if (bytes == null || bytes.isEmpty) {
        setState(() {
          _isProcessing = false;
          _statusText = 'Failed to load demo sample: $sampleName';
        });
        return;
      }

      _isProcessing = false;
      _sendAudio(bytes);
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _statusText = 'Demo send error: $e';
      });
    }
  }

  void _endCall({bool remoteEnded = false}) {
    if (!remoteEnded) {
      try {
        _channel?.sink.add(
          jsonEncode({
            'type': 'call_end',
            'call_id': _callId,
            'target_client': widget.targetId,
          }),
        );
      } catch (_) {}
    }
    _callId = null;
    Navigator.pop(context);
  }

  @override
  void dispose() {
    if (!_isDisposed) {
      _isDisposed = true;
      _callId = null;
      try {
        _channel?.sink.close();
      } catch (_) {}
      try {
        _audioPlayer.stop();
        _audioPlayer.dispose();
      } catch (_) {}
      try {
        _audioRecorder.dispose();
      } catch (_) {}
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isUserA = widget.myId == 'user_a';
    final langDirection = 'Hear ${widget.receiveLang.toUpperCase()}';

    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text('${widget.myId.toUpperCase()} in Call'),
        centerTitle: true,
        leading: Padding(
          padding: const EdgeInsets.only(left: 16.0),
          child: Icon(
            _isConnected ? Icons.wifi : Icons.wifi_off,
            color: _isConnected ? Colors.greenAccent : Colors.redAccent,
            size: 20,
          ),
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 12.0),
          child: Column(
            children: [
              // Peer Info & Translation Banner
              Text(
                widget.targetId.toUpperCase(),
                style: const TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                _callAccepted ? 'Call Active' : 'Connecting...',
                style: TextStyle(
                  color: _callAccepted
                      ? Colors.greenAccent
                      : Colors.amberAccent,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: Colors.green.shade900.withAlpha(128),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: Colors.greenAccent.shade400,
                    width: 1,
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.translate,
                      size: 16,
                      color: Colors.greenAccent,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'TRANSLATION ON ($langDirection)',
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: Colors.greenAccent,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // Peer Avatar with Pulse Status
              Stack(
                alignment: Alignment.center,
                children: [
                  CircleAvatar(
                    radius: 48,
                    backgroundColor: _isPlayingTranslatedAudio
                        ? Colors.greenAccent.withAlpha(51)
                        : Colors.blueGrey.shade800,
                  ),
                  CircleAvatar(
                    radius: 40,
                    backgroundColor: isUserA ? Colors.indigo : Colors.teal,
                    child: Text(
                      isUserA ? 'B' : 'A',
                      style: const TextStyle(
                        fontSize: 34,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),

              // Status line
              Text(
                _statusText,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                  color: _isPlayingTranslatedAudio
                      ? Colors.greenAccent
                      : _isRecording
                      ? Colors.redAccent
                      : Colors.grey.shade400,
                ),
              ),
              const SizedBox(height: 20),

              // Transcript & Translation Cards
              Expanded(
                child: ListView(
                  children: [
                    // Original Utterance
                    Card(
                      color: const Color(0xFF1E1E1E),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                const Icon(
                                  Icons.mic,
                                  size: 16,
                                  color: Colors.orangeAccent,
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  'Original Utterance (${(_displaySourceLang ?? widget.sourceLang).toUpperCase()}):',
                                  style: const TextStyle(
                                    fontSize: 12,
                                    color: Colors.orangeAccent,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              _originalTranscript.isEmpty
                                  ? '(No speech captured yet)'
                                  : '"$_originalTranscript"',
                              style: TextStyle(
                                fontSize: 16,
                                fontStyle: FontStyle.italic,
                                color: _originalTranscript.isEmpty
                                    ? Colors.grey
                                    : Colors.white,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),

                    // Translated Cloned Speech
                    Card(
                      color: const Color(0xFF1E1E1E),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                const Icon(
                                  Icons.record_voice_over,
                                  size: 16,
                                  color: Colors.cyanAccent,
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  'Cloned Voice Output (${(_displayTargetLang ?? widget.targetLang).toUpperCase()}):',
                                  style: const TextStyle(
                                    fontSize: 12,
                                    color: Colors.cyanAccent,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              _translatedText.isEmpty
                                  ? '(Awaiting translated synthesis...)'
                                  : '"$_translatedText"',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: _translatedText.isEmpty
                                    ? Colors.grey
                                    : Colors.white,
                              ),
                            ),
                            if (_lastMetrics != null) ...[
                              const Divider(height: 24),
                              Wrap(
                                spacing: 8,
                                children: [
                                  Chip(
                                    label: Text(
                                      'STT: ${_lastMetrics!["stt_ms"]}ms',
                                    ),
                                    backgroundColor: Colors.blueGrey.shade900,
                                    labelStyle: const TextStyle(fontSize: 11),
                                  ),
                                  Chip(
                                    label: Text(
                                      'NMT: ${_lastMetrics!["translation_ms"]}ms',
                                    ),
                                    backgroundColor: Colors.blueGrey.shade900,
                                    labelStyle: const TextStyle(fontSize: 11),
                                  ),
                                  Chip(
                                    label: Text(
                                      'TTS: ${_lastMetrics!["tts_ms"]}ms',
                                    ),
                                    backgroundColor: Colors.blueGrey.shade900,
                                    labelStyle: const TextStyle(fontSize: 11),
                                  ),
                                  Chip(
                                    label: Text(
                                      'Total: ${_lastMetrics!["total_ms"]}ms',
                                    ),
                                    backgroundColor: Colors.teal.shade900,
                                    labelStyle: const TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              // Demo Utterance Shortcut Button
              Padding(
                padding: const EdgeInsets.only(bottom: 12.0),
                child: OutlinedButton.icon(
                  onPressed: _isProcessing || !_callAccepted
                      ? null
                      : _sendDemoUtterance,
                  icon: const Icon(Icons.play_circle_outline),
                  label: Text(
                    isUserA
                        ? 'Send Demo German: "Hallo, ich freue mich..."'
                        : 'Send Demo English: "Where are you going?"',
                    style: const TextStyle(fontSize: 13),
                  ),
                ),
              ),

              // Call Controls: Mute, PTT Record, Speaker, End
              Container(
                padding: const EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: const Color(0xFF1E1E1E),
                  borderRadius: BorderRadius.circular(32),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    // Mute
                    IconButton(
                      icon: Icon(_isMuted ? Icons.mic_off : Icons.mic),
                      color: _isMuted ? Colors.redAccent : Colors.white,
                      onPressed: () => setState(() {
                        _isMuted = !_isMuted;
                        _isRecording = false;
                      }),
                      tooltip: 'Mute Microphone',
                    ),

                    // Push to Talk / Utterance Button
                    GestureDetector(
                      onTapDown: (_) => _startRecording(),
                      onTapUp: (_) => _stopRecordingAndSend(),
                      onTapCancel: () => _stopRecordingAndSend(),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 12,
                        ),
                        decoration: BoxDecoration(
                          color: _isRecording
                              ? Colors.redAccent
                              : Colors.blueAccent,
                          borderRadius: BorderRadius.circular(24),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              _isRecording
                                  ? Icons.fiber_manual_record
                                  : Icons.mic,
                              color: Colors.white,
                              size: 20,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              _isRecording
                                  ? 'Release to Send'
                                  : 'Hold to Speak',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),

                    // Speaker Toggle
                    IconButton(
                      icon: Icon(
                        _speakerOn ? Icons.volume_up : Icons.volume_off,
                      ),
                      color: _speakerOn ? Colors.white : Colors.grey,
                      onPressed: () => setState(() => _speakerOn = !_speakerOn),
                      tooltip: 'Toggle Speaker',
                    ),

                    // End Call
                    IconButton(
                      icon: const Icon(Icons.call_end),
                      color: Colors.redAccent,
                      onPressed: () => _endCall(),
                      tooltip: 'End Call',
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
