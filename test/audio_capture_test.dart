import 'package:flutter_test/flutter_test.dart';
import 'package:translation_call_demo/audio_capture.dart';

import 'dart:typed_data';

void main() {
  test('pre-roll retains the latest 300 milliseconds and isolates copies', () {
    final buffer = PcmPreRoll();
    buffer.add(List.filled(6400, 1));
    buffer.add(List.filled(6400, 2));
    final captured = buffer.take();
    expect(captured.length, 9600);
    expect(captured.take(3200).every((x) => x == 1), isTrue);
    expect(captured.skip(3200).every((x) => x == 2), isTrue);
    buffer.clear();
    expect(buffer.take(), isEmpty);
    expect(captured.length, 9600);
  });
  test('WAV header represents PCM16 mono at the true sample rate', () {
    final wav = pcm16Wav(List.filled(32000, 0));
    final data = ByteData.sublistView(wav);
    expect(data.getUint32(24, Endian.little), 16000);
    expect(data.getUint32(40, Endian.little), 32000);
    expect(data.getUint16(22, Endian.little), 1);
    expect(wav.length, 32044);
  });
}
