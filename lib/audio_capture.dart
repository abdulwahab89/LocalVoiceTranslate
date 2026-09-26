import 'dart:typed_data';

/// PCM16 mono ring used only before push-to-talk; no audio is sent until pressed.
class PcmPreRoll {
  PcmPreRoll({this.maxBytes = 9600}); // 300 ms at 16 kHz, PCM16 mono
  final int maxBytes;
  final List<int> _bytes = [];
  void add(List<int> bytes) {
    _bytes.addAll(bytes);
    if (_bytes.length > maxBytes) {
      _bytes.removeRange(0, _bytes.length - maxBytes);
    }
  }

  List<int> take() => List<int>.of(_bytes);
  void clear() => _bytes.clear();
}

Uint8List pcm16Wav(List<int> pcm, {int sampleRate = 16000}) {
  if (pcm.length.isOdd) throw ArgumentError('Incomplete PCM16 sample');
  final result = Uint8List(44 + pcm.length);
  final header = ByteData.sublistView(result);
  void text(int offset, String value) =>
      result.setRange(offset, offset + value.length, value.codeUnits);
  text(0, 'RIFF');
  header.setUint32(4, 36 + pcm.length, Endian.little);
  text(8, 'WAVE');
  text(12, 'fmt ');
  header.setUint32(16, 16, Endian.little);
  header.setUint16(20, 1, Endian.little);
  header.setUint16(22, 1, Endian.little);
  header.setUint32(24, sampleRate, Endian.little);
  header.setUint32(28, sampleRate * 2, Endian.little);
  header.setUint16(32, 2, Endian.little);
  header.setUint16(34, 16, Endian.little);
  text(36, 'data');
  header.setUint32(40, pcm.length, Endian.little);
  result.setRange(44, result.length, pcm);
  return result;
}
