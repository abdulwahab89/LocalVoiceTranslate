"""One audio contract: finite float32 mono, with actual resampling."""
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def mono_resample(audio, source_rate, target_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError('Audio must contain finite mono/stereo samples')
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError('Invalid audio sample rate')
    if source_rate != target_rate:
        factor = gcd(source_rate, target_rate)
        audio = resample_poly(audio, target_rate // factor, source_rate // factor)
    return np.ascontiguousarray(audio, dtype=np.float32)


def read_audio(path, sample_rate=16000):
    data, rate = sf.read(str(path), dtype='float32')
    return mono_resample(data, rate, sample_rate)
