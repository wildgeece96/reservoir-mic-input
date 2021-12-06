"""マイクから取得された波形データ処理の共通部分"""
import numpy as np
import librosa


class AudioConverter(object):
    """マイクからの音声をメルスペクトラムに変換するclass"""
    def __init__(self, chunk, n_mels, sr):
        self.chunk = chunk
        self.mel_basis = librosa.filters.mel(sr, n_fft=chunk, n_mels=n_mels)
        self.mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=sr // 2)

    def convert_to_mel(self, data):
        data_fft = np.abs(np.fft.fft(data) / 2**16)
        data_fft = data_fft[:self.chunk // 2 + 1]
        return np.log10(np.dot(self.mel_basis, data_fft.reshape(-1, 1)))
