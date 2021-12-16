"""マイクから取得された波形データ処理の共通部分"""
import numpy as np
import librosa


class AudioConverter(object):
    """マイクからの音声をメルスペクトラムに変換するclass"""
    def __init__(self, chunk_size: int, n_fft: int, n_mels: int,
                 sample_rate: int):
        """クラスの初期化

        Parameters
        ----------
        chunk_size : int
            変換時に入力される音声波形のチャンクサイズ。
        n_fft : int
            音声データを変換するときに fft をかける単位
        n_mels : int
            メルスペクラムとして出力する時の次元数. (n_mels, n_frame) が変換結果となる
        sample_rate : int
            サンプリングレート
        """
        self.chunk = chunk_size
        self.n_fft = n_fft
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft,
            n_mels=n_mels)  # (n_mels, n_fft//2 + 1)
        self.mel_freqs = librosa.mel_frequencies(n_mels=n_mels,
                                                 fmax=sample_rate // 2)
        self.scale = 1.0
        self.sample_rate = sample_rate
        # 今回は処理を簡略化するために n_fft で余る部分については padding を行う
        self.n_padding = n_fft - chunk_size % n_fft
        self.n_frame = (chunk_size + self.n_padding) // n_fft

    def convert_to_mel(self, data) -> np.array:
        """波形データをメルスペクトラムに変換する. データは (n_mels, n_frame) の形式"""
        if data.max() > self.scale:
            self.scale = data.max()
        data = data / self.scale
        data_2d = np.concatenate([data, np.zeros(self.n_padding)],
                                 axis=-1).reshape(
                                     -1, self.n_fft)  # (n_frame, n_fft)
        data_fft = np.abs(np.fft.fft(data_2d, axis=1))  # (n_frame, n_fft)
        data_fft = data_fft[:, :self.n_fft // 2 + 1]
        return np.log10(np.dot(self.mel_basis, data_fft.T) +
                        1e-9)  # (n_mels, n_frame)

    def __call__(self, data: np.array) -> np.array:
        """波形データをメルスペクトラムに変換する"""
        return self.convert_to_mel(data)
