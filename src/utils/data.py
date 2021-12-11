from typing import List
from typing import Tuple
from typing import Dict
import numpy as np

from src.audio_process.process import AudioConverter


def make_train_dataset(
    audio_data: List[Tuple[np.array, str]],
    converter: AudioConverter,
    n_classes: int = 3,
    data_mapping: Dict = dict()) -> Tuple[np.array, np.array]:
    """音声データ一覧から学習に使う用のデータを作成する

        Parameters
        ----------
        audio_data : List[Tuple]
            1 つ 1 つの要素は (audio:1d-array, label: str) となっている。
            入力となる音声信号データとそれにつけられたラベル('hi-hat','snare' など)
        converter : AudioConverter
            波形データをメルスペクトラムに変換する
        n_classes : int, optional
            予測するクラス数
    
        Returns
        -------
        Tuple[np.array, np.array]
            全ての音声を結合した長いスペクトログラムとフレームごとのラベルを one-hot 形式に変換したもの
    """
    input_spectrogram_list = []
    label_seq_list = []
    chunk = converter.chunk
    for idx, (audio, label) in enumerate(audio_data):
        n_frame = (len(audio) - 1) // chunk
        for i in range(n_frame):
            start = i * chunk
            end = (i + 1) * chunk
            if i == 0:
                converted_audio = converter(audio[:end])  # (N_MELS, 1)
            else:
                converted_audio = converter(audio[start:end])  # (N_MELS, 1)
            input_spectrogram_list.append(converted_audio)
            label_seq_part = np.zeros([n_classes, 1])
            label_seq_part[data_mapping[label], 0] = 1
            label_seq_list.append(label_seq_part)

    input_spectrogram = np.concatenate(input_spectrogram_list, axis=1)
    label_seq = np.concatenate(label_seq_list, axis=1)
    return (input_spectrogram, label_seq)
