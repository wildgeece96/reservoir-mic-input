from typing import List
from typing import Any
from typing import Tuple
from typing import Dict
from collections import defaultdict
import numpy as np
import random

from src.audio_process import AudioConverter


class DataSet(object):
    """学習用のデータセットクラス"""
    def __init__(self, input_data: List[Any], label_data: List[Any]):
        self.input_data = input_data
        self.label_data = label_data

    def __getitem__(self, idx):
        return (self.input_data[idx], self.label_data[idx])

    def __len__(self):
        return len(self.input_data)


class AudioDataSet(DataSet):
    """スペクトログラムをランダムにつなぎ合わせて取得することを想定している DataSet クラス"""
    def __init__(self,
                 spectrograms: List[np.array],
                 label_seqs: List[np.array],
                 places: List[str],
                 silent_label: str = "silent"):
        super(AudioDataSet, self).__init__(spectrograms, label_seqs)
        self._places = places
        class_index = defaultdict(list)
        label_set = set()
        for i in range(len(self)):
            spectrogram, label_seq, place = self[i]
            label = int(label_seq[0])  # label_seq は同じ値が続いている想定
            class_index[f"{label:02d}-{place}"].append(i)
            label_set.add(label)
        self._class_index = class_index
        self._label_list = list(label_set)
        self._place_list = list(set(places))
        self._silent_label = silent_label

    def __getitem__(self, idx):
        return (self.input_data[idx], self.label_data[idx], self._places[idx])

    @property
    def label_list(self):
        return self._label_list

    @property
    def place_list(self):
        return self._place_list

    @property
    def silent_label_idx(self):
        if self._silent_label in self._label_list:
            return self._label_list.index(self._silent_label)
        else:
            return -1

    def get_random_item(self, label: int = 0, place: str = "loft"):
        """指定されたラベルのデータでランダムなものを返す

        Parameters
        ----------
        label : int, optional
            ラベル, by default 0
        place : str, optional
            収録した場所, by default "loft"

        Returns
        -------
        Tuple[spectrogram: np.array, label_seq: np.array]
            ランダムに選択されたデータ

        Raises
        ------
        ValueError
            [description]
        """
        if label not in self._label_list:
            raise ValueError("Invalid value %s please specify among {%s}",
                             label, self._label_set)
        index = random.choice(self._class_index[f"{label:02d}-{place}"])
        return self[index]


class ESNDataGenerator(object):
    """ESN を学習させるためのデータセットを生成するクラス. PyTorch ライクに使う"""
    def __init__(self, dataset: AudioDataSet, epochs: int, num_concat: int):
        """初期化関数

        Parameters
        ----------
        dataset : AudioDataSet
            データセットクラス
        epochs : int
            学習を回すエポック数
        num_concat : int
            1 サンプルあたり、いくつの音声データを連続させるか
        """
        self._dataset = dataset
        self.iter_count = 0
        self._epochs = epochs
        self.max_iter_count = self._epochs * len(self)
        self._num_concat = num_concat

    @property
    def dataset(self):
        return self._dataset

    @property
    def num_concat(self):
        return self._num_concat

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.array, np.array]:
        """concatenate multiple audio samples to make training sample.

        Returns
        -------
        Tuple[np.array, np.array]
            spectrograms: (n_mels, n_frame)
            label_seqs: (n_frame)

        Raises
        ------
        StopIteration
            [description]
        """
        if self.iter_count > self.max_iter_count:
            self.iter_count = 0
            raise StopIteration
        self.iter_count += 1
        spectrograms = []
        label_seqs = []
        selected_place = random.choice(self._dataset.place_list)
        for i in range(self.num_concat):
            random_label = random.choice(self.dataset.label_list)
            spectrogram, label_seq, _ = self.dataset.get_random_item(
                label=random_label,
                place=selected_place)  # (n_mels, n_frame), (n_frame)
            assert spectrogram.shape[1] == label_seq.shape[
                0], "The length of spectrogram and label_seq are different %s vs %s" % (
                    spectrogram.shape, label_seq.shape)
            spectrograms.append(spectrogram)
            label_seqs.append(label_seq)
        spectrograms = np.concatenate(spectrograms, axis=1)
        label_seqs = np.concatenate(label_seqs, axis=0)
        return (spectrograms, label_seqs)


def make_audio_dataset(
    audio_data: List[Tuple[np.array, str, str]],
    converter: AudioConverter,
    n_classes: int = 3,
    data_mapping: Dict = dict()) -> AudioDataSet:
    """音声データ一覧から学習に使う用のデータを作成する

        Parameters
        ----------
        audio_data : List[Tuple]
            1 つ 1 つの要素は (audio:1d-array, label: str, place: str) となっている。
            入力となる音声信号データとそれにつけられたラベル('hi-hat','snare' など)
        converter : AudioConverter
            波形データをメルスペクトラムに変換する
        n_classes : int, optional
            予測するクラス数
    
        Returns
        -------
        Tuple[input_spectrogram: List[array(n_mels, n_frame)], seq_labels: List[array(n_frame, n_samples)]
            スペクトログラムとフレームごとのラベル列
    """
    input_spectrograms = []
    label_seqs = []
    places = []
    chunk = converter.chunk
    for idx, (audio, label, place) in enumerate(audio_data):
        places.append(place)
        n_frame = (len(audio) - 1) // chunk
        input_spectrogram_item = []
        label_seq = []
        for i in range(n_frame):
            start = i * chunk
            end = (i + 1) * chunk
            if i == 0:
                converted_audio = converter(audio[:end])  # (n_mels, n_frame)
            else:
                converted_audio = converter(
                    audio[start:end])  # (n_mels, n_frame)
            input_spectrogram_item.append(converted_audio)
            label_seq_part = np.array([data_mapping[label]] *
                                      converted_audio.shape[1],
                                      dtype=np.float32)
            label_seq.append(label_seq_part)
        input_spectrograms.append(
            np.concatenate(input_spectrogram_item, axis=1))
        label_seqs.append(np.concatenate(label_seq, axis=0))
    return AudioDataSet(input_spectrograms, label_seqs, places)
