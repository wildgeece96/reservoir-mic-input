import os
import json
from typing import Dict
import numpy as np
import joblib
import sklearn
from sklearn.linear_model import Ridge


class ESN_2D(object):
    """Echo State Network which has 2 dimensional structure.
    """
    def __init__(
            self,
            height=10,
            width=10,
            input_dim=80,
            output_dim=4,
            alpha=0.8,  # 自己状態の保存の度合い
            dtype='float32',
            decoder=None):
        """ネットワークの初期か

        Parameters
        ----------
        height : int, optional
            ネットワークの高さ(2次元構造にノードを配置する想定), by default 10
        width : int, optional
            ネットワークの幅（2次元構造にノードを配置する想定), by default 10
        input_dim : int, optional
            入力の次元数, by default 80
        output_dim : int, optional
            ネットワークの出力次元数, by default 4
        alpha : float, optional
            直前の state を持つ割合, input の scale は 1 - slpha となる, by default 0.8
        dtype : str, optional
            network 内部で持つ数値データの型 by default 'float32'
        decoder : sklearn.model, optional
            内部状態をデコードするクラス。sklearn 上で取得できるモデルを想定
        """
        self.dtype = dtype
        self.scale = np.sqrt(width * height)
        self.alpha = alpha
        self.height = height
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._x = np.random.randn(width * height).astype(dtype)

        self.w_inter = np.random.randn(width * height,
                                       width * height) / self.scale
        self.w_inter.astype(dtype)
        self._adjust_w_inter_params(height, width)
        # echo state property を持たせるための重み調整
        self.w_inter /= np.linalg.norm(self.w_inter)
        self.w_inter *= 0.99

        self.w_in = np.random.randn(input_dim,
                                    width * height) / self.scale * 2.0
        # mask the input weight
        self.w_in *= np.where(
            np.random.rand(input_dim, height * width) < 0.8, 0, 1.0)
        self.w_in.astype(dtype)

        # 活性化関数
        self.g = np.tanh

        # デコーダー
        self.decoder = decoder

    def __call__(self, u, return_preds=False):
        """Update state."""
        return self.step(u, return_preds)

    def step(self, u, return_preds=False):
        """Update state and return output.

        Parameters:
        =========
        u: ndarray. (input_dim,).
        """
        u = u.astype(self.dtype)
        updated_value = self.alpha * np.dot(
            self.w_inter, self._x) + (1. - self.alpha) * np.dot(u, self.w_in)
        self._x = self.g(updated_value)
        if return_preds:
            return self.decoder.predict(self._x)

    @property
    def x(self):
        return self._x.reshape(self.height, self.width)

    @property
    def x_flatten(self):
        return self._x

    @property
    def config(self):
        """ネットワークのconfig部分の書き出し"""
        dtype_name = self.dtype if type(
            self.dtype) == str else self.dtype.__name__
        return {
            "height": self.height,
            "width": self.width,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "alpha": self.alpha,
            "dtype": self.dtype
        }

    def set_decoder(self, decoder: sklearn.linear_model):
        self.decoder = decoder

    def _adjust_w_inter_params(self, height, width):
        # 格子状に並べたニューロンの結合をニューロン同士の距離にしたがって結合の強さを調節する
        for i in range(height):
            for j in range(width):
                distance = self._calc_distance(i, j, height, width)
                self.w_inter[i * height + j] /= distance

    def _calc_distance(self, i, j, height, width):
        # ニューロン同士の距離を計算する
        distance = np.zeros(height * width, dtype=self.dtype) + 1e-3
        for _i in range(height):
            for _j in range(width):
                if _i == i and _j == j:
                    distance[_i * height + _j] = 1.
                else:
                    distance[_i * height + _j] = np.sqrt((_i - i)**2 +
                                                         (_j - j)**2)
        return distance

    def save(self, dir_path="./"):
        """ネットワークの重みを保存する"""
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, "network_weights.npz")
        config_path = os.path.join(dir_path, "network_configs.json")
        decoder_path = os.path.join(dir_path, "decoder.pkl")
        np.savez(file_path, w_in=self.w_in, w_inter=self.w_inter)
        with open(config_path, "w") as f:
            json_strings = json.dumps(self.config, indent=4)
            f.write(json_strings)

        joblib.dump(self.decoder, decoder_path)

    def _load_weights(self, file_path="network_weights.npz"):
        """保存された情報からネットワークを復元する
        """
        npz_weights = np.load(file_path)
        self.w_in = npz_weights["w_in"]
        self.w_inter = npz_weights["w_inter"]

    def _load_decoder(self, file_path="decoder.pkl"):
        """保存された情報からデコーダーを復元する
        """
        self.decoder = joblib.load(file_path)

    @classmethod
    def load(cls, dir_path="./"):
        """保存された情報を読み込んだネットワークを返す

        Parameters
        ----------
        dir_path : str, optional
            ネットワーク情報が保存された場所, by default "./"
        """
        file_path = os.path.join(dir_path, "network_weights.npz")
        config_path = os.path.join(dir_path, "network_configs.json")
        decoder_path = os.path.join(dir_path, "decoder.pkl")
        with open(config_path, "r") as f:
            config = json.load(f)
        loaded_network = cls(**config)
        loaded_network._load_weights(file_path)
        loaded_network._load_decoder(decoder_path)
        return loaded_network
