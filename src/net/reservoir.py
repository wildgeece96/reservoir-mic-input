"""リザバー部分(ESN)を実装するためのモジュールを定義する"""
import os
import json
from typing import Dict
from typing import Tuple
from typing import List
import numpy as np
import joblib
import sklearn
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression


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
            input_offset=4.5,
            sparse_rate=0.80,
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
            直前の state を持つ割合, input の scale は 1 - alpha となる, by default 0.8
        input_offset : float, optional
            音声が全て負の値のため、固定値としてどれぐらい足すかを指定する
        sparse_rate : float, optional 
            内部結合をスパースにどの程度するかを指定する。 1 に近いほどよりスパースになる
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
        self.input_offset = input_offset
        self.sparse_rate = sparse_rate
        self._x = np.random.randn(width * height).astype(dtype)

        self.w_inter = (np.random.rand(width * height, width * height) -
                        0.5) / self.scale * 2.0
        self.w_inter.astype(dtype)
        self._adjust_w_inter_params(height, width)
        self._make_w_inter_sparse()
        # echo state property を持たせるための重み調整
        self.w_inter /= np.linalg.eig(self.w_inter)[0].max()
        self.w_inter *= 0.99

        self.w_in = np.random.randn(input_dim,
                                    width * height) / self.scale * 2.0
        # mask the input weight
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
        u += self.input_offset
        updated_value = self.alpha * np.dot(
            self.w_inter, self._x) + (1. - self.alpha) * np.dot(u, self.w_in)
        self._x = self.g(updated_value)
        if return_preds:
            return self.decoder.predict(self._x.reshape(1, -1)).flatten()

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
            "dtype": self.dtype,
            "input_offset": self.input_offset,
            "sparse_rate": self.sparse_rate
        }

    def set_decoder(self, decoder: sklearn.linear_model):
        self.decoder = decoder

    def _make_w_inter_sparse(self):
        """内部結合をスパースなものに変換する"""
        prob_map = np.random.rand(*self.w_inter.shape)
        sparse_map = np.where(prob_map > self.sparse_rate, 1., 0.)
        self.w_inter *= sparse_map

    def _adjust_w_inter_params(self, height, width):
        # 格子状に並べたニューロンの結合をニューロン同士の距離にしたがって結合の強さを調節する
        for i in range(height):
            for j in range(width):
                distance = self._calc_distance(i, j, height, width)
                self.w_inter[i * width + j] /= distance

    def _calc_distance(self, i, j, height, width):
        # ニューロン同士の距離を計算する
        distance = np.zeros(height * width, dtype=self.dtype) + 1e-3
        for _i in range(height):
            for _j in range(width):
                if _i == i and _j == j:
                    distance[_i * width + _j] = 1.
                else:
                    distance[_i * width + _j] = np.sqrt((_i - i)**2 +
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
        config_path = os.path.join(dir_path, "network_configs.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        loaded_network = cls(**config)
        decoder_path = os.path.join(dir_path, "decoder.pkl")
        file_path = os.path.join(dir_path, "network_weights.npz")
        loaded_network._load_weights(file_path)
        loaded_network._load_decoder(decoder_path)
        return loaded_network


class ReservoirLayer(object):
    """reservoir layer"""
    def __init__(self,
                 input_dim: int = 100,
                 output_dim: int = 100,
                 act_func: str = "tanh"):
        w = (np.random.rand(output_dim, input_dim) - 0.5) / (output_dim *
                                                             input_dim / 2.0)
        # 入力と出力の次元が同じ時のみ、スペクトル半径の調整を行う
        if input_dim == output_dim:
            w = w / np.linalg.eig(w)[0].max() * 0.99
        self.bias = np.random.zeros(output_dim)
        self.act_func_str = act_func
        if act_func == "tanh":
            self.act_func = np.tanh
        else:
            raise ValueError("Invalid activate function type %s", act_func)

    def __call__(self, x: np.array):
        return self.step(x)

    def step(self, x: np.array, prev_state: np.array = None):
        u = np.dot(self.w, x) + self.bias
        return self.act_func(u)

    @classmethod
    def load(cls, config: Dict, file_path: str):
        """定義済みの情報から読み込む

        Parameters
        ----------
        config : Dict
            このレイヤーのメタ情報
        w : np.array
            結合重み
        bias : np.array
            バイアスの重み
        """
        layer = cls(**config)
        npz = np.load(file_path)
        layer.w = npz["w"]
        layer.bias = npz["bias"]
        return layer

    def save(self, file_path: str):
        """layer を保存できるようにする.基本的には Reservoir クラスと一緒の使用を想定

        Parameters
        ----------
        file_path : str
            [description]
        """
        config = {}
        config["input_dim"] = self.w.shape[1]
        config["output_dim"] = self.w.shape[0]
        config["act_func"] = self.act_func_str
        np.savez(file_path, w=self.w, bias=self.bias)
        return config


class Reservoir(object):
    """reservoir part of ESN. 入力のマッピングは別途行う必要があり、ここでは内部状態の更新のみを行う"""
    def __init__(self, dims: List[int] = [5, 5], leaky_rate: float = 0.0):
        """初期化

        Parameters
        ----------
        dims : List[int], optional
            リザバーのレイヤーの次元数, by default [5, 5]
        leaky_rate : float, optional
            直前の状態をどれほど保持するか、その比率, by default 0.0
        """
        assert len(dims) > 1, "The length of dims should be longer than 2."
        assert dims[0] == dims[-1],\
            "The first and end values of dims should be the same but different %f vs %f" % (
            dims[0], dims[-1])
        layers = []
        for i in range(len(dims) - 1):
            layers.append(ReservoirLayer(dims[i], dims[i + 1],
                                         act_func="tanh"))
        self.layers = layers
        self.leaky_rate = leaky_rate
        self.dims = dims

    def __call__(self, x: np.array, prev_states: List[np.array]):
        return self.forward(x, prev_states)

    def forward(
            self, x: np.array,
            prev_states: List[np.ndarray]) -> Tuple[np.array, List[np.array]]:
        """情報の処理を行う. 入力の x は 入力のマッピングされた値と直前の state x を足し合わせたもの"""
        states = []
        for i, layer in enumerate(self.layers):
            _x = layer(x)
            x = self.leaky_rate * prev_states[i] + (1 - self.leaky_rate) * _x
            states.append(x)
        return (x, states)

    @property
    def n_layers(self):
        return len(self.layers)

    def save(self, dir_path: str):
        """Reservoir を保存する

        Parameters
        ----------
        dir_path : str
            Reservoir の情報を保存したいディレクトリへの path
        """
        layer_configs = []
        for idx, layer in enumerate(self.layers):
            layer_weight_file_path = os.path.join(dir_path,
                                                  f"layer_{idx:02d}.npz")
            layer_config = layer.save(layer_weight_file_path)
            layer_configs.append({
                "config": layer_config,
                "file_path": layer_weight_file_path
            })
        config = {}
        config["meta_data"] = {
            "leaky_rate": self.leaky_rate,
            "dims": self.dims,
        }
        config["layer_configs"] = layer_configs
        config_path = os.path.join(dir_path, "reservoir_config.json")
        with open(config_path, "w") as f:
            json_string = json.dumps(config, indent=4)
            f.write(json_string)

    def generate_init_state(self) -> List[np.array]:
        """内部状態の初期値を生成する

        Returns
        -------
        List[np.array]
            初期値
        """
        return [np.random.randn(dim) for dim in self.dims[1:]]

    @classmethod
    def load(cls, dir_path: str):
        """保存された Reservoir 情報を読み込む"""
        config_path = os.path.join(dir_path, "reservoir_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        reservoir = cls(**config["meta_data"])
        layers = []
        for layer_config in config["layer_configs"]:
            layers.append(ReservoirLayer.load(**layer_config))
        reservoir.layers = layers
        return reservoir


class ESN(object):
    """Echo State Network which has 2 dimensional structure.
    """
    def __init__(
            self,
            reservoir_dims: List[int],
            input_dim: int,
            output_dim: int = 4,
            leaky_rate: float = 0.8,  # 自己状態の保存の度合い
            input_offset: float = 4.5,
            input_scale: float = 0.01,
            dtype='float32',
            decoder=None):
        """ネットワークの初期か

        Parameters
        ----------
        reservoir_dims : List[int]
            リザバー部分のネットワークサイズを情報伝播の順番に記載したもの
        input_dim : int
            入力の次元数
        output_dim : int, optional
            ネットワークの出力次元数, by default 4
        leaky_rate : float, optional
            直前の state を持つ割合, input の scale は 1 - alpha となる, by default 0.8
        input_offset : float, optional
            音声が全て負の値のため、固定値としてどれぐらい足すかを指定する
        dtype : str, optional
            network 内部で持つ数値データの型 by default 'float32'
        decoder : sklearn.model, optional
            内部状態をデコードするクラス。sklearn 上で取得できるモデルを想定
        """
        self.dtype = dtype
        self.leaky_rate = leaky_rate
        self.reservoir_dims = reservoir_dims
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_offset = input_offset
        self.input_scale = input_scale

        self.w_in = np.random.rand(input_dim, reservoir_dims[0]) * input_scale

        # デコーダー
        self.decoder = decoder
        # リザバー部分
        self.reservoir = Reservoir(dims=reservoir_dims, leaky_rate=leaky_rate)
        self._states = self.reservoir.generate_init_state()

    def __call__(self, x, u, return_preds=False):
        """Update state."""
        return self.step(x, u, return_preds)

    def step(self,
             x: np.array,
             u: np.array,
             return_preds: bool = False,
             initialize_state: bool = False) -> Tuple[np.array, np.array]:
        """Update state and return output.

        Parameters
        ----------
        x : np.array, (dims[0], )
            直前のステップで得られたネットワークの状態
        u : np.array, (input_dim, )
            入力となる信号
        return_preds : bool, optional
            decoder の推測値を返すかどうか, by default False
        initialize_state : bool, optional
            reservoir の内部状態をリセットするかどうか, by default False

        Returns
        -------
        Tuple[np.array, np.array]
            The updated state and prediction generated by decoder.
        """
        if initialize_state:
            self._states = self.reservoir.generate_init_state()
        u = u.astype(self.dtype)
        u += self.input_offset
        x += np.dot(u, self.w_in)
        x, states = self.reservoir(x, self._states)
        self._states = states
        if return_preds:
            preds = self.decoder.predict(self._x.reshape(1, -1)).flatten()
        else:
            preds = None
        return (x, preds)

    @property
    def states(self):
        return self._states

    @property
    def config(self):
        """ネットワークのconfig部分の書き出し"""
        dtype_name = self.dtype if type(
            self.dtype) == str else self.dtype.__name__
        return {
            "reservoir_dims": self.reservoir_dims,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "leaky_rate": self.leaky_rate,
            "dtype": self.dtype,
            "input_offset": self.input_offset,
            "input_scale": self.input_scale,
        }

    def set_decoder(self, decoder: sklearn.linear_model):
        self.decoder = decoder

    def save(self, dir_path="./"):
        """ネットワークの重みを保存する"""
        os.makedirs(dir_path, exist_ok=True)
        # リザバー部分の保存
        self.reservoir.save(dir_path)
        # リザバー部分の外側の保存
        file_path = os.path.join(dir_path, "network_weights.npz")
        config_path = os.path.join(dir_path, "network_configs.json")
        decoder_path = os.path.join(dir_path, "decoder.pkl")
        np.savez(file_path, w_in=self.w_in)
        with open(config_path, "w") as f:
            json_strings = json.dumps(self.config, indent=4)
            f.write(json_strings)
        joblib.dump(self.decoder, decoder_path)

    def _load_weights(self, file_path="network_weights.npz"):
        """保存された情報からネットワークを復元する
        """
        npz_weights = np.load(file_path)
        self.w_in = npz_weights["w_in"]

    def _load_decoder(self, file_path="decoder.pkl"):
        """保存された情報からデコーダーを復元する
        """
        self.decoder = joblib.load(file_path)

    def _load_reservoir(self, dir_path="./"):
        """リザバー部分の読み込み
        """
        self.reservoir = Reservoir.load(dir_path)

    @classmethod
    def load(cls, dir_path="./"):
        """保存された情報を読み込んだネットワークを返す

        Parameters
        ----------
        dir_path : str, optional
            ネットワーク情報が保存された場所, by default "./"
        """
        config_path = os.path.join(dir_path, "network_configs.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        loaded_network = cls(**config)
        decoder_path = os.path.join(dir_path, "decoder.pkl")
        file_path = os.path.join(dir_path, "network_weights.npz")
        loaded_network._load_weights(file_path)
        if os.path.exists(decoder_path):
            loaded_network._load_decoder(decoder_path)
        loaded_network._load_reservoir(dir_path)
        return loaded_network
