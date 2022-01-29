"""SAL を使って重みを学習させるためのリザバー"""
from typing import List
from typing import Dict
import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.autograd import Variable


class SALReservoirLayer(nn.Module):
    """reservoir layer with SAL"""
    def __init__(self,
                 input_dim: int = 100,
                 output_dim: int = 100,
                 act_func: str = "tanh"):
        super(SALReservoirLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act_func_str = act_func
        if act_func == "tanh":
            self.act = torch.tanh
            self.inv_act = lambda u, o: 1 - o**2
        else:
            raise ValueError("Invalid function type '%s' only valid on '%s'" %
                             (act_func, "tanh"))

    def forward(self, x):
        u = self.linear(x)
        x = self.act(u)
        s_each_w = self.inv_act(u, x) * torch.sqrt(
            torch.sum(self.linear.weight**2, dim=1))
        sensitivity = torch.mean(s_each_w)
        return x, sensitivity

    def save(self, file_path: str) -> Dict:
        """Layer を ReservoirLayer クラスで読み込める形で保存する

        Parameters
        ----------
        file_path : str
            重みデータを保存するnpzファイル

        Returns
        -------
        Dict
            この layer のメタ情報
        """
        config = {}
        w = self.linear.weight.to('cpu').detach().numpy().copy()
        bias = self.linear.bias.to('cpu').detach().numpy().copy()
        config["input_dim"] = w.shape[1]
        config["output_dim"] = w.shape[0]
        config["act_func"] = self.act_func_str
        np.savez(file_path, w=w, bias=bias)
        return config


class SALReservoir(nn.Module):
    """reservoir with SAL"""
    def __init__(
        self,
        dims: List[int] = [100, 100],
        beta: float = 0.99,  # sensitivity を計算するときの移動平均の度合い
        leaky_rate: float = 0.8  # どれぐらいの比率で前ステップの状態を持っておくか
    ):
        super(SALReservoir, self).__init__()
        assert dims[0] == dims[-1], \
            "The first and end values of dims should be the same but different %f vs %f" % (
            dims[0], dims[-1])
        layers = []
        for i in range(len(dims) - 1):
            layers.append(
                SALReservoirLayer(dims[i], dims[i + 1], act_func="tanh"))
        self.layers = layers
        self.n_layers = len(layers)
        self.beta = beta
        self.leaky_rate = leaky_rate
        self.dims = dims

    def forward(self,
                x,
                prev_states: List[torch.Tensor],
                prev_s: torch.Tensor = 0.0):
        sensitivity_list = torch.empty(self.n_layers)
        states = []
        for i, layer in enumerate(self.layers):
            _x, _s = layer(x)
            x = self.leaky_rate * prev_states[i] + (1 - self.leaky_rate) * _x
            states.append(x)
            sensitivity_list[i] = _s
        s = torch.mean(sensitivity_list)
        s_bar = self.beta * prev_s + (1 - self.beta) * s
        return x, s_bar, states

    def generate_init_state(self):
        return [
            Variable(torch.Tensor(np.random.randn(dim)))
            for dim in self.dims[1:]
        ]

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


def export_sal_trained_reservoir(reservoir: SALReservoir, w_in: np.array,
                                 config: Dict, dir_path: str) -> None:
    """SAL で学習させた Reservoir を ESN クラスで読み込めるようにファイル出力をする

    Parameters
    ----------
    reservoir : SALReservoir
        リザバー
    w_in : np.array
        入力のマッピングに使った重み
    config : Dict
        追加の設定情報
    """
    os.makedirs(dir_path, exist_ok=True)
    reservoir.save(dir_path)
    input_dim = w_in.shape[0]
    output_dim = config["output_dim"]
    reservoir_dims = reservoir.dims
    leaky_rate = reservoir.leaky_rate
    input_scale = config["input_scale"]
    input_offset = config["input_offset"]

    dtype_name = config["dtype"] if type(
        config["dtype"]) == str else config["dtype"].__name__
    config = {
        "reservoir_dims": reservoir_dims,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "leaky_rate": leaky_rate,
        "dtype": dtype_name,
        "input_offset": input_offset,
        "input_scale": input_scale,
    }

    file_path = os.path.join(dir_path, "network_weights.npz")
    config_path = os.path.join(dir_path, "network_configs.json")
    np.savez(file_path, w_in=w_in)
    with open(config_path, "w") as f:
        json_strings = json.dumps(config, indent=4)
        f.write(json_strings)
