import numpy as np


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
            input_scale=0.5,
            dtype=np.float32):
        """ネットワークの初期か

        Parameters
        ----------
        height : int, optional
            [description], by default 10
        width : int, optional
            [description], by default 10
        input_dim : int, optional
            [description], by default 80
        output_dim : int, optional
            [description], by default 4
        alpha : float, optional
            [description], by default 0.8
        dtype : [type], optional
            [description], by default np.float32
        """
        self.dtype = dtype
        self.scale = np.sqrt(width * height)
        self.alpha = alpha
        self.width = width
        self.height = height
        self.input_scale = input_scale
        self._x = np.random.randn(width * height).astype(dtype)

        self.w_inter = np.random.randn(width * height,
                                       width * height) / self.scale
        self.w_inter.astype(dtype)
        self.set_w_inter(height, width)
        # echo state property を持たせるための重み調整
        self.w_inter /= np.linalg.norm(self.w_inter)
        self.w_inter *= 0.99

        self.w_in = np.random.randn(input_dim,
                                    width * height) / self.scale * 2.0
        # mask the input weight
        self.w_in *= np.where(
            np.random.rand(input_dim, height * width) < 0.8, 0, 1.0)
        self.w_in.astype(dtype)
        self.w_out = np.random.randn(height * width, output_dim)

        # 活性化関数
        self.g = np.tanh

    def __call__(self, u):
        """Update state."""
        return self.step(u)

    def step(self, u):
        """Update state and return output.

        Parameters:
        =========
        u: ndarray. (input_dim,).
        """
        update_value = np.dot(
            self.w_inter, self._x) + np.dot(u, self.w_in) * self.input_scale
        self._x = self.g(update_value)
        return np.dot(self._x, self.w_out)

    @property
    def x(self):
        return self._x.reshape(self.height, self.width)

    def set_w_inter(self, height, width):
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
                    # alpha は自己結合の度合いなので逆数をとっておく
                    distance[_i * height + _j] = 1. / self.alpha
                else:
                    distance[_i * height + _j] = np.sqrt((_i - i)**2 +
                                                         (_j - j)**2)
        return distance

    def save(self, file_path="network_weights.npz"):
        """ネットワークの重みを保存する"""
        np.savez(file_path,
                 w_in=self.w_in,
                 w_inter=self.w_inter,
                 w_out=self.w_out)

    def load(self, file_path="network_weights.npz"):
        """ネットワークの重みを読み出す
        """
        npz_weights = np.load(file_path)
        self.w_in = npz_weights["w_in"]
        self.w_inter = npz_weights["w_inter"]
        self.w_out = npz_weights["w_out"]
