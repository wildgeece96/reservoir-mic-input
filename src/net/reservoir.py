import numpy as np


class ESN_2D(object):
    """Echo State Network which has 2 dimensional structure.
    """
    def __init__(self,
                 height=10,
                 width=10,
                 input_dim=80,
                 alpha=0.8,
                 dtype=np.float32):
        self.dtype = dtype
        self.scale = np.sqrt(width * height)
        self.alpha = alpha
        self.width = width
        self.height = height
        self._x = np.random.randn(width * height).astype(dtype)

        self.w_inter = np.random.randn(width * height,
                                       width * height) / self.scale
        self.w_inter.astype(dtype)
        self.set_w_inter(height, width)
        self.w_inter /= np.linalg.norm(self.w_inter)
        self.w_inter *= 0.99

        self.w_in = np.random.randn(input_dim,
                                    width * height) / self.scale * 2.0
        # mask the input weight
        self.w_in *= np.where(
            np.random.rand(input_dim, height * width) < 0.8, 0, 1.0)
        self.w_in.astype(dtype)

        self.g = np.tanh

    def step(self, u):
        """Update status.
        Parameters:
        =========
        u: ndarray. (input_dim,).
        """
        update_value = np.dot(self.w_inter, self._x) + np.dot(u, self.w_in)
        self._x = self.g(update_value)

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
        distance = np.zeros(height * width, dtype=self.dtype)
        for _i in range(height):
            for _j in range(width):
                if _i == i and _j == j:
                    distance[_i * height + _j] = self.alpha
                else:
                    distance[_i * height + _j] = np.sqrt((_i - i)**2 +
                                                         (_j - j)**2)
        return distance
