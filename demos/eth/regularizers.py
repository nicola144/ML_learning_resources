import numpy as np


class Regularizer(object):
    """docstring for Regularizer"""

    def __init__(self, reg):
        super().__init__()
        self._lambda = reg

    def set_lambda(self, reg):
        self._lambda = reg

    def get_lambda(self):
        return self._lambda

    def loss(self, w):
        return 0

    def gradient(self, w):
        return np.zeros_like(w)


class L2Regularizer(Regularizer):
    """docstring for L2Regularizer"""

    def __init__(self, reg):
        super().__init__(reg)

    def loss(self, w):
        return self._lambda * np.square(np.linalg.norm(w[:-1], 2))

    def gradient(self, w):
        gradient = np.zeros_like(w)
        gradient[:-1] = self._lambda * w[:-1]
        return gradient


class L1Regularizer(Regularizer):
    """docstring for L2Regularizer"""

    def __init__(self, reg):
        super().__init__(reg)

    def loss(self, w):
        return self._lambda * np.linalg.norm(w[:-1], 1)

    def gradient(self, w):
        gradient = np.zeros_like(w)
        gradient[:-1] = self._lambda * np.sign(w[:-1])
        return gradient
