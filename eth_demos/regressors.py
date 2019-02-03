import numpy as np


class LinearRegressor(object):
    def __init__(self, X, Y):
        self._Xtr = X
        self._Ytr = Y
        self._w = np.random.randn(X.shape[1])

    def get_number_samples(self):
        return self._Xtr.shape[0]

    def set_weights(self, w):
        self._w = w

    def get_weights(self):
        return self._w

    def load_data(self, Xtr, Ytr=None):
        self._Xtr = Xtr
        self._Ytr = Ytr

    def calculate_weights(self):
        dim = self._Xtr.shape[1]
        self._w = np.dot(np.linalg.pinv(np.dot(self._Xtr.T, self._Xtr)
                                        + self._lambda * np.eye(dim)), np.dot(self._Xtr.T, self._Ytr))
        return self._w

    def predict(self, X):
        return np.dot(X, self._w)

    def test_loss(self, w, X, Y):
        w_old = self.get_weights()
        self.set_weights(w)
        error = self.predict(X) - Y

        self.set_weights(w_old)
        return np.dot(error.T, error)

    def loss(self, w, indexes):
        self.set_weights(w)
        error = self.predict(self._Xtr[indexes, :]) - self._Ytr[indexes]
        return np.dot(error.T, error) / indexes.size  # + 1./2 * self._lambda * np.dot(self._w, self._w)

    def gradient(self, w, indexes):
        self.set_weights(w)
        error = self.predict(self._Xtr[indexes, :]) - self._Ytr[indexes]
        return np.dot(self._Xtr[indexes, :].T, error) / indexes.size  # + self._lambda * w
