import numpy as np
from util import dist


class Classifier(object):
    """docstring for Classifier"""

    def __init__(self, X, Y):
        super().__init__()
        self._Xtr = X
        self._Ytr = Y
        self._Xtest = None
        self._Ytest = None
        self._w = None
        self._class_cost = np.array([1, 1])

    def load_data(self, X, Y):
        self._Xtr = X
        self._Ytr = Y

    def load_test_data(self, X, Y):
        self._Xtest = X
        self._Ytest = Y 

    def set_weights(self, w):
        self._w = w

    def set_class_cost(self, cost_array):
        self._class_cost = cost_array

    def get_number_samples(self):
        return self._Xtr.shape[0]

    def predict(self, X, w=None):
        pass

    def loss(self, w, indexes):
        pass

    def gradient(self, w, indexes):
        pass

    def test_loss(self, w):
        pass


class Perceptron(Classifier):
    """docstring for Perceptron"""

    def __init__(self, X, Y):
        super().__init__(X, Y)
        self._w = np.random.randn(X.shape[1])

    def predict(self, X, w=None):
        if w is None:
            w = self._w
        z = np.dot(X, w)
        return np.sign(z)

    def loss(self, w, indexes=None):
        if indexes is None:
            indexes = np.arange(0, self.get_number_samples(), 1)
        error = -np.dot(self._Xtr[indexes, :], w) * self._Ytr[indexes]
        error[error < 0] = 0.

        error_idx = ((self._Ytr[indexes][error > 0] + 1) / 2).astype(
            np.int)  # (y+1)/2 maps {-1,1} to {0, 1} for indexing
        weighted_error = self._class_cost[error_idx] * error[error > 0]

        return np.sum(weighted_error) / indexes.size

    def gradient(self, w, indexes=None):
        if indexes is None:
            indexes = np.arange(0, self.get_number_samples(), 1)
        error = -np.dot(self._Xtr[indexes, :], w) * self._Ytr[indexes]
        gradient = -self._Xtr[indexes, :] * self._Ytr[indexes, np.newaxis]
        gradient[error < 0] = 0

        error_idx = ((self._Ytr[indexes][error > 0] + 1) / 2).astype(
            np.int)  # (y+1)/2 maps {-1,1} to {0, 1} for indexing
        weighted_grad = self._class_cost[error_idx, np.newaxis] * gradient[error > 0]

        return np.sum(weighted_grad, axis=0)

    def test_loss(self, w):
        error = -np.dot(self._Xtest, w) * self._Ytest
        error[error < 0] = 0.

        error_idx = ((self._Ytest[error > 0] + 1) / 2).astype(np.int)  # (y+1)/2 maps {-1,1} to {0, 1} for indexing
        weighted_error = self._class_cost[error_idx] * error[error > 0]

        return np.sum(weighted_error) / self._Ytest.size


class SVM(Classifier):
    """docstring for Perceptron"""

    def __init__(self, X, Y):
        super().__init__(X, Y)
        self._w = np.random.randn(X.shape[1])

    def predict(self, X, w=None):
        if w is None:
            w = self._w
        z = np.dot(X, w)
        return np.sign(z)

    def loss(self, w, indexes=None):
        if indexes is None:
            indexes = np.arange(0, self.get_number_samples(), 1)
        error = 1 - np.dot(self._Xtr[indexes, :], w) * self._Ytr[indexes]
        error[error < 0] = 0

        error_idx = ((self._Ytr[indexes][error > 0] + 1) / 2).astype(
            np.int)  # (y+1)/2 maps {-1,1} to {0, 1} for indexing
        weighted_error = self._class_cost[error_idx] * error[error > 0]

        return np.sum(weighted_error) / indexes.size

    def gradient(self, w, indexes=None):
        if indexes is None:
            indexes = np.arange(0, self.get_number_samples(), 1)
        error = 1 - np.dot(self._Xtr[indexes, :], w) * self._Ytr[indexes]
        gradient = -self._Xtr[indexes, :] * self._Ytr[indexes, np.newaxis]
        gradient[error < 0] = 0

        error_idx = ((self._Ytr[indexes][error > 0] + 1) / 2).astype(
            np.int)  # (y+1)/2 maps {-1,1} to {0, 1} for indexing
        weighted_grad = self._class_cost[error_idx, np.newaxis] * gradient[error > 0]

        return np.sum(weighted_grad, axis=0)

    def test_loss(self, w):
        error = 1 - np.dot(self._Xtest, w) * self._Ytest
        error[error < 0] = 0

        error_idx = ((self._Ytest[error > 0] + 1) / 2).astype(
            np.int)  # (y+1)/2 maps {-1,1} to {0, 1} for indexing
        weighted_error = self._class_cost[error_idx] * error[error > 0]

        return np.sum(weighted_error) / self._Ytest.size


class Logistic(Classifier):
    """docstring for Logistic"""

    def __init__(self, X, Y):
        super().__init__(X, Y)
        self._w = np.random.randn(X.shape[1])

    def predict(self, X, w=None):
        if w is None:
            w = self._w
        z = np.dot(X, w)
        return 1 / (1 + np.exp(-z))

    def loss(self, w, indexes=None):
        if indexes is None:
            indexes = np.arange(0, self.get_number_samples(), 1)
        z = np.dot(self._Xtr[indexes, :], w) * self._Ytr[indexes]
        error = np.log(1 + np.exp(-z))
        return np.sum(error) / indexes.size

    def gradient(self, w, indexes=None):
        if indexes is None:
            indexes = np.arange(0, self.get_number_samples(), 1)
        z = np.dot(self._Xtr[indexes, :], w) * self._Ytr[indexes]
        alpha = (np.exp(-z) / (1 + np.exp(-z)) * self._Ytr[indexes])
        gradient = -(alpha[:, np.newaxis] * self._Xtr[indexes, :])
        return np.sum(gradient, axis=0)

    def test_loss(self, w):
        z = np.dot(self._Xtest, w) * self._Ytest
        error = np.log(1 + np.exp(-z))
        return np.sum(error) / self._Ytest.size


class kNN(Classifier):
    def __init__(self, X, Y, k=1):
        super().__init__(X, Y)
        self._w = k

    def set_k(self, k):
        self._w = k

    def get_k(self):
        return k

    def predict(self, X):
        Y = np.zeros((X.shape[0]))
        i = 0
        for x in X:
            D = dist(self._Xtr, x)
            indexes = np.argsort(D, axis=0)[0:self._w]

            Y[i] = np.sum(self._Ytr[indexes])
            i += 1

        return np.sign(Y)
