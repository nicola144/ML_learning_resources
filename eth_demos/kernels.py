import numpy as np

class Kernel(object):
    def __init__(self, Xtr, Ytr, reg=0.0, deg=0, bw=None, prediction=True):
        self._Xtr = Xtr
        self._Ytr = Ytr
        self._lambda = reg
        self._deg = deg
        self._bw = bw
        self._Ktr = self._lambda * np.eye(self._Xtr.shape[0])

        self._alpha = None
        self._Kpr = None
        self._prediction = prediction

    def set_regularization(self, reg):
        self._lambda = reg

    def set_weights(self, alpha):
        self._alpha = alpha

    def get_number_samples(self):
        return self._Ytr.size

    def load_data(self, Xtr, Ytr=None):
        self._Xtr = Xtr
        self._Ytr = Ytr
        self._Ktr = self._lambda * np.eye(self._Xtr.shape[0])

    def calculate_alpha(self, Ytr):
        self._alpha = np.dot(np.linalg.pinv(self._Ktr), Ytr)  # compute regression coefficients

    def _build_prediction_kernel(self, X):
        self._Kpr = np.zeros((self._Xtr.shape[0], X.shape[0]))

    def predict(self, X):
        self._build_prediction_kernel(X)

    def _predict(self):
        if self._prediction:
            return np.dot(self._alpha.T, self._Kpr).T
        else:
            return np.sign(np.dot(self._alpha.T, self._Kpr).T)

    def loss(self, alpha, indexes):
        if self._prediction:
            return self.prediction_loss(alpha, indexes)
        else:
            return self.classification_loss(alpha, indexes)

    def gradient(self, alpha, indexes):
        if self._prediction:
            return self.prediction_gradient(alpha, indexes)
        else:
            return self.classification_gradient(alpha, indexes)

    def prediction_loss(self, alpha, indexes):
        self.set_weights(np.reshape(alpha, [-1, 1]))
        Yhat = np.dot(self._alpha.T, self._Ktr[:, indexes]).T
        error = np.reshape(Yhat, [-1, 1]) - self._Ytr[indexes]
        return np.sum(np.square(error)) + self._lambda * np.inner(Yhat, self._alpha)

    def prediction_gradient(self, alpha, indexes):
        self.set_weights(np.reshape(alpha, [-1, 1]))
        Yhat = np.dot(self._alpha.T, self._Ktr[:, indexes]).T
        error = np.reshape(Yhat, [-1, 1]) - self._Ytr[indexes]

        grad = np.zeros((alpha.size, 1))
        grad[indexes] = np.dot(self._Ktr[indexes][:, indexes], (error + self._lambda * self._alpha[indexes]))
        return np.reshape(grad, newshape=alpha.shape) / indexes.size

    def classification_loss(self, alpha, indexes):
        self.set_weights(alpha)
        Yhat = np.sign(np.dot(self._alpha.T, self._Ktr[:, indexes]).T)
        error = -Yhat * self._Ytr[indexes]
        error[error < 0] = 0
        return np.sum(error)

    def classification_gradient(self, alpha, indexes):
        self.set_weights(alpha)
        gradient = np.zeros_like(alpha)

        Yhat = np.sign(np.dot(self._alpha.T, self._Ktr[:, indexes]).T)
        wrong_indexes = (indexes[Yhat != self._Ytr[indexes]])
        gradient[wrong_indexes] = -self._Ytr[wrong_indexes]
        return gradient

    def build_kernel(self, X):
        pass


class SumKernel(Kernel):
    def __init__(self, kernel_list, Xtr, Ytr, reg=0.0, deg=0, bw=None, prediction=True):
        super().__init__(Xtr=Xtr, Ytr=Ytr, reg=reg, deg=0, bw=0, prediction=prediction)
        self._kernel_list = []
        for idx, kernel_name in enumerate(kernel_list):
            try:
                local_deg = deg[idx]
            except TypeError:
                local_deg = deg
            try:
                local_bw = bw[idx]
            except TypeError:
                local_bw = bw
            
            kernel = kernel_name(Xtr=Xtr, Ytr=Ytr, reg=0, deg=local_deg, bw=local_bw, prediction=prediction)

            self._kernel_list.append(kernel)
            self._Ktr += kernel.build_kernel(self._Xtr)

    def predict(self, Y):
        super().predict(Y)
        for kernel in self._kernel_list:
            self._Kpr += kernel.build_kernel(Y)
        return self._predict()


class LinearKernel(Kernel):
    def __init__(self, Xtr, Ytr=None, reg=0.0, deg=1, bw=None, prediction=True):
        super().__init__(Xtr=Xtr, Ytr=Ytr, reg=reg, deg=deg, bw=bw, prediction=prediction)
        self._Ktr += self.build_kernel(self._Xtr)

    def load_data(self, Xtr, Ytr=None):
        super().load_data(Xtr, Ytr)
        self._Ktr += self.build_kernel(self._Xtr)

    def predict(self, Y):
        super().predict(Y)
        self._Kpr += self.build_kernel(Y)  # compute kernel between train and test points
        return self._predict()

    def build_kernel(self, Y):
        rows = self._Xtr.shape[0]
        cols = Y.shape[0]
        K = np.zeros((rows, cols))
        for col in range(cols):
            K[:, col] = (np.dot(self._Xtr, Y[col, :].T))
        return K


class PolynomialKernel(Kernel):
    def __init__(self, Xtr, Ytr=None, reg=0.0, deg=1, bw=None, prediction=True):
        super().__init__(Xtr=Xtr, Ytr=Ytr, reg=reg, deg=deg, bw=bw, prediction=prediction)
        self._Ktr += self.build_kernel(self._Xtr)

    def load_data(self, Xtr, Ytr=None):
        super().load_data(Xtr, Ytr)
        self._Ktr += self.build_kernel(self._Xtr)

    def predict(self, Y):
        super().predict(Y)
        self._Kpr += self.build_kernel(Y)
        return self._predict()

    def build_kernel(self, Y):
        rows = self._Xtr.shape[0]
        cols = Y.shape[0]
        K = np.zeros((rows, cols))
        for col in range(cols):
            K[:, col] = np.power(1+np.dot(self._Xtr, Y[col, :].T), self._deg)
        return K 


class LaplacianKernel(Kernel):
    def __init__(self, Xtr, Ytr=None, reg=0.0, deg=0, bw=0.2, prediction=True):
        super().__init__(Xtr=Xtr, Ytr=Ytr, reg=reg, deg=deg, bw=bw, prediction=prediction)
        self._Ktr += self.build_kernel(self._Xtr)

    def load_data(self, Xtr, Ytr=None):
        super().load_data(Xtr, Ytr)
        self._Ktr += self.build_kernel(self._Xtr)

    def predict(self, Y):
        super().predict(Y)
        self._Kpr += self.build_kernel(Y)
        return self._predict()

    def build_kernel(self, Y):
        rows = self._Xtr.shape[0]
        cols = Y.shape[0]
        K = np.zeros((rows, cols))
        for col in range(cols):
            dist = np.linalg.norm(self._Xtr - Y[col, :], ord=1, axis=1) / self._bw 
            K[:, col] = np.exp(-dist)
        return K


class GaussianKernel(Kernel):
    def __init__(self, Xtr, Ytr=None, reg=0.0, deg=0, bw=0.2, prediction=True):
        super().__init__(Xtr=Xtr, Ytr=Ytr, reg=reg, deg=deg, bw=bw, prediction=prediction)
        self._Ktr += self.build_kernel(self._Xtr)

    def load_data(self, Xtr, Ytr=None):
        super().load_data(Xtr, Ytr)
        self._Ktr += self.build_kernel(self._Xtr)

    def predict(self, Y):
        super().predict(Y)
        self._Kpr += self.build_kernel(Y)
        return self._predict()

    def build_kernel(self, Y):
        rows = self._Xtr.shape[0]
        cols = Y.shape[0]
        K = np.zeros((rows, cols))
        for col in range(cols):
            dist = np.square(np.linalg.norm(self._Xtr - Y[col, :], ord=2, axis=1) / self._bw)
            K[:, col] = np.exp(-dist)
        return K


class PeriodicKernel(Kernel):
    def __init__(self, Xtr, Ytr=None, reg=0.0, deg=0, bw=2.5, prediction=True):
        super().__init__(Xtr=Xtr, Ytr=Ytr, reg=reg, deg=deg, bw=bw, prediction=prediction)
        self._Ktr += self.build_kernel(self._Xtr)

    def load_data(self, Xtr, Ytr=None):
        super().load_data(Xtr, Ytr)
        self._Ktr += self.build_kernel(self._Xtr) # compute kernel between train points

    def predict(self, Y):
        super().predict(Y)
        self._Kpr += self.build_kernel(Y) # compute kernel between train and test points
        return self._predict()

    def build_kernel(self, Y):
        rows = self._Xtr.shape[0]
        cols = Y.shape[0]
        K = np.zeros((rows, cols))
        for col in range(cols):
            dist = (np.linalg.norm(self._Xtr - Y[col, :], ord=2, axis=1)) * self._bw
            K[:, col] = np.exp(-np.square(np.sin(dist)))
        return K

