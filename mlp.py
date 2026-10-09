import os
import pickle
import numpy as np

EPS = 1e-15


def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
def relu(z): return np.maximum(0, z)
def tanh(z): return np.tanh(z)

ACTIVATIONS = {
    'sigmoid': (sigmoid, lambda a: a * (1 - a)),
    'relu': (relu, lambda a: (a > 0).astype(float)),
    'tanh': (tanh, lambda a: 1 - a ** 2),
}


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def binary_cross_entropy(Y, P):
    P = np.clip(P, EPS, 1 - EPS)
    return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))


def init_weights(n_in, n_out, how, rng):
    if how == 'heUniform':
        lim = np.sqrt(6.0 / n_in)
        return rng.uniform(-lim, lim, (n_in, n_out))
    if how == 'heNormal':
        return rng.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
    if how == 'xavier':
        lim = np.sqrt(6.0 / (n_in + n_out))
        return rng.uniform(-lim, lim, (n_in, n_out))
    raise ValueError(f'неизвестная инициализация: {how}')


class DenseLayer:

    def __init__(self, n_in, n_out, activation='relu', initializer='heUniform', rng=None):
        rng = rng if rng is not None else np.random.RandomState(0)
        self.activation = activation
        self.W = init_weights(n_in, n_out, initializer, rng)
        self.b = np.zeros((1, n_out))
        self.vW, self.vb = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        z = x @ self.W + self.b
        self.a = softmax(z) if self.activation == 'softmax' else ACTIVATIONS[self.activation][0](z)
        return self.a

    def backward_z(self, dz):
        self.gW = self.x.T @ dz
        self.gb = dz.sum(axis=0, keepdims=True)
        return dz @ self.W.T

    def backward(self, da):
        return self.backward_z(da * ACTIVATIONS[self.activation][1](self.a))

    def update(self, lr, momentum=0.0):
        self.vW = momentum * self.vW - lr * self.gW
        self.vb = momentum * self.vb - lr * self.gb
        self.W += self.vW
        self.b += self.vb


class MLP:
    def __init__(self, sizes, activation='relu', initializer='heUniform', seed=0):
        """sizes = [n_in, h1, h2, ..., n_out]. Скрытые слои - activation, выход - softmax."""
        self.sizes, self.activation, self.initializer = list(sizes), activation, initializer
        rng = np.random.RandomState(seed)
        self.layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            act = 'softmax' if i == len(sizes) - 2 else activation
            self.layers.append(DenseLayer(a, b, act, initializer, rng))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    predict_proba = forward

    def backward(self, Y):
        grad = (self.layers[-1].a - Y) / Y.shape[0]
        grad = self.layers[-1].backward_z(grad)
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)

    def train_step(self, X, Y, lr, momentum=0.0):
        self.forward(X)
        self.backward(Y)
        for layer in self.layers:
            layer.update(lr, momentum)

    def get_weights(self):
        return [(l.W.copy(), l.b.copy()) for l in self.layers]

    def set_weights(self, weights):
        for l, (W, b) in zip(self.layers, weights):
            l.W, l.b = W.copy(), b.copy()

    def save(self, path, mean, std):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'sizes': self.sizes, 'activation': self.activation, 'initializer': self.initializer,
                         'weights': self.get_weights(), 'mean': mean, 'std': std}, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        net = MLP(d['sizes'], d['activation'], d['initializer'])
        net.set_weights(d['weights'])
        return net, d['mean'], d['std']