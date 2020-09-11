import numpy as np


class Optimizers:

    @staticmethod
    def get(name, b, W, learning_rate, **kwargs):
        return getattr(Optimizers(), name)(b, W, learning_rate, **kwargs)

    def sgd(self, b, W, learning_rate):

        def update(b, W, grad_bias, grad_embeddings, i):
            return b - learning_rate * grad_bias, W - learning_rate * grad_embeddings

        return update

    def adagrad(self, b, W, learning_rate, momentum=0.9):

        print("momentum: %s" % momentum)

        grad_bias_sum = np.zeros_like(b)
        grad_embeddings_sum = np.zeros_like(W)
        m_bias = np.zeros_like(b)
        m_embeddings = np.zeros_like(W)

        def update(b, W, grad_bias, grad_embeddings, i):
            nonlocal grad_bias_sum, grad_embeddings_sum, m_bias, m_embeddings

            grad_bias_sum += np.square(grad_bias)
            grad_embeddings_sum += np.square(grad_embeddings)
            ada_bias = np.where(grad_bias_sum > 0, 1. / np.sqrt(grad_bias_sum), 0.0)
            ada_embeddings = np.where(grad_embeddings_sum > 0, 1. / np.sqrt(grad_embeddings_sum), 0.0)
            m_bias = (1. - momentum) * (grad_bias * ada_bias) + momentum * m_bias
            m_embeddings = (1. - momentum) * (grad_embeddings * ada_embeddings) + momentum * m_embeddings

            return b - learning_rate * m_bias, W - learning_rate * m_embeddings

        return update

    def momentum(self, b, W, learning_rate, mass=0.5):

        print("mass: %s" % mass)

        v_b = np.zeros_like(b)
        v_W = np.zeros_like(W)

        def update(b, W, grad_bias, grad_embeddings, i):
            nonlocal v_b, v_W

            v_b = mass * v_b + grad_bias
            v_W = mass * v_W + grad_embeddings
            return b - learning_rate * v_b, W - learning_rate * v_W

        return update

    def nesterov(self, b, W, learning_rate, mass=0.5):

        print("mass: %s" % mass)

        v_b = np.zeros_like(b)
        v_W = np.zeros_like(W)

        def update(b, W, grad_bias, grad_embeddings, i):
            nonlocal v_b, v_W

            v_b = mass * v_b + grad_bias
            b_updated = b - learning_rate * (mass * v_b + grad_bias)

            v_W = mass * v_W + grad_embeddings
            W_updated = W - learning_rate * (mass * v_W + grad_embeddings)

            return b_updated, W_updated

        return update

    def adam(self, b, W, learning_rate, b1=0.9, b2=0.999, eps=1e-8):

        print("b1=%s, b2=%s, eps=%s" % (b1, b2, eps))

        m_b = np.zeros_like(b)
        v_b = np.zeros_like(b)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)

        def update(b, W, grad_bias, grad_embeddings, i):
            nonlocal m_b, v_b, m_W, v_W

            m_b = (1 - b1) * grad_bias + b1 * m_b
            v_b = (1 - b2) * np.square(grad_bias) + b2 * v_b
            m_b_hat = m_b / (1 - b1 ** (i + 1))
            v_b_hat = v_b / (1 - b2 ** (i + 1))
            b_updated = b - learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

            m_W = (1 - b1) * grad_embeddings + b1 * m_W
            v_W = (1 - b2) * np.square(grad_embeddings) + b2 * v_W
            m_W_hat = m_W / (1 - b1 ** (i + 1))
            v_W_hat = v_W / (1 - b2 ** (i + 1))
            W_updated = W - learning_rate * m_W_hat / (np.sqrt(v_W_hat) + eps)

            return b_updated, W_updated

        return update

    def rmsprop(self, b, W, learning_rate, gamma=0.9, eps=1e-8):

        print("gamma: %s, eps: %s" % (gamma, eps))

        avg_sq_grad_b = np.zeros_like(b)
        avg_sq_grad_W = np.zeros_like(W)

        def update(b, W, grad_bias, grad_embeddings, i):
            nonlocal avg_sq_grad_b, avg_sq_grad_W

            avg_sq_grad_b = avg_sq_grad_b * gamma + np.square(grad_bias) * (1. - gamma)
            b_updated = b - learning_rate * grad_bias / np.sqrt(avg_sq_grad_b + eps)

            avg_sq_grad_W = avg_sq_grad_W * gamma + np.square(grad_embeddings) * (1. - gamma)
            W_updated = W - learning_rate * grad_embeddings / np.sqrt(avg_sq_grad_W + eps)

            return b_updated, W_updated

        return update
