import chainer
import chainer.links as L
import chainer.functions as F

from chainerrl.functions.lower_triangular_matrix import lower_triangular_matrix


class NafSharedQFunction(chainer.Chain):
    def __init__(self, state_dim, action_num, *, hidden_size=200, use_batch_norm=True):
        super(NafSharedQFunction, self).__init__()
        L_lower_size = action_num * (action_num + 1) // 2
        L_diag = action_num
        L_rest = L_lower_size - action_num
        self._use_batch_norm = use_batch_norm
        with self.init_scope():
            self._shared_bn0 = L.BatchNormalization(size=[state_dim], axis=0)
            self._linear_shared1 = L.Linear(
                in_size=state_dim, out_size=hidden_size, nobias=use_batch_norm)
            self._shared_bn1 = L.BatchNormalization(size=[hidden_size], axis=0)
            self._linear_shared2 = L.Linear(
                in_size=hidden_size, out_size=hidden_size, nobias=use_batch_norm)
            self._shared_bn2 = L.BatchNormalization(size=[hidden_size], axis=0)

            self._linear_mu = L.Linear(
                in_size=hidden_size, out_size=action_num)
            self._linear_L_diag = L.Linear(
                in_size=hidden_size, out_size=L_diag)
            self._linear_L_rest = L.Linear(
                in_size=hidden_size, out_size=L_rest)
            self._linear_v = L.Linear(
                in_size=hidden_size, out_size=1)

    def __call__(self, s, a):
        x = self._encode(s)
        return self._a(x, a) + self._v(x)

    def advantage(self, s, a):
        x = self._encode(s)
        return self._a(x, a)

    def value(self, s):
        x = self._encode(s)
        return self._v(x)

    def pi(self, s):
        x = self._encode(s)
        return self._mu(x)

    def _a(self, x, a):
        L_matrix = self._L_matrix(x)
        mu = self._mu(x)

        P_matrix = F.matmul(L_matrix, L_matrix, transb=True)
        a_minus_mu = (a - mu)[:, :, None]
        return -0.5 * F.matmul(a_minus_mu, F.matmul(P_matrix, a_minus_mu), transa=True)[:, 0]

    def _v(self, x):
        return self._linear_v(x)

    def _L_matrix(self, x):
        diag = F.exp(self._linear_L_diag(x))
        rest = self._linear_L_rest(x)
        return lower_triangular_matrix(diag, rest)

    def _mu(self, x):
        return self._linear_mu(x)

    def _encode(self, s):
        if self._use_batch_norm:
            h = self._shared_bn0(s)
        else:
            h = s
        h = self._linear_shared1(h)
        if self._use_batch_norm:
            h = self._shared_bn1(h)
        else:
            pass
        h = F.relu(h)
        h = self._linear_shared2(h)
        if self._use_batch_norm:
            h = self._shared_bn2(h)
        else:
            pass
        return F.relu(h)


if __name__ == "__main__":
    import numpy as np

    batch_size = 10
    state_num = 10
    action_num = 5
    q_function = NafSharedQFunction(state_num, action_num)

    state = np.ones(shape=(batch_size, state_num), dtype=np.float32)
    action = np.ones(shape=(batch_size, action_num))
    q_value = q_function(state, action)
    print('q shape: ', q_value.shape)
    print('q: ', q_value)
    assert q_value.shape == (batch_size, 1)
