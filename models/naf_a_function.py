import chainer
import chainer.links as L
import chainer.functions as F

from chainerrl.functions.lower_triangular_matrix import lower_triangular_matrix


class NafLMatrix(chainer.Chain):
    def __init__(self, state_dim, action_num, hidden_size):
        super(NafLMatrix, self).__init__()
        L_lower_size = action_num * (action_num + 1) // 2
        L_diag = action_num
        L_rest = L_lower_size - action_num
        with self.init_scope():
            self._linear_L1 = L.Linear(
                in_size=state_dim, out_size=hidden_size, nobias=True)
            self._L_bn1 = L.BatchNormalization(axis=0)
            self._linear_L2 = L.Linear(
                in_size=hidden_size, out_size=hidden_size, nobias=True)
            self._L_bn2 = L.BatchNormalization(axis=0)
            self._linear_L3_diag = L.Linear(
                in_size=hidden_size, out_size=L_diag)
            self._linear_L3_rest = L.Linear(
                in_size=hidden_size, out_size=L_rest)

    def __call__(self, s):
        h = self._linear_L1(s)
        h = self._L_bn1(h)
        h = F.relu(h)
        h = self._linear_L2(h)
        h = self._L_bn2(h)
        h = F.relu(h)
        diag = F.exp(self._linear_L3_diag(h))
        rest = self._linear_L3_rest(h)
        return lower_triangular_matrix(diag, rest)


class NafMu(chainer.Chain):
    def __init__(self, state_dim, action_num, hidden_size):
        super(NafMu, self).__init__()
        with self.init_scope():
            self._linear_mu1 = L.Linear(
                in_size=state_dim, out_size=hidden_size, nobias=True)
            self._mu_bn1 = L.BatchNormalization(axis=0)
            self._linear_mu2 = L.Linear(
                in_size=hidden_size, out_size=hidden_size, nobias=True)
            self._mu_bn2 = L.BatchNormalization(axis=0)
            self._linear_mu3 = L.Linear(
                in_size=hidden_size, out_size=action_num)

    def __call__(self, s):
        h = self._linear_mu1(s)
        h = self._mu_bn1(h)
        h = F.relu(h)
        h = self._linear_mu2(h)
        h = self._mu_bn2(h)
        h = F.relu(h)
        return self._linear_mu3(h)


class NafAFunction(chainer.Chain):
    def __init__(self, state_dim, action_num, *, hidden_size=200):
        super(NafAFunction, self).__init__()
        with self.init_scope():
            self._L = NafLMatrix(state_dim=state_dim,
                                 action_num=action_num,
                                 hidden_size=hidden_size)
            self._mu = NafMu(state_dim=state_dim,
                             action_num=action_num,
                             hidden_size=hidden_size)

    def __call__(self, s, a):
        L_matrix = self._L(s)
        mu = self._mu(s)

        P_matrix = F.matmul(L_matrix, L_matrix, transb=True)
        a_minus_mu = (a - mu)[:, :, None]
        return -0.5 * F.matmul(a_minus_mu, F.matmul(P_matrix, a_minus_mu), transa=True)[:, 0]


if __name__ == "__main__":
    import numpy as np

    def verify_batch_norm_behavior():
        state1 = np.diag(v=[1, 1])
        state1 = np.reshape(a=state1, newshape=(1, 2, 2))
        state2 = np.full((9, 2, 2), 5, dtype=np.float32)
        print('state1 shape: ', state1.shape)
        print('state2 shape: ', state2.shape)
        state = np.concatenate([state1, state2], axis=0)

        mu = np.mean(state, axis=0)
        print('mu: ', mu)

        var = np.var(state, axis=0)
        print('var: ', var)

        expected = (state - mu) / np.sqrt(var)
        print('expected: ', expected)

        test_L_bn1 = L.BatchNormalization(axis=0)
        normalized = test_L_bn1(state)
        print('normalized: ', normalized)
    # verify_batch_norm_behavior()

    def assert_is_lower_triangular_matrix(batch):
        for matrix in batch:
            matrix = matrix.array
            diag = np.diag(matrix)
            assert np.all(diag != 0)

            (rows, columns) = matrix.shape
            for row in range(rows):
                for column in range(columns):
                    if column <= row:
                        assert matrix[row][column] != 0
                    else:
                        assert matrix[row][column] == 0

    batch_size = 10
    state_num = 10
    action_num = 5
    advantage_function = NafAFunction(state_num, action_num)

    state = np.ones(shape=(batch_size, state_num), dtype=np.float32)
    L_matrix = advantage_function._L(state)

    assert_is_lower_triangular_matrix(L_matrix)
    assert L_matrix.shape == (batch_size, action_num, action_num)

    action = np.ones(shape=(batch_size, action_num))
    advantage = advantage_function(state, action)
    print('advantage shape: ', advantage.shape)
    print('advantage: ', advantage)
    assert advantage.shape == (batch_size, 1)
