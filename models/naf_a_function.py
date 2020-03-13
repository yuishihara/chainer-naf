import chainer
import chainer.links as L
import chainer.functions as F


class NafAFunction(chainer.Chain):
    def __init__(self, state_dim, action_num, *, hidden_size=200):
        super(NafAFunction, self).__init__()
        L_matrix_size = action_num * (action_num + 1) / 2
        L_diag = action_num
        L_rest = L_matrix_size - action_num
        self._linear_L1 = L.Linear(in_size=state_dim, out_size=hidden_size)
        self._L_bn1 = L.BatchNormalization()
        self._linear_L2 = L.Linear(in_size=hidden_size, out_size=hidden_size)
        self._L_bn2 = L.BatchNormalization()
        self._linear_L3_diag = L.Linear(in_size=hidden_size, out_size=L_diag)
        self._linear_L3_rest = L.Linear(in_size=hidden_size, out_size=L_rest)

        self._linear_mu1 = L.Linear(in_size=state_dim, out_size=hidden_size)
        self._mu_bn1 = L.BatchNormalization()
        self._linear_mu2 = L.Linear(in_size=hidden_size, out_size=hidden_size)
        self._mu_bn2 = L.BatchNormalization()
        self._linear_mu3 = L.Linear(in_size=hidden_size, out_size=action_num)

    def __call__(self, s, a):
        pass
