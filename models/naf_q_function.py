import chainer
import chainer.links as L
import chainer.functions as F

from .naf_a_function import NafAFunction
from .naf_v_function import NafVFunction


class NafQFunction(chainer.Chain):
    def __init__(self, state_dim, action_num):
        super(NafQFunction, self).__init__()
        with self.init_scope():
            self._a = NafAFunction(
                state_dim=state_dim, action_num=action_num)
            self._v = NafVFunction(state_dim=state_dim)

    def __call__(self, s, a):
        return self.advantage(s, a) + self.value(s)

    def advantage(self, s, a):
        return self._a(s, a)

    def value(self, s):
        return self._v(s)

    def pi(self, s):
        return self._a._mu(s)

    def training_params(self):
        return (self._v, self._a._L, self._a._mu)


if __name__ == "__main__":
    import numpy as np

    batch_size = 10
    state_num = 10
    action_num = 5
    q_function = NafQFunction(state_num, action_num)

    state = np.ones(shape=(batch_size, state_num), dtype=np.float32)
    action = np.ones(shape=(batch_size, action_num))
    q_value = q_function(state, action)
    print('q shape: ', q_value.shape)
    print('q: ', q_value)
    assert q_value.shape == (batch_size, 1)

    assert len(q_function.training_params()) == 3
