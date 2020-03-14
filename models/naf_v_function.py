import chainer
import chainer.links as L
import chainer.functions as F


class NafVFunction(chainer.Chain):
    def __init__(self, state_dim, *, hidden_size=200):
        super(NafVFunction, self).__init__()
        with self.init_scope():
            self._linear1 = L.Linear(
                in_size=state_dim, out_size=hidden_size, nobias=True)
            self._bn1 = L.BatchNormalization(axis=0)
            self._linear2 = L.Linear(
                in_size=hidden_size, out_size=hidden_size, nobias=True)
            self._bn2 = L.BatchNormalization(axis=0)
            self._linear3 = L.Linear(in_size=hidden_size, out_size=1)

    def __call__(self, s):
        h = self._linear1(s)
        h = self._bn1(h)
        h = F.relu(h)
        h = self._linear2(h)
        h = self._bn2(h)
        h = F.relu(h)
        return self._linear3(h)


if __name__ == "__main__":
    import numpy as np

    batch_size = 10
    state_num = 10
    value_function = NafVFunction(state_num)

    state = np.ones(shape=(batch_size, state_num), dtype=np.float32)
    value = value_function(state)
    print('value shape: ', value.shape)
    print('value: ', value)
    assert value.shape == (batch_size, 1)
