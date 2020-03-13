import chainer
import chainer.links as L
import chainer.functions as F


class NafQFunction(chainer.Chain):
    def __init__(self, state_dim, action_num):
        super(NafQFunction, self).__init__()
        pass
