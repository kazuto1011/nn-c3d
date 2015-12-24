import chainer
import chainer.functions as F
import chainer.links as L

class C3DNet(chainer.Chain):
    def __init__(self):
        super(C3DNet, self).__init__(
            fc6=L.Linear(4096, 4096), # fc6 - fc7
            fc7=L.Linear(4096, 4096), # fc7 - fc8
            fc8=L.Linear(4096, 5)     # fc8 - prob
        )

    def __call__(self, x):
        h = F.dropout(F.relu(x))
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        return self.fc8(h)
