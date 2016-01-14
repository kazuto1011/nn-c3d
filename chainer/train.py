# coding: utf-8

import argparse
import numpy as np
import six, time
import matplotlib.pyplot as plt

import chainer
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.loss import mean_squared_error
from chainer import serializers

import c3d_data as data
import c3d_net as net

parser = argparse.ArgumentParser(description='3-layer NNs to train C3D fatures')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# Prepare dataset
print 'Loading dataset...'
c3d_data = data.load_feature()
c3d_data = data.shuffle_video(c3d_data)

c3d_data['data'] = c3d_data['data'].astype(np.float32)
c3d_data['target'] = c3d_data['target'].astype(np.int32)

N = 6000
x_train, x_test = np.split(c3d_data['data'], [N])
y_train, y_test = np.split(c3d_data['target'], [N])
N_test = y_test.size

# 3-layer nets
model = L.Classifier(net.C3DNet())
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

# Setup model
optimizer = optimizers.Adam(alpha=1e-4)
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)

# Training params
batchsize = 50
n_epoch = 20

# Graph
train_acc = []
train_loss = []
test_acc = []
test_loss = []

start_time = time.clock()
for epoch in six.moves.range(1, n_epoch + 1):
    print "epoch: %d" % epoch

    # Train
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        # Shuffle
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

        # forward-propagation and optimization
        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    train_acc.append(sum_accuracy/N)
    train_loss.append(sum_loss/N)
    print('train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N))

    # Test
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]), volatile='on')
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]), volatile='on')

        loss = model(x, t)

        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    test_acc.append(sum_accuracy/N_test)
    test_loss.append(sum_loss/N_test)
    print('test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test))

print time.clock() - start_time

print('save the model')
serializers.save_hdf5('c3d_3layer.model', model)
print('save the optimizer')
serializers.save_hdf5('c3d_3layer.state', optimizer)

# Accuracy
plt.figure(figsize=(8,6))
plt.plot(range(len(train_acc)), train_acc)
plt.plot(range(len(test_acc)), test_acc)
plt.legend(["train","test"],loc=4)
plt.title("Accuracy")
plt.plot()

# Error
plt.figure(figsize=(8,6))
plt.plot(range(len(train_loss)), train_loss)
plt.plot(range(len(test_loss)), test_loss)
plt.legend(["train","test"],loc=4)
plt.title("Error")
plt.plot()

plt.show()
