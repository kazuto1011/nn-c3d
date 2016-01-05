import sys, cv2, six, time
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

import c3d_data

print 'Preparing dataset...'
dataset = c3d_data.load_feature()
dataset = c3d_data.shuffle_video(dataset)

dataset['data'] = dataset['data'].astype(np.float32)
dataset['target'] = dataset['target'].astype(np.float32)

N_TRAIN = 6000
x_train, x_test = np.split(dataset['data'], [N_TRAIN])
y_train, y_test = np.split(dataset['target'], [N_TRAIN])

def inference(feature, keep_prob):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    with tf.name_scope('fc6') as scope:
        h_fc6 = tf.nn.dropout(tf.nn.relu(feature), keep_prob)

    with tf.name_scope('fc7') as scope:
        W_fc7 = weight_variable([4096,4096])
        b_fc7 = bias_variable([4096])
        h_fc7 = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7), keep_prob)

    with tf.name_scope('softmax') as scope:
        W_softmax = weight_variable([4096,5])
        b_softmax = bias_variable([5])
        prob = tf.nn.softmax(tf.matmul(h_fc7, W_softmax) + b_softmax)

    return prob

def loss(logits, labels):
    cross_entoropy = -tf.reduce_sum(labels*tf.clip_by_value(logits,1e-10,1.0))
    tf.scalar_summary('cross_entoropy', cross_entoropy) # for TensorBoard
    return cross_entoropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary('train_accuracy', accuracy) # for TensorBoard
    return accuracy

start_time = time.clock()
with tf.Graph().as_default():
    x         = tf.placeholder("float", shape=[None, 4096])
    y         = tf.placeholder("float", shape=[None, 5])
    keep_prob = tf.placeholder("float")

    logits     = inference(x, keep_prob)
    loss_value = loss(logits, y)
    train_op   = training(loss_value, learning_rate=1e-4)
    accur      = accuracy(logits, y)

    init_op = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init_op)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('./tensorboard', sess.graph_def)

    # Training params
    batchsize = 50
    n_epoch = 20

    for epoch in six.moves.range(1, n_epoch+1):
        print "epoch %d" % (epoch)
        perm = np.random.permutation(N_TRAIN)
        for step in six.moves.range(0, N_TRAIN, batchsize):
            x_batch = x_train[perm[step:step + batchsize]]
            y_batch = y_train[perm[step:step + batchsize]]
            sess.run(train_op, feed_dict={x:x_batch, y:y_batch, keep_prob:0.5})
            if step % 1000 == 0:
                train_accur = sess.run(accur, feed_dict={x:x_batch, y:y_batch, keep_prob:1.0})
                print "step %d, training accuracy %g" % (step,train_accur)
                summary_str = sess.run(summary_op, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
                summary_writer.add_summary(summary_str, epoch*N_TRAIN + step)

        print "test accuracy %g"%sess.run(accur, feed_dict={x:x_test, y:y_test, keep_prob:1.0})

print "time %lf"%(time.clock() - start_time)
