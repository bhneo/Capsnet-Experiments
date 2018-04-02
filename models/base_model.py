"""
License: Apache-2.0
Author: Zhao Lei
E-mail: bhneo@126.com
"""
import abc
import tensorflow as tf


class BaseModel(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, height=28, width=28, channels=1, num_label=10, is_training=True):
        '''
        Args:
            height: Integer, the height of input.
            width: Integer, the width of input.
            channels: Integer, the channels of input.
            num_label: Integer, the category number.
            is_training: ...
        '''
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels), name='images')
            self.labels = tf.placeholder(tf.int32, shape=(None,), name='labels')
            self.one_hot_labels = tf.one_hot(self.labels, depth=self.num_label, axis=1, dtype=tf.float32)
            self.batch_size = self.images.get_shape().as_list()[0]
            logits = self.build_arch()

            if is_training:
                loss = self.loss()
                self._summary()

                self.global_step = tf.Variable(1, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(loss, global_step=self.global_step)

            with tf.variable_scope('accuracy'):
                logits_idx = tf.to_int32(tf.argmax(tf.softmax(logits, axis=1), axis=1))
                correct_prediction = tf.equal(tf.to_int32(self.labels), logits_idx)
                self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    @abc.abstractclassmethod
    def build_arch(self):
        raise NotImplementedError('Not implemented')

    @abc.abstractclassmethod
    def loss(self):
        raise NotImplementedError('Not implemented')

    @abc.abstractclassmethod
    def _summary(self):
        raise NotImplementedError('Not implemented')
