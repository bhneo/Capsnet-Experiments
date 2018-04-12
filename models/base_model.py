"""
License: Apache-2.0
Author: Zhao Lei
E-mail: bhneo@126.com
"""
import abc
import tensorflow as tf


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, inputs, num_label=10, is_training=True):
        """ init model

        :param inputs: input data, formatted as (X, y)
        :param num_label:
        :param is_training:
        """
        self.height, self.width, self.channels = inputs.get_shape().as_list()[1, 2, 3]
        self.num_label = num_label
        self.summary_arr = []

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images, self.labels = inputs
            self.one_hot_labels = tf.one_hot(self.labels, depth=self.num_label, axis=1, dtype=tf.float32)
            outputs = self.build_arch(self.images)

            if is_training:
                loss = self.loss((self.images, self.one_hot_labels), outputs)

                self.global_step = tf.Variable(1, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(loss, global_step=self.global_step)

            with tf.variable_scope('accuracy'):
                activations_idx = tf.to_int32(tf.argmax(tf.nn.softmax(outputs['activations'], axis=1), axis=1))
                correct_prediction = tf.equal(tf.to_int32(self.labels), activations_idx)
                self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                self.summary('histogram', 'accuracy', self.accuracy)

            self.merged_summary = tf.summary.merge(self.summary)

    @abc.abstractclassmethod
    def build_arch(self, image):
        raise NotImplementedError('Not implemented')

    @abc.abstractclassmethod
    def loss(self, inputs, outputs):
        raise NotImplementedError('Not implemented')

    def summary(self, sum_type, name, obj):
        if sum_type == 'scalar':
            self.summary_arr.append(tf.summary.scalar(name, obj))
        elif sum_type == 'histogram':
            self.summary_arr.append(tf.summary.histogram(name, obj))
        elif sum_type == 'image':
            self.summary_arr.append(tf.summary.image(name, obj))

    @staticmethod
    def create_conv_layers(inputs, filters=256, kernel=9):
        """ create convolution layers before caps layers to extract feature

        :param inputs:
        :param filters:
        :param kernel:
        :return:
        """
        conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel, strides=1, padding='VALID',
                                activation=tf.nn.relu)
        return conv
