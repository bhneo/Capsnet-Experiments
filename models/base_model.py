"""
License: Apache-2.0
Author: Zhao Lei
E-mail: bhneo@126.com
"""
import abc
import tensorflow as tf


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.graph = tf.Graph()

    def __call__(self, inputs, num_label=10, is_training=True, distort=False, standardization=False):
        """

        :param inputs: input data, formatted as (X, y)
        :param num_label:
        :param is_training:
        :return:
        """
        self.num_label = num_label
        self.summary_arr = []
        with self.graph.as_default():
            self.images, self.labels = inputs
            image_shape = self.images.get_shape().as_list()
            self.height, self.width, self.channels = image_shape[1], image_shape[2], image_shape[3]

            # 用于评估过程的图像数据预处理
            if distort:
                # 为图像添加padding = 4，图像尺寸变为[32+4,32+4],为后面的随机裁切留出位置
                padded_image = tf.image.resize_image_with_crop_or_pad(self.images, self.width + 4, self.height + 4)
                # 下面的这些操作为原始图像添加了很多不同的distortions，扩增了原始训练数据集
                self.images = self.distort_resize(padded_image, self.height, self.width, self.channels)

            if standardization:
                # 数据集标准化操作：减去均值 + 方差标准化
                self.images = tf.image.per_image_standardization(self.images)

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
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self.summary('scalar', 'accuracy', self.accuracy)

            self.merged_summary = tf.summary.merge(self.summary_arr)

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

    @staticmethod
    def distort_resize(images, height, width, channel, max_delta=63, contrast_lower=0.2, contrast_upper=1.8):
        """
        Distorts input images for CIFAR training. Adds standard distortions such as flipping, cropping and changing brightness
        and contrast.
        :param images: A float32 tensor.
        :param height:
        :param width:
        :param channel:
        :param max_delta:
        :param contrast_lower:
        :param contrast_upper:
        :return: distorted_image: A float32 tensor with shape [image_size, image_size, channel].
        """
        distorted_image = tf.random_crop(images, [height, width, channel])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=max_delta)
        distorted_image = tf.image.random_contrast(
            distorted_image, lower=contrast_lower, upper=contrast_upper)
        distorted_image.set_shape([height, width, channel])
        return distorted_image
