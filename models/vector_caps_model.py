"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""
from models.base_model import BaseModel
from capslayer.ops import epsilon

import tensorflow as tf
import capslayer


class CapsNet(BaseModel):

    def __init__(self,
                 height=28,
                 width=28,
                 channels=1,
                 num_label=10,
                 is_training=True,
                 m_plus=0.9,
                 m_minus=0.1,
                 lambda_val=0.5,
                 reconstruction=False,
                 regularization_scale=0.392,
                 mask_with_y=False):
        """
        Args:
            height: Integer, the height of input.
            width: Integer, the width of input.
            channels: Integer, the channels of input.
            num_label: Integer, the category number.
        """
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val
        self.reconstruction = reconstruction
        self.regularization_scale = regularization_scale
        self.mask_with_y = mask_with_y
        BaseModel.__init__(self, height, width, channels, num_label, is_training)

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, return with shape [batch_size, 20, 20, 256]
            conv1 = tf.layers.conv2d(self.images, filters=256, kernel_size=9, strides=1, padding='VALID',
                                     activation=tf.nn.relu)

        # return primaryCaps: [batch_size, 1152, 8, 1], activation: [batch_size, 1152]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = capslayer.layers.vector_primary_caps(conv1, filters=32, kernel_size=9, strides=2, cap_size=8,
                                                               activation=tf.nn.relu)

        # return digitCaps: [batch_size, num_label, 16, 1], activation: [batch_size, num_label]
        with tf.variable_scope('DigitCaps_layer'):
            self.digitCaps = capslayer.layers.fully_connected_v1(primaryCaps, 10)
            # calc | | v_c | |
            self.activation = tf.sqrt(tf.reduce_sum(tf.square(self.digitCaps),
                                                    axis=2, keepdims=True) + epsilon)

        if self.reconstruction:
            # 1. Do masking, how:
            with tf.variable_scope('Masking'):
                # a). do softmax(||v_c||)
                # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
                self.softmax_v = tf.nn.softmax(self.activation, axis=1)
                assert self.softmax_v.get_shape() == [self.batch_size, 10, 1, 1]

                # b). pick out the index of max softmax val of the 10 caps
                # [batch_size, 10, 1, 1] => [batch_size] (index)
                self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
                assert self.argmax_idx.get_shape() == [self.batch_size, 1, 1]
                self.argmax_idx = tf.reshape(self.argmax_idx, shape=(self.batch_size,))

                # Method 1.
                if not self.mask_with_y:
                    # c). indexing
                    # It's not easy to understand the indexing process with argmax_idx
                    # as we are 3-dim animal
                    masked_v = []
                    for batch_size in range(self.batch_size):
                        v = self.digitCaps[batch_size][self.argmax_idx[batch_size], :]
                        masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                    self.masked_v = tf.concat(masked_v, axis=0)
                    assert self.masked_v.get_shape() == [self.batch_size, 1, 16, 1]
                # Method 2. masking with true label, default mode
                else:
                    # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
                    self.masked_v = tf.multiply(tf.squeeze(self.digitCaps), tf.reshape(self.labels, (-1, 10, 1)))

            # Decoder structure in Fig. 2
            # Reconstructe the MNIST images with 3 FC layers
            # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
            with tf.variable_scope('Decoder'):
                active_caps = tf.reshape(self.masked_v, shape=(self.batch_size, -1))
                fc1 = tf.layers.dense(active_caps, units=512)
                assert fc1.get_shape() == [self.batch_size, 512]
                fc2 = tf.layers.dense(fc1, 1024)
                assert fc2.get_shape() == [self.batch_size, 1024]
                self.decoded = tf.layers.dense(fc2, units=self.height * self.width * self.channels,
                                               activation=tf.sigmoid)

        return self.activation

    def loss(self):
        # 1. Margin loss

        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., self.m_plus - self.activation))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.activation - self.m_minus))

        # reshape: [batch_size, num_label, 1, 1] => [batch_size, num_label]
        max_l = tf.reshape(max_l, shape=(self.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(self.batch_size, -1))

        # calc T_c: [batch_size, num_label]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.one_hot_labels
        # [batch_size, num_label], element-wise multiply
        L_c = T_c * max_l + self.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        if self.reconstruction:
            orgin = tf.reshape(self.images, shape=(self.batch_size, -1))
            squared = tf.square(self.decoded - orgin)
            self.reconstruction_err = tf.reduce_mean(squared)

            # 3. Total loss
            # The paper uses sum of squared error as reconstruction error, but we
            # have used reduce_mean in `# 2 The reconstruction loss` to calculate
            # mean squared error. In order to keep in line with the paper,the
            # regularization scale should be 0.0005*784=0.392
            return self.margin_loss + self.regularization_scale * self.reconstruction_err
        else:
            return self.margin_loss

    # Summary
    def _summary(self):
        recon_img = tf.reshape(self.decoded, shape=(self.batch_size, self.height, self.width, self.channels))
        train_summary = [tf.summary.scalar('margin_loss', self.margin_loss),
                         tf.summary.histogram('activation', self.activation),
                         tf.summary.histogram('accuracy', self.accuracy)]
        if self.reconstruction:
            train_summary.append(tf.summary.scalar('reconstruction_loss', self.reconstruction_err))
            train_summary.append(tf.summary.scalar('total_loss', self.loss))
            train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)
