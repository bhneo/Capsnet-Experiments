"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import tensorflow as tf
from models.base_model import BaseModel
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
                 regularization_scale=0.392):
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
        BaseModel.__init__(height, width, channels, num_label, is_training)

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, return with shape [batch_size, 20, 20, 256]
            conv1 = tf.layers.conv2d(self.images, num_outputs=256, kernel_size=9, stride=1, padding='VALID')

        # return primaryCaps: [batch_size, 1152, 8, 1], activation: [batch_size, 1152]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps, activation = capslayer.layers.primaryCaps(conv1, filters=32, kernel_size=9, strides=2, out_caps_shape=[8, 1])

        # return digitCaps: [batch_size, num_label, 16, 1], activation: [batch_size, num_label]
        with tf.variable_scope('DigitCaps_layer'):
            primaryCaps = tf.reshape(primaryCaps, shape=[self.batch_size, -1, 8, 1])
            self.digitCaps, self.activation = capslayer.layers.fully_connected(primaryCaps, activation, num_outputs=10, out_caps_shape=[16, 1], routing_method='DynamicRouting')

        if self.reconstruction:
            # Decoder structure in Fig. 2
            # Reconstructe the MNIST images with 3 FC layers
            # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
            with tf.variable_scope('Decoder'):
                masked_caps = tf.multiply(self.digitCaps, tf.reshape(self.Y, (-1, self.num_label, 1, 1)))
                active_caps = tf.reshape(masked_caps, shape=(self.batch_size, -1))
                fc1 = tf.contrib.layers.fully_connected(active_caps, num_outputs=512)
                fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
                self.decoded = tf.layers.fully_connected(fc2, num_outputs=self.height * self.width * self.channels,
                                                         activation_fn=tf.sigmoid)

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
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.loss))
        recon_img = tf.reshape(self.decoded, shape=(self.batch_size, self.height, self.width, self.channels))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        train_summary.append(tf.summary.histogram('activation', self.activation))
        self.train_summary = tf.summary.merge(train_summary)
