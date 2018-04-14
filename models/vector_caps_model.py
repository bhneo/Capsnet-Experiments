from models.base_model import BaseModel
from capslayer.ops import epsilon

import tensorflow as tf
import capslayer


class CapsNet(BaseModel):

    def __init__(self,
                 m_plus=0.9,
                 m_minus=0.1,
                 lambda_val=0.5,
                 reconstruction=False,
                 regularization_scale=0.392,
                 mask_with_y=False):

        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val
        self.reconstruction = reconstruction
        self.regularization_scale = regularization_scale
        self.mask_with_y = mask_with_y
        BaseModel.__init__(self)

    def __call__(self, inputs, num_label=10, is_training=True, distort=False, standardization=False):
        BaseModel.__call__(self, inputs, num_label, is_training, distort, standardization)

    def build_arch(self, images):
        outputs = {}
        with tf.variable_scope('Convolution_layer'):
            # Conv1, return with shape [batch_size, 20, 20, 256]
            conv = BaseModel.create_conv_layers(images, filters=256, kernel=9)

        # return primaryCaps: [batch_size, 1152, 8], activation: [batch_size, 1152]
        with tf.variable_scope('PrimaryCaps_layer'):
            primary_caps = capslayer.layers.vector_primary_caps(conv, filters=32, kernel_size=9, strides=2, cap_size=8)

        # return digitCaps: [batch_size, num_label, 16, 1], activation: [batch_size, num_label]
        with tf.variable_scope('DigitCaps_layer'):
            digit_caps = capslayer.layers.vector_fully_connected(primary_caps, 10, 16)
            # now [128,10,16]
            # calc || v_c ||
            activation = tf.sqrt(tf.reduce_sum(tf.square(digit_caps), axis=-1) + epsilon)
            outputs['activations'] = activation
            # activation [128,10]
            self.summary('histogram', 'activation', activation)

        if self.reconstruction:
            # 1. Do masking, how:
            with tf.variable_scope('Masking'):
                # Method 1.
                if not self.mask_with_y:
                    # pick out the index of max val of the 10 caps
                    # [batch_size, 10] => [batch_size] (index)
                    argmax_idx = tf.to_int32(tf.argmax(activation, axis=1))
                    argmax_idx = tf.reshape(argmax_idx, shape=(-1, 1))
                    pre_label = tf.one_hot(argmax_idx, self.labels.get_shape().as_list()[1])
                    masked_v = tf.multiply(digit_caps, tf.reshape(pre_label, (-1, 10, 1)))
                # Method 2. masking with true label, default mode
                else:
                    masked_v = tf.multiply(digit_caps, tf.reshape(self.labels, (-1, 10, 1)))

            # Decoder structure in Fig. 2
            # Reconstructe the MNIST images with 3 FC layers
            # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
            with tf.variable_scope('Decoder'):
                # [128,10,16]->[128,160]
                active_caps = tf.layers.flatten(masked_v)
                # active_caps = tf.reshape(masked_v, shape=(self.batch_size, -1))
                fc1 = tf.layers.dense(active_caps, units=512)
                # assert fc1.get_shape() == [self.batch_size, 512]
                fc2 = tf.layers.dense(fc1, 1024)
                # assert fc2.get_shape() == [self.batch_size, 1024]
                decoded = tf.layers.dense(fc2, units=self.height * self.width * self.channels,
                                          activation=tf.sigmoid)
                outputs['decoded'] = decoded
                recon_img = tf.reshape(decoded, shape=(-1, self.height, self.width, self.channels))
                self.summary('image', 'reconstruction_img', recon_img)

        return outputs

    def loss(self, inputs, outputs):
        images, one_hot_labels = inputs
        activation = outputs['activations']
        decoded = outputs['decoded']
        # 1. Margin loss

        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., self.m_plus - activation))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., activation - self.m_minus))

        # reshape: [batch_size, num_label, 1, 1] => [batch_size, num_label]
        # max_l = tf.reshape(max_l, shape=(self.batch_size, -1))
        # max_r = tf.reshape(max_r, shape=(self.batch_size, -1))
        max_l = tf.layers.flatten(max_l)
        max_r = tf.layers.flatten(max_r)

        # calc T_c: [batch_size, num_label]
        # T_c = Y, is my understanding correct? Try it.
        t_c = one_hot_labels
        # [batch_size, num_label], element-wise multiply
        l_c = t_c * max_l + self.lambda_val * (1 - t_c) * max_r

        margin_loss = tf.reduce_mean(tf.reduce_sum(l_c, axis=1))
        self.summary('scalar', 'margin_loss', margin_loss)

        # 2. The reconstruction loss
        if self.reconstruction:
            # origin = tf.reshape(images, shape=(self.batch_size, -1))
            origin = tf.layers.flatten(images)
            squared = tf.square(decoded - origin)
            reconstruction_err = tf.reduce_mean(squared)
            self.summary('scalar', 'reconstruction_loss', reconstruction_err)

            # 3. Total loss
            # The paper uses sum of squared error as reconstruction error, but we
            # have used reduce_mean in `# 2 The reconstruction loss` to calculate
            # mean squared error. In order to keep in line with the paper,the
            # regularization scale should be 0.0005*784=0.392
            total_loss = margin_loss + self.regularization_scale * reconstruction_err
            self.summary('scalar', 'total_loss', total_loss)
            return total_loss
        else:
            return margin_loss


