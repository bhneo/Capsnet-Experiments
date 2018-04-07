from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from capslayer import layers


class LayersTest(tf.test.TestCase):

    def testCapsule(self):
        """Tests the correct output and variable declaration of layers.capsule."""
        input_tensor = tf.random_uniform((4, 3, 2))
        output = layers.capsule(
            input_tensor=input_tensor,
            input_dim=3,
            output_dim=2,
            layer_name='capsule',
            input_atoms=2,
            output_atoms=5,
            num_routing=3,
            leaky=False)
        self.assertListEqual(output.get_shape().as_list(), [4, 2, 5])
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.assertEqual(len(trainable_vars), 2)
        self.assertStartsWith(trainable_vars[0].name, 'capsule')

    def testConvSlimCapsule(self):
        """Tests the creation of layers.conv_slim_capsule.

        Tests the correct number of variables are declared and shape of the output
        is as a 5D list with the correct numbers.
        """
        # input_tensor = tf.random_uniform((6, 4, 2, 3, 3))
        input_tensor = tf.random_uniform((128, 1, 256, 20, 20))
        output = layers.conv_slim_capsule(
            input_tensor=input_tensor,
            input_dim=1,
            output_dim=32,
            layer_name='conv_capsule',
            input_atoms=256,
            output_atoms=8,
            stride=2,
            kernel_size=9,
            padding='SAME',
            num_routing=3,
            leaky=False)
        self.assertListEqual(output.get_shape().as_list(), [None, 2, 5, 3, 3])
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.assertEqual(len(trainable_vars), 2)
        self.assertStartsWith(trainable_vars[0].name, 'conv_capsule')


if __name__ == '__main__':
    tf.test.main()