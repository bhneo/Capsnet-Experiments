import tensorflow as tf
import numpy as np
import time

from capslayer import layers
from capslayer import losses
from capslayer import ops


class SquashTest(tf.test.TestCase):

    def testSquash(self):
        """Checks the value and shape of the squash output given an input."""
        input_tensor = tf.ones((1, 1, 1, 1, 1, 1))
        squashed = ops.squash(input_tensor)
        self.assertEqual(len(squashed.get_shape()), 6)
        with self.test_session() as sess:
            r_squashed = sess.run(squashed)
        scale = 0.5
        self.assertEqual(np.array(r_squashed).shape, input_tensor.get_shape())
        self.assertAllClose(np.linalg.norm(r_squashed, axis=2), [[[[[scale]]]]])

    def testSquash_(self):
        """Checks the value and shape of the squash output given an input."""
        input_tensor = tf.ones((1, 1, 1, 1, 1, 1))
        squashed = ops._squash(input_tensor)
        self.assertEqual(len(squashed.get_shape()), 6)
        with self.test_session() as sess:
            r_squashed = sess.run(squashed)
        scale = 0.5
        self.assertEqual(np.array(r_squashed).shape, input_tensor.get_shape())
        self.assertAllClose(np.linalg.norm(r_squashed, axis=2), [[[[[scale]]]]])

    # def testProcessSquashDetail(self):
    #     vector = tf.ones((1, 1, 1, 1, 20, 1))
    #     squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    #     scalar_factor = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + 0.000001)
    #     result = scalar_factor * vector
    #     with self.test_session() as sess:
    #         r_squared_norm, r_scalar_factor, r_result = sess.run([squared_norm, scalar_factor, result])
    #     print('r_squared_norm',r_squared_norm)
    #     print('r_scalar_factor', r_scalar_factor)
    #     print('r_result', r_result)
    #     print('result shape', np.shape(r_result))
    #
    # def testProcessSquashDetail_(self):
    #     vector = tf.ones((1, 1, 20, 1, 1, 1))
    #     norm = tf.norm(vector, 2, keepdims=True)
    #     norm_squared = norm * norm
    #     result = (vector / norm) * (norm_squared / (1 + norm_squared))
    #     with self.test_session() as sess:
    #         r_norm, r_norm_squared, r_result = sess.run([norm, norm_squared, result])
    #     print('r_norm',r_norm)
    #     print('r_norm_squared', r_norm_squared)
    #     print('r_result', r_result)
    #     print('result shape', np.shape(r_result))

    def testSquashTime(self):
        input_tensor = tf.ones((128, 1, 1000, 1, 20, 1))
        squashed = ops.squash(input_tensor,axis=-2)
        with self.test_session() as sess:
            time1 = time.time()
            for i in range(100):
                r_squashed = sess.run(squashed)
            time2 = time.time()
        print('Squash time:',time2-time1)

    def testSquashTime_(self):
        input_tensor = tf.ones((128, 1, 1000, 1, 20, 1))
        squashed = ops._squash(input_tensor, axis=-2)
        with self.test_session() as sess:
            time1 = time.time()
            for i in range(100):
                r_squashed = sess.run(squashed)
            time2 = time.time()
        print('_Squash time:',time2-time1)


if __name__ == '__main__':
    tf.test.main()
