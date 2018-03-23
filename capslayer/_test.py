import tensorflow as tf
import numpy as np
import time

from capslayer import layers
from capslayer import losses
from capslayer import ops


class CapsLayerTest(tf.test.TestCase):

    def test1(self):
        b_IJ = tf.constant(np.zeros([10, 10, 1, 1], dtype=np.float32))

        b_ij = tf.zeros(shape=[10, 10, 1, 1], dtype=np.float32)
        tf.Constant
        b_IJ += 1
        b_ij += 2

        with self.test_session() as sess:
            IJ,ij = sess.run([b_IJ, b_ij])

        print(IJ,ij)


if __name__ == '__main__':
    tf.test.main()
