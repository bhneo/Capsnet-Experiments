import tensorflow as tf
import numpy as np
from capslayer import layers


class LossTest(tf.test.TestCase):

    def testMarginLoss(self):
        """Checks the correct margin loss output for a simple scenario.

        In the first example it should only penalize the second logit.
        In the second example it should penalize second and third logit.
        """
        labels = [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        logits = [[-0.3, 0.3, 0.9], [1.2, 0.5, -0.5]]
        costs = [[0, 0.5 * 0.2 * 0.2, 0], [0, 0.4 * 0.4, 1.4 * 1.4]]
        sum_costs = np.sum(costs)
        margin_output = layers._margin_loss(
            labels=tf.constant(labels), raw_logits=tf.constant(logits))
        with self.test_session() as sess:
            output = sess.run(margin_output)
        self.assertAlmostEqual(0.5 * sum_costs, np.sum(output), places=6)


if __name__ == '__main__':
    tf.test.main()