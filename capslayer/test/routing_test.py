import tensorflow as tf
import numpy as np
import ops





class RoutingTest(tf.test.TestCase):

    def testLeakyRoutingRankThree(self):
        """Checks the shape of the leaky routing output given a rank 3 input.

        When using leaky routing the some of routing logits should be always less
        than 1 because some portion of the probability is leaked.
        """
        logits = tf.ones((2, 3, 4))
        leaky = ops._leaky_routing(logits, 4)
        self.assertEqual(len(leaky.get_shape()), 3)
        with self.test_session() as sess:
            r_leaky = sess.run(leaky)
        self.assertEqual(np.array(r_leaky).shape, logits.get_shape())
        self.assertTrue(np.less(np.sum(r_leaky, axis=2), 1.0).all(keepdims=False))

    def testUpdateRouting(self):
        """Tests the correct shape of the output of update_routing function.

        Checks that routing iterations change the activation value.
        """
        votes = np.reshape(np.arange(8, dtype=np.float32), (1, 2, 2, 2))
        biases = tf.zeros((2, 2))
        logit_shape = (1, 2, 2)
        activations_1 = ops._update_routing(
            votes,
            biases,
            logit_shape,
            num_dims=4,
            input_dim=2,
            num_routing=1,
            output_dim=2,
            leaky=False)
        activations_2 = ops._update_routing(
            votes,
            biases,
            logit_shape,
            num_dims=4,
            input_dim=2,
            num_routing=1,
            output_dim=2,
            leaky=False)
        activations_3 = ops._update_routing(
            votes,
            biases,
            logit_shape,
            num_dims=4,
            input_dim=2,
            num_routing=30,
            output_dim=2,
            leaky=False)
        self.assertEqual(len(activations_3.get_shape()), 3)
        self.assertEqual(len(activations_1.get_shape()), 3)
        with self.test_session() as sess:
            act_1, act_2, act_3 = sess.run(
                [activations_1, activations_2, activations_3])
        self.assertNotAlmostEquals(np.sum(act_1), np.sum(act_3))
        self.assertAlmostEquals(np.sum(act_1), np.sum(act_2))


if __name__ == '__main__':
    tf.test.main()