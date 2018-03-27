import tensorflow as tf

epsilon = 1e-9


def spread_loss(labels, logits, margin, regularizer=None):
    '''
    Args:
        labels: [batch_size, num_label, 1].
        logits: [batch_size, num_label, 1].
        margin: Integer or 1-D Tensor.
        regularizer: use regularization.

    Returns:
        loss: Spread loss.
    '''
    # a_target: [batch_size, 1, 1]
    a_target = tf.matmul(labels, logits, transpose_a=True)
    dist = tf.maximum(0., margin - (a_target - logits))
    loss = tf.reduce_mean(tf.square(tf.matmul(1 - labels, dist, transpose_a=True)))
    if regularizer is not None:
        regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += tf.reduce_mean(regularizer)
    return (loss)


def margin_loss(label, v, m_plus=0.9, m_minus=0.1, lambda_val=0.5):
    batch_size = v.get_shape().as_list()[0]
    # 1. The margin loss
    v_length = tf.sqrt(tf.reduce_sum(tf.square(v), axis=2, keepdim=True) + epsilon)
    # [batch_size, 10, 1, 1]
    # max_l = max(0, m_plus-||v_c||)^2
    max_l = tf.square(tf.maximum(0., m_plus - v_length))
    # max_r = max(0, ||v_c||-m_minus)^2
    max_r = tf.square(tf.maximum(0., v_length - m_minus))
    assert max_l.get_shape() == [batch_size, 10, 1, 1]

    # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
    max_l = tf.reshape(max_l, shape=(batch_size, -1))
    max_r = tf.reshape(max_r, shape=(batch_size, -1))

    # calc T_c: [batch_size, 10]
    # T_c = Y, is my understanding correct? Try it.
    T_c = label
    # [batch_size, 10], element-wise multiply
    L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r

    loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
    return loss


def _margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
    """Penalizes deviations from margin for each logit.

  Each wrong logit costs its distance to margin. For negative logits margin is
  0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
  margin is 0.4 from each side.

  Args:
    labels: tensor, one hot encoding of ground truth.
    raw_logits: tensor, model predictions in range [0, 1]
    margin: scalar, the margin after subtracting 0.5 from raw_logits.
    downweight: scalar, the factor for negative cost.

  Returns:
    A tensor with cost for each data point of shape [batch_size].
  """
    logits = raw_logits - 0.5
    positive_cost = labels * tf.cast(tf.less(logits, margin),
                                     tf.float32) * tf.pow(logits - margin, 2)
    negative_cost = (1 - labels) * tf.cast(
        tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
    return 0.5 * positive_cost + downweight * 0.5 * negative_cost


def cross_entropy(labels, logits, regularizer=None):
    '''
    Args:
        ...

    Returns:
        ...
    '''
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    if regularizer is not None:
        regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += tf.reduce_mean(regularizer)
    return (loss)
