import numpy as np
import tensorflow as tf

epsilon = 1e-9

def squash(input_tensor, axis=-2):
    '''Squashing function
    Args:
        input_tensor: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1]
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    squared_norm = tf.reduce_sum(tf.square(input_tensor), axis, keepdims=True)
    scalar_factor = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + epsilon)
    return(scalar_factor * input_tensor)

def _squash(input_tensor, axis=2):
     """Applies norm nonlinearity (squash) to a capsule layer.

     Args:
       input_tensor: Input tensor. Shape is [batch, num_channels, num_atoms] for a
         fully connected capsule layer or
         [batch, num_channels, num_atoms, height, width] for a convolutional
         capsule layer.

     Returns:
       A tensor with same shape as input (rank 3) for output of this layer.
     """
     norm = tf.norm(input_tensor, axis=axis, keepdims=True)
     norm_squared = norm * norm
     return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

def _leaky_routing(logits, output_dim):
  """Adds extra dimmension to routing logits.

  This enables active capsules to be routed to the extra dim if they are not a
  good fit for any of the capsules in layer above.

  Args:
    logits: The original logits. shape is
      [input_capsule_num, output_capsule_num] if fully connected. Otherwise, it
      has two more dimmensions.
    output_dim: The number of units in the second dimmension of logits.

  Returns:
    Routing probabilities for each pair of capsules. Same shape as logits.
  """

  # leak is a zero matrix with same shape as logits except dim(2) = 1 because
  # of the reduce_sum.
  leak = tf.zeros_like(logits, optimize=True)
  leak = tf.reduce_sum(leak, axis=2, keepdims=True)
  leaky_logits = tf.concat([leak, logits], axis=2)
  leaky_routing = tf.nn.softmax(leaky_logits, dim=2)
  return tf.split(leaky_routing, [1, output_dim], 2)[1]


def _update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                    num_routing, leaky):
  """Sums over scaled votes and applies squash to compute the activations.

  Iteratively updates routing logits (scales) based on the similarity between
  the activation of this layer and the votes of the layer below.

  Args:
    votes: tensor, The transformed outputs of the layer below.
    biases: tensor, Bias variable.
    logit_shape: tensor, shape of the logit to be initialized.
    num_dims: scalar, number of dimmensions in votes. For fully connected
      capsule it is 4, for convolutional 6.
    input_dim: scalar, number of capsules in the input layer.
    output_dim: scalar, number of capsules in the output layer.
    num_routing: scalar, Number of routing iterations.
    leaky: boolean, if set use leaky routing.

  Returns:
    The activation tensor of the output layer after num_routing iterations.
  """
  votes_t_shape = [3, 0, 1, 2]
  for i in range(num_dims - 4):
    votes_t_shape += [i + 4]
  r_t_shape = [1, 2, 3, 0]
  for i in range(num_dims - 4):
    r_t_shape += [i + 4]
  votes_trans = tf.transpose(votes, votes_t_shape)

  def _body(i, logits, activations):
    """Routing while loop."""
    # route: [batch, input_dim, output_dim, ...]
    # corresponding to line 4 of routing algorithm
    if leaky:
      route = _leaky_routing(logits, output_dim)  # heikeji
    else:
      route = tf.nn.softmax(logits, dim=2)

    # line 5
    # again, the paper does not explicit use the bias term(Eq.2
    # s_j=sum_i(c_ij*u_hat_ij)), but here we see that.
    preactivate_unrolled = route * votes_trans
    preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
    preactivate = tf.reduce_sum(preact_trans, axis=1) + biases

    # line 6
    activation = _squash(preactivate)
    activations = activations.write(i, activation)

    # line 7
    # distances: [batch, input_dim, output_dim]
    act_3d = tf.expand_dims(activation, 1)
    tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
    tile_shape[1] = input_dim
    act_replicated = tf.tile(act_3d, tile_shape)
    # distances = u_hat * v
    distances = tf.reduce_sum(votes * act_replicated, axis=3)
    logits += distances
    return (i + 1, logits, activations)

  activations = tf.TensorArray(
      dtype=tf.float32, size=num_routing, clear_after_read=False)
  # logits corresponds to b_ij
  logits = tf.fill(logit_shape, 0.0)
  i = tf.constant(0, dtype=tf.int32)
  _, logits, activations = tf.while_loop(
      lambda i, logits, activations: i < num_routing,
      _body,
      loop_vars=[i, logits, activations],
      swap_memory=True)

  return activations.read(num_routing - 1)

def routing(u_hat, b_ij, nb_capsules, nb_capsules_p, iterations=4):
    """
        Routing algorithm

        **input:
            *u_hat: Dot product (weights between previous capsule and current capsule)
            *b_ij: the log prior probabilities that capsule i should be coupled to capsule j
            *nb_capsules_p: Number of capsule in the previous layer
            *nb_capsules: Number of capsule in this layer
    """
    # Start the routing algorithm
    for it in range(iterations):
        with tf.variable_scope('routing_' + str(it)):
            # Line 4 of algo
            # probabilities that capsule i should be coupled to capsule j.
            # c_ij:  [nb_capsules_p, nb_capsules, 1, 1]
            c_ij = tf.nn.softmax(b_ij, dim=2)

            # Line 5 of algo
            # c_ij:  [      nb_capsules_p, nb_capsules, 1,         1]
            # u_hat: [?,    nb_capsules_p, nb_capsules, len_v_j,   1]
            s_j = tf.multiply(c_ij, u_hat)
            # s_j: [?, nb_capsules_p, nb_capsules, len_v_j, 1]
            s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)
            # s_j: [?, 1, nb_capsules, len_v_j, 1)

            # line 6:
            # squash using Eq.1,
            v_j = squash(s_j)
            # v_j: [1, 1, nb_capsules, len_v_j, 1)

            # line 7:
            # Frist reshape & tile v_j
            # [? ,  1,              nb_capsules,    len_v_j, 1] ->
            # [?,   nb_capsules_p,  nb_capsules,    len_v_j, 1]
            v_j_tiled = tf.tile(v_j, [1, nb_capsules_p, 1, 1, 1])
            # u_hat:    [?,             nb_capsules_p, nb_capsules, len_v_j, 1]
            # v_j_tiled [1,             nb_capsules_p, nb_capsules, len_v_j, 1]
            u_dot_v = tf.matmul(u_hat, v_j_tiled, transpose_a=True)
            # u_produce_v: [?, nb_capsules_p, nb_capsules, 1, 1]
            b_ij += tf.reduce_sum(u_dot_v, axis=0, keep_dims=True)
            #b_ih: [1, nb_capsules_p, nb_capsules, 1, 1]

    return tf.squeeze(v_j, axis=1)


def routing(vote,
            activation=None,
            num_outputs=32,
            out_caps_shape=[4, 4],
            method='EMRouting',
            num_iter=3,
            regularizer=None):
    ''' Routing-by-agreement algorithm.
    Args:
        alias H = out_caps_shape[0]*out_caps_shape[1].

        vote: [batch_size, num_inputs, num_outputs, H].
        activation: [batch_size, num_inputs, 1, 1].
        num_outputs: ...
        out_caps_shape: ...
        method: method for updating coupling coefficients between vote and pose['EMRouting', 'DynamicRouting'].
        num_iter: the number of routing iteration.
        regularizer: A (Tensor -> Tensor or None) function; the result of applying it on a newly created variable
                will be added to the collection tf.GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.

    Returns:
        pose: [batch_size, 1, 1, num_outputs] + out_caps_shape.
        activation: [batch_size, 1, 1, num_outputs].
    '''
    vote_stopped = tf.stop_gradient(vote, name="stop_gradient")
    batch_size = vote.shape[0].value
    if method == 'EMRouting':
        shape = vote.get_shape().as_list()[:3] + [1]
        # R: [batch_size, num_inputs, num_outputs, 1]
        R = tf.constant(np.ones(shape, dtype=np.float32) / num_outputs)
        for t_iter in range(num_iter):
            with tf.variable_scope('M-STEP') as scope:
                if t_iter > 0:
                    scope.reuse_variables()
                # It's no need to do the `E-STEP` in the last iteration
                if t_iter == num_iter - 1:
                    pose, stddev, activation_prime = M_step(R, activation, vote)
                    break
                else:
                    pose, stddev, activation_prime = M_step(R, activation, vote_stopped)
            with tf.variable_scope('E-STEP'):
                R = E_step(pose, stddev, activation_prime, vote_stopped)
        pose = tf.reshape(pose, shape=[batch_size, 1, 1, num_outputs] + out_caps_shape)
        activation = tf.reshape(activation_prime, shape=[batch_size, 1, 1, -1])
        return(pose, activation)
    elif method == 'DynamicRouting':
        B = tf.constant(np.zeros([batch_size, vote.shape[1].value, num_outputs, 1, 1], dtype=np.float32))
        for r_iter in range(num_iter):
            with tf.variable_scope('iter_' + str(r_iter)):
                coef = tf.softmax(B, axis=2)
                if r_iter == num_iter - 1:
                    s = tf.reduce_sum(tf.multiply(coef, vote), axis=1, keepdims=True)
                    pose = squash(s)
                else:
                    s = tf.reduce_sum(tf.multiply(coef, vote_stopped), axis=1, keepdims=True)
                    pose = squash(s)
                    shape = [batch_size, vote.shape[1].value, num_outputs] + out_caps_shape
                    pose = tf.multiply(pose, tf.constant(1., shape=shape))
                    B += tf.matmul(vote_stopped, pose, transpose_a=True)
        return(pose, activation)

    else:
        raise Exception('Invalid routing method!', method)


def M_step(R, activation, vote, lambda_val=0.9, regularizer=None):
    '''
    Args:
        alias H = out_caps_shape[0]*out_caps_shape[1]

        vote: [batch_size, num_inputs, num_outputs, H]
        activation: [batch_size, num_inputs, 1, 1]
        R: [batch_size, num_inputs, num_outputs, 1]
        lambda_val: ...

    Returns:
        pose & stddev: [batch_size, 1, num_outputs, H]
        activation: [batch_size, 1, num_outputs, 1]
    '''
    batch_size = vote.shape[0].value
    # line 2
    R = tf.multiply(R, activation)
    R_sum_i = tf.reduce_sum(R, axis=1, keepdims=True) + epsilon

    # line 3
    # mean: [batch_size, 1, num_outputs, H]
    pose = tf.reduce_sum(R * vote, axis=1, keepdims=True) / R_sum_i

    # line 4
    stddev = tf.sqrt(tf.reduce_sum(R * tf.square(vote - pose), axis=1, keepdims=True) / R_sum_i + epsilon)

    # line 5, cost: [batch_size, 1, num_outputs, H]
    H = vote.shape[-1].value
    beta_v = tf.get_variable('beta_v', shape=[batch_size, 1, pose.shape[2].value, H], regularizer=regularizer)
    cost = (beta_v + tf.log(stddev)) * R_sum_i

    # line 6
    beta_a = tf.get_variable('beta_a', shape=[batch_size, 1, pose.shape[2], 1], regularizer=regularizer)
    activation = tf.nn.sigmoid(lambda_val * (beta_a - tf.reduce_sum(cost, axis=3, keepdims=True)))

    return(pose, stddev, activation)


def E_step(pose, stddev, activation, vote):
    '''
    Args:
        alias H = out_caps_shape[0]*out_caps_shape[1]

        pose & stddev: [batch_size, 1, num_outputs, H]
        activation: [batch_size, 1, num_outputs, 1]
        vote: [batch_size, num_inputs, num_outputs, H]

    Returns:
        pose & var: [batch_size, 1, num_outputs, H]
        activation: [batch_size, 1, num_outputs, 1]
    '''
    # line 2
    var = tf.square(stddev)
    x = tf.reduce_sum(tf.square(vote - pose) / (2 * var), axis=-1, keepdims=True)
    peak_height = 1 / (tf.reduce_prod(tf.sqrt(2 * np.pi * var + epsilon), axis=-1, keepdims=True) + epsilon)
    P = peak_height * tf.exp(-x)

    # line 3
    R = tf.nn.softmax(activation * P, axis=2)
    return(R)
