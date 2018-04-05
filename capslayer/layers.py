'''
This module provides a set of high-level neural networks layers.
'''

import tensorflow as tf
import numpy as np
from functools import reduce

from capslayer.utils import get_transformation_matrix_shape
from capslayer.ops import routing
from capslayer.ops import dynamic_routing
from capslayer.ops import squash
from capslayer.ops import _update_routing
from capslayer import variables


def fully_connected_v1(inputs, num_outputs):
    """

    :param inputs: [batch_size, input_num, vec_len, 1]
    :param num_outputs:
    :return:
    """
    batch_size = inputs.get_shape().as_list()[0]
    inputs = tf.reshape(inputs, shape=(batch_size, -1, 1, inputs.shape[-2].value, 1))

    with tf.variable_scope('routing'):
        b_ij = tf.constant(np.zeros([batch_size, inputs.shape[1].value, num_outputs, 1, 1], dtype=np.float32))
        capsules = dynamic_routing(inputs, b_ij)
        capsules = tf.squeeze(capsules, axis=1)
    return capsules


def fully_connected(inputs, activation,
                    num_outputs,
                    out_caps_shape,
                    routing_method='EMRouting',
                    reuse=None):
    '''A capsule fully connected layer.
    Args:
        inputs: A tensor with shape [batch_size, num_inputs] + in_caps_shape.
        activation: [batch_size, num_inputs]
        num_outputs: Integer, the number of output capsules in the layer.
        out_caps_shape: A list with two elements, pose shape of output capsules.
    Returns:
        pose: [batch_size, num_outputs] + out_caps_shape
        activation: [batch_size, num_outputs]
    '''
    in_pose_shape = inputs.get_shape().as_list()
    num_inputs = in_pose_shape[1]
    batch_size = in_pose_shape[0]
    T_size = get_transformation_matrix_shape(in_pose_shape[-2:], out_caps_shape)
    T_shape = [1, num_inputs, num_outputs] + T_size
    T_matrix = tf.get_variable("transformation_matrix", shape=T_shape)
    T_matrix = tf.tile(T_matrix, [batch_size, 1, 1, 1, 1])
    inputs = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, num_outputs, 1, 1])
    with tf.variable_scope('transformation'):
        # vote: [batch_size, num_inputs, num_outputs] + out_caps_shape
        vote = tf.matmul(T_matrix, inputs)
    with tf.variable_scope('routing'):
        if routing_method == 'EMRouting':
            activation = tf.reshape(activation, shape=activation.get_shape().as_list() + [1, 1])
            vote = tf.reshape(vote, shape=[batch_size, num_inputs, num_outputs, -1])
            pose, activation = routing(vote, activation, num_outputs, out_caps_shape, routing_method)
            pose = tf.reshape(pose, shape=[batch_size, num_outputs] + out_caps_shape)
            activation = tf.reshape(activation, shape=[batch_size, -1])
        elif routing_method == 'DynamicRouting':
            pose, _ = routing(vote, activation, num_outputs=num_outputs, out_caps_shape=out_caps_shape,
                              method=routing_method)
            pose = tf.squeeze(pose, axis=1)
            activation = tf.squeeze(euclidean_norm(pose))
        else:
            raise EnvironmentError()
    return (pose, activation)


def fully_connected_caps_layer_thibo(input_layer, capsules_size, nb_capsules, iterations=4):
    """
        Second layer receiving inputs from all capsules of the layer below
            **input:
                *input_layer: (Tensor)
                *capsules_size: (Integer) Size of each capsule
                *nb_capsules: (Integer) Number of capsule
                *iterations: (Integer) Number of iteration for the routing algorithm

            i refer to the layer below.
            j refer to the layer above (the current layer).
    """
    shape = input_layer.get_shape().as_list()
    # Get the size of each capsule in the previous layer and the current layer.
    len_u_i = np.prod(shape[2])
    len_v_j = capsules_size
    # Get the number of capsule in the layer bellow.
    nb_capsules_p = np.prod(shape[1])

    # w_ij: Used to compute u_hat by multiplying the output ui of a capsule in the layer below
    # with this matrix
    # [nb_capsules_p, nb_capsules, len_v_j, len_u_i]
    _init = tf.random_normal_initializer(stddev=0.01, seed=0)
    _shape = (nb_capsules_p, nb_capsules, len_v_j, len_u_i)
    w_ij = tf.get_variable('weight', shape=_shape, dtype=tf.float32, initializer=_init)

    # Adding one dimension to the input [batch_size, nb_capsules_p,    length(u_i), 1] ->
    #                                   [batch_size, nb_capsules_p, 1, length(u_i), 1]
    # To allow the next dot product
    input_layer = tf.reshape(input_layer, shape=(-1, nb_capsules_p, 1, len_u_i, 1))
    input_layer = tf.tile(input_layer, [1, 1, nb_capsules, 1, 1])

    # Eq.2, calc u_hat
    # Prediction uj|i made by capsule i
    # w_ij:  [              nb_capsules_p, nb_capsules, len_v_j,  len_u_i, ]
    # input: [batch_size,   nb_capsules_p, nb_capsules, len_ui,   1]
    # u_hat: [batch_size,   nb_capsules_p, nb_capsules, len_v_j, 1]
    # Each capsule of the previous layer capsule layer is associated to a capsule of this layer
    u_hat = tf.einsum('abdc,iabcf->iabdf', w_ij, input_layer)

    # bij are the log prior probabilities that capsule i should be coupled to capsule j
    # [nb_capsules_p, nb_capsules, 1, 1]
    b_ij = tf.zeros(shape=[nb_capsules_p, nb_capsules, 1, 1], dtype=np.float32)

    return routing(u_hat, b_ij, nb_capsules, nb_capsules_p, iterations=iterations)


def primaryCaps_v1(input, filters, kernel_size=9, stride=2, vec_len=8, activation=tf.nn.relu):
    """build primary caps layer according to the 1st paper

    :param input: the input tensor, shape is [batch_size, width, height, channels]
    :param kernel_size: ...
    :param stride: ...
    :return: caps: [batch_size, width, height, channels] + cap_shape
    """
    capsules = tf.layers.conv2d(input, filters * vec_len, kernel_size, stride, padding='VALID', activation=activation)
    batch_size = input.get_shape().as_list()[0]
    capsules = tf.reshape(capsules, (batch_size, -1, vec_len, 1))

    return squash(capsules)


def primaryCaps(input, filters,
                kernel_size,
                strides,
                out_caps_shape,
                method=None,
                regularizer=None):
    """PrimaryCaps layer
    Args:
        input: [batch_size, in_height, in_width, in_channels].
        filters: Integer, the dimensionality of the output space.
        kernel_size: ...
        strides: ...
        out_caps_shape: ...
        method: the method of calculating probability of entity existence(logistic, norm, None)
    Returns:
        pose: [batch_size, out_height, out_width, filters] + out_caps_shape
        activation: [batch_size, out_height, out_width, filters]
    """
    # pose matrix
    pose_size = reduce(lambda x, y: x * y, out_caps_shape)
    pose = tf.layers.conv2d(input, filters * pose_size,
                            kernel_size=kernel_size,
                            strides=strides, activation=None,
                            activity_regularizer=regularizer)
    pose_shape = pose.get_shape().as_list()[:3] + [filters] + out_caps_shape
    pose = tf.reshape(pose, shape=pose_shape)

    if method == 'logistic':
        # logistic activation unit
        activation = tf.layers.conv2d(input, filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      activation=tf.nn.sigmoid,
                                      activity_regularizer=regularizer)
    elif method == 'norm':
        activation = tf.sqrt(tf.reduce_sum(tf.square(pose), axis=2, keepdims=True) + 1e-9)
    else:
        activation = None

    return pose, activation


def conv2d(in_pose,
           activation,
           filters,
           out_caps_shape,
           kernel_size,
           strides=(1, 1),
           coordinate_addition=False,
           regularizer=None,
           reuse=None):
    '''A capsule convolutional layer.
    Args:
        in_pose: A tensor with shape [batch_size, in_height, in_width, in_channels] + in_caps_shape.
        activation: A tensor with shape [batch_size, in_height, in_width, in_channels]
        filters: ...
        out_caps_shape: ...
        kernel_size: ...
        strides: ...
        coordinate_addition: ...
        regularizer: apply regularization on a newly created variable and add the variable to the collection tf.GraphKeys.REGULARIZATION_LOSSES.
        reuse: ...
    Returns:
        out_pose: A tensor with shape [batch_size, out_height, out_height, out_channals] + out_caps_shape,
        out_activation: A tensor with shape [batch_size, out_height, out_height, out_channels]
    '''
    # do some preparation stuff
    in_pose_shape = in_pose.get_shape().as_list()
    in_caps_shape = in_pose_shape[-2:]
    batch_size = in_pose_shape[0]
    in_channels = in_pose_shape[3]

    T_size = get_transformation_matrix_shape(in_caps_shape, out_caps_shape)
    if isinstance(kernel_size, int):
        h_kernel_size = kernel_size
        w_kernel_size = kernel_size
    elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
        h_kernel_size = kernel_size[0]
        w_kernel_size = kernel_size[1]
    if isinstance(strides, int):
        h_stride = strides
        w_stride = strides
    elif isinstance(strides, (list, tuple)) and len(strides) == 2:
        h_stride = strides[0]
        w_stride = strides[1]
    num_inputs = h_kernel_size * w_kernel_size * in_channels
    batch_shape = [batch_size, h_kernel_size, w_kernel_size, in_channels]
    T_shape = (1, num_inputs, filters) + tuple(T_size)

    T_matrix = tf.get_variable("transformation_matrix", shape=T_shape, regularizer=regularizer)
    T_matrix_batched = tf.tile(T_matrix, [batch_size, 1, 1, 1, 1])

    h_step = int((in_pose_shape[1] - h_kernel_size) / h_stride + 1)
    w_step = int((in_pose_shape[2] - w_kernel_size) / w_stride + 1)
    out_pose = []
    out_activation = []
    # start to do capsule convolution.
    # Note: there should be another way more computationally efficient to do this
    for i in range(h_step):
        col_pose = []
        col_prob = []
        h_s = i * h_stride
        h_e = h_s + h_kernel_size
        for j in range(w_step):
            with tf.variable_scope("transformation"):
                begin = [0, i * h_stride, j * w_stride, 0, 0, 0]
                size = batch_shape + in_caps_shape
                w_s = j * w_stride
                pose_sliced = in_pose[:, h_s:h_e, w_s:(w_s + w_kernel_size), :, :, :]
                pose_reshaped = tf.reshape(pose_sliced, shape=[batch_size, num_inputs, 1] + in_caps_shape)
                shape = [batch_size, num_inputs, filters] + in_caps_shape
                batch_pose = tf.multiply(pose_reshaped, tf.constant(1., shape=shape))
                vote = tf.reshape(tf.matmul(T_matrix_batched, batch_pose), shape=[batch_size, num_inputs, filters, -1])
                # do Coordinate Addition. Note: not yet completed
                if coordinate_addition:
                    x = j / w_step
                    y = i / h_step

            with tf.variable_scope("routing") as scope:
                if i > 0 or j > 0:
                    scope.reuse_variables()
                begin = [0, i * h_stride, j * w_stride, 0]
                size = [batch_size, h_kernel_size, w_kernel_size, in_channels]
                prob = tf.slice(activation, begin, size)
                prob = tf.reshape(prob, shape=[batch_size, -1, 1, 1])
                pose, prob = routing(vote, prob, filters, out_caps_shape, method="EMRouting", regularizer=regularizer)
            col_pose.append(pose)
            col_prob.append(prob)
        col_pose = tf.concat(col_pose, axis=2)
        col_prob = tf.concat(col_prob, axis=2)
        out_pose.append(col_pose)
        out_activation.append(col_prob)
    out_pose = tf.concat(out_pose, axis=1)
    out_activation = tf.concat(out_activation, axis=1)

    return (out_pose, out_activation)


def conv_slim_capsule(input_tensor,
                      input_dim,
                      output_dim,
                      layer_name,
                      input_atoms=8,
                      output_atoms=8,
                      stride=2,
                      kernel_size=5,
                      padding='SAME',
                      **routing_args):
    """Builds a slim convolutional capsule layer.

  This layer performs 2D convolution given 5D input tensor of shape
  `[batch, input_dim, input_atoms, input_height, input_width]`. Then refines
  the votes with routing and applies Squash non linearity for each capsule.

  Each capsule in this layer is a convolutional unit and shares its kernel over
  the position grid and different capsules of layer below. Therefore, number
  of trainable variables in this layer is:

    kernel: [kernel_size, kernel_size, input_atoms, output_dim * output_atoms]
    bias: [output_dim, output_atoms]

  Output of a conv2d layer is a single capsule with channel number of atoms.
  Therefore conv_slim_capsule is suitable to be added on top of a conv2d layer
  with num_routing=1, input_dim=1 and input_atoms=conv_channels.

  Args:
    input_tensor: tensor, of rank 5. Last two dimmensions representing height
      and width position grid.
    input_dim: scalar, number of capsules in the layer below.
    output_dim: scalar, number of capsules in this layer.
    layer_name: string, Name of this layer.
    input_atoms: scalar, number of units in each capsule of input layer.
    output_atoms: scalar, number of units in each capsule of output layer.
    stride: scalar, stride of the convolutional kernel.
    kernel_size: scalar, convolutional kernels are [kernel_size, kernel_size].
    padding: 'SAME' or 'VALID', padding mechanism for convolutional kernels.
    **routing_args: dictionary {leaky, num_routing}, args to be passed to the
      update_routing function.

  Returns:
    Tensor of activations for this layer of shape
      `[batch, output_dim, output_atoms, out_height, out_width]`. If padding is
      'SAME', out_height = in_height and out_width = in_width. Otherwise, height
      and width is adjusted with same rules as 'VALID' in tf.nn.conv2d.
  """
    with tf.variable_scope(layer_name):
        # convolution. return [batch_size, 1, 32, 8, 6, 6]
        kernel = variables.weight_variable(shape=[
            kernel_size, kernel_size, input_atoms, output_dim * output_atoms
        ])
        biases = variables.bias_variable([output_dim, output_atoms, 1, 1])
        votes, votes_shape, input_shape = _depthwise_conv3d(
            input_tensor, kernel, input_dim, output_dim, input_atoms, output_atoms,
            stride, padding)
        # convolution End

        with tf.name_scope('routing'):
            logit_shape = tf.stack([
                input_shape[0], input_dim, output_dim, votes_shape[2], votes_shape[3]
            ])
            biases_replicated = tf.tile(biases,
                                        [1, 1, votes_shape[2], votes_shape[3]])
            activations = _update_routing(
                votes=votes,
                biases=biases_replicated,
                logit_shape=logit_shape,
                num_dims=6,
                input_dim=input_dim,
                output_dim=output_dim,
                **routing_args)
        return activations


def _depthwise_conv3d(input_tensor,
                      kernel,
                      input_dim,
                      output_dim,
                      input_atoms=8,
                      output_atoms=8,
                      stride=2,
                      padding='SAME'):
    """Performs 2D convolution given a 5D input tensor.

  This layer given an input tensor of shape
  `[batch, input_dim, input_atoms, input_height, input_width]` squeezes the
  first two dimmensions to get a 4D tensor as the input of tf.nn.conv2d. Then
  splits the first dimmension and the last dimmension and returns the 6D
  convolution output.

  Args:
    input_tensor: tensor, of rank 5. Last two dimmensions representing height
      and width position grid.
    kernel: Tensor, convolutional kernel variables.
    input_dim: scalar, number of capsules in the layer below.
    output_dim: scalar, number of capsules in this layer.
    input_atoms: scalar, number of units in each capsule of input layer.
    output_atoms: scalar, number of units in each capsule of output layer.
    stride: scalar, stride of the convolutional kernel.
    padding: 'SAME' or 'VALID', padding mechanism for convolutional kernels.

  Returns:
    6D Tensor output of a 2D convolution with shape
      `[batch, input_dim, output_dim, output_atoms, out_height, out_width]`,
      the convolution output shape and the input shape.
      If padding is 'SAME', out_height = in_height and out_width = in_width.
      Otherwise, height and width is adjusted with same rules as 'VALID' in
      tf.nn.conv2d.
  """
    with tf.name_scope('conv'):
        input_shape = tf.shape(input_tensor)
        _, _, _, in_height, in_width = input_tensor.get_shape()
        # Reshape input_tensor to 4D by merging first two dimmensions.
        # tf.nn.conv2d only accepts 4D tensors.

        input_tensor_reshaped = tf.reshape(input_tensor, [
            input_shape[0] * input_dim, input_atoms, input_shape[3], input_shape[4]
        ])
        input_tensor_reshaped.set_shape((None, input_atoms, in_height.value,
                                         in_width.value))
        conv = tf.nn.conv2d(
            input_tensor_reshaped,
            kernel,
            [1, 1, stride, stride],
            padding=padding,
            data_format='NCHW')
        conv_shape = tf.shape(conv)
        _, _, conv_height, conv_width = conv.get_shape()
        # Reshape back to 6D by splitting first dimmension to batch and input_dim
        # and splitting second dimmension to output_dim and output_atoms.

        conv_reshaped = tf.reshape(conv, [
            input_shape[0], input_dim, output_dim, output_atoms, conv_shape[2],
            conv_shape[3]
        ])
        conv_reshaped.set_shape((None, input_dim, output_dim, output_atoms,
                                 conv_height.value, conv_width.value))
        return conv_reshaped, conv_shape, input_shape


def capsule(input_tensor,
            input_dim,
            output_dim,
            layer_name,
            input_atoms=8,
            output_atoms=8,
            **routing_args):
    """Builds a fully connected capsule layer.

  Given an input tensor of shape `[batch, input_dim, input_atoms]`, this op
  performs the following:

    1. For each input capsule, multiples it with the weight variable to get
      votes of shape `[batch, input_dim, output_dim, output_atoms]`.
    2. Scales the votes for each output capsule by iterative routing.
    3. Squashes the output of each capsule to have norm less than one.

  Each capsule of this layer has one weight tensor for each capsules of layer
  below. Therefore, this layer has the following number of trainable variables:
    w: [input_dim * num_in_atoms, output_dim * num_out_atoms]
    b: [output_dim * num_out_atoms]

  Args:
    input_tensor: tensor, activation output of the layer below.
    input_dim: scalar, number of capsules in the layer below.
    output_dim: scalar, number of capsules in this layer.
    layer_name: string, Name of this layer.
    input_atoms: scalar, number of units in each capsule of input layer.
    output_atoms: scalar, number of units in each capsule of output layer.
    **routing_args: dictionary {leaky, num_routing}, args for routing function.

  Returns:
    Tensor of activations for this layer of shape
      `[batch, output_dim, output_atoms]`.
  """
    with tf.variable_scope(layer_name):
        # weights variable will hold the state of the weights for the layer
        weights = variables.weight_variable(
            [input_dim, input_atoms, output_dim * output_atoms])
        biases = variables.bias_variable([output_dim, output_atoms])
        # Eq.2, u_hat = W * u
        with tf.name_scope('Wx_plus_b'):
            # Depthwise matmul: [b, d, c] ** [d, c, o_c] = [b, d, o_c]
            # To do this: tile input, do element-wise multiplication and reduce
            # sum over input_atoms dimmension.
            input_tiled = tf.tile(
                tf.expand_dims(input_tensor, -1),
                [1, 1, 1, output_dim * output_atoms])
            votes = tf.reduce_sum(input_tiled * weights, axis=2)
            votes_reshaped = tf.reshape(votes,
                                        [-1, input_dim, output_dim, output_atoms])
        # Eq.2 End, get votes_reshaped [batch_size, 1152, 10, 16]
        with tf.name_scope('routing'):
            input_shape = tf.shape(input_tensor)
            logit_shape = tf.stack([input_shape[0], input_dim, output_dim])
            # Routing algorithm, return [batch_size, 10, 16]
            activations = _update_routing(
                votes=votes_reshaped,
                biases=biases,
                logit_shape=logit_shape,
                num_dims=4,
                input_dim=input_dim,
                output_dim=output_dim,
                **routing_args)
        return activations
