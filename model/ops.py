# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf





def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm",
               parameter_update_device='/cpu:0'):
    with tf.device(parameter_update_device):
        var = tf.contrib.layers.batch_norm(x, decay=decay, updates_collections=None, epsilon=epsilon,
                                        scale=True, is_training=is_training, scope=scope)

    return var


def conv2d(x, output_filters, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="conv2d",
           parameter_update_device='/cpu:0'):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()

        # W = tf.get_variable('W', [kh, kw, shape[-1], output_filters],
        #                     initializer=tf.truncated_normal_initializer(stddev=stddev))

        W = parameter_variable_creation_with_device_selection('W',shape=[kh, kw, shape[-1], output_filters],
                                                              initializer=tf.truncated_normal_initializer(stddev=stddev),
                                                              parameter_update_device=parameter_update_device)


        Wconv = tf.nn.conv2d(x, W, strides=[1, sh, sw, 1], padding='SAME')

        #biases = tf.get_variable('b', [output_filters], initializer=tf.constant_initializer(0.0))
        biases = parameter_variable_creation_with_device_selection('b',shape=[output_filters],
                                                                   initializer=tf.constant_initializer(0.0),
                                                                   parameter_update_device=parameter_update_device)

        Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())

        return Wconv_plus_b


def deconv2d(x, output_shape, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="deconv2d",
             parameter_update_device='/cpu:0'):
    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        input_shape = x.get_shape().as_list()
        # W = tf.get_variable('W', [kh, kw, output_shape[-1], input_shape[-1]],
        #                     initializer=tf.random_normal_initializer(stddev=stddev))
        W = parameter_variable_creation_with_device_selection('W',shape=[kh, kw, output_shape[-1], input_shape[-1]],
                                                              initializer=tf.random_normal_initializer(stddev=stddev),
                                                              parameter_update_device=parameter_update_device)

        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape,
                                        strides=[1, sh, sw, 1])

        # biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        biases = parameter_variable_creation_with_device_selection('b',shape=[output_shape[-1]],
                                                                   initializer=tf.constant_initializer(0.0),
                                                                   parameter_update_device=parameter_update_device)
        deconv_plus_b = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv_plus_b


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def fc(x, output_size, stddev=0.02, scope="fc",
       parameter_update_device='/cpu:0'):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        # W = tf.get_variable("W", [shape[1], output_size], tf.float32,
        #                     tf.random_normal_initializer(stddev=stddev))
        # b = tf.get_variable("b", [output_size],
        #                     initializer=tf.constant_initializer(0.0))

        W = parameter_variable_creation_with_device_selection("W", shape=[shape[1], output_size],
                                                              initializer=tf.random_normal_initializer(stddev=stddev),
                                                              parameter_update_device=parameter_update_device)
        b = parameter_variable_creation_with_device_selection("b", shape=[output_size],
                                                              initializer=tf.constant_initializer(0.0),
                                                              parameter_update_device=parameter_update_device)
        return tf.matmul(x, W) + b


def init_embedding_dictionary(size, dimension, stddev=0.01, scope="generator",
                              parameter_update_device='/cpu:0'):
    with tf.variable_scope(scope):
        # return tf.get_variable("gen_ebdd_dictionary", [size, dimension], tf.float32,
        #                        tf.random_normal_initializer(stddev=stddev))

        return parameter_variable_creation_with_device_selection("gen_ebdd_dictionary",shape=[size, dimension],
                                                                 initializer=tf.random_normal_initializer(stddev=stddev),
                                                                 parameter_update_device=parameter_update_device)


def init_embedding_weights(size, stddev=1, scope="generator", name='tmp',
                           parameter_update_device='/cpu:0'):
    with tf.variable_scope(scope):
        # init_weight = tf.get_variable(name, size, tf.float32,
        #                               tf.random_normal_initializer(stddev=stddev))
        init_weight = parameter_variable_creation_with_device_selection(name,shape=size,
                                                                        initializer=tf.random_normal_initializer(stddev=stddev),
                                                                        parameter_update_device=parameter_update_device)

        # if weight_norm_mark==True:
        #     init_weight=weight_norm(input=init_weight)
        return init_weight


def weight_norm(input):
    # output=input
    sum_value = tf.reduce_sum(input, axis=1)
    sum_value = tf.expand_dims(sum_value, axis=1)
    # sum_value=tf.transpose(sum_value)
    one_multipliers = tf.ones([1, int(input.shape[1])], dtype=tf.float32)
    sum_value = tf.matmul(sum_value, one_multipliers)
    output = tf.truediv(input, tf.abs(sum_value))
    return output


def conditional_instance_norm(x, ids, labels_num, mixed=False, scope="conditional_instance_norm"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        batch_size, output_filters = shape[0], shape[-1]
        scale = tf.get_variable("scale", [labels_num, output_filters], tf.float32, tf.constant_initializer(1.0))
        shift = tf.get_variable("shift", [labels_num, output_filters], tf.float32, tf.constant_initializer(0.0))

        mu, sigma = tf.nn.moments(x, [1, 2], keep_dims=True)
        norm = (x - mu) / tf.sqrt(sigma + 1e-5)

        batch_scale = tf.reshape(tf.nn.embedding_lookup([scale], ids=ids), [batch_size, 1, 1, output_filters])
        batch_shift = tf.reshape(tf.nn.embedding_lookup([shift], ids=ids), [batch_size, 1, 1, output_filters])

        z = norm * batch_scale + batch_shift
        return z


def parameter_variable_creation_with_device_selection(name, shape, initializer,
                                                      parameter_update_device='/cpu:0'):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device(parameter_update_device):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var