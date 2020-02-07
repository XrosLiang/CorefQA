#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: utils.py
@time: 2020/01/06
@contact: wu.wei@pku.edu.cn
"""
import tensorflow as tf


def shape(x, dim):
    return x.get_shape()[dim].value or tf.shape(x)[dim]


def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout,
         output_weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
         hidden_initializer=tf.truncated_normal_initializer(stddev=0.02)):
    if len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

    if len(inputs.get_shape()) == 3:
        batch_size = shape(inputs, 0)
        seqlen = shape(inputs, 1)
        emb_size = shape(inputs, 2)
        current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size],
                                         initializer=hidden_initializer)
        hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size], initializer=tf.zeros_initializer())
        current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, dropout)
        current_inputs = current_outputs

    output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size],
                                     initializer=output_weights_initializer)
    output_bias = tf.get_variable("output_bias", [output_size], initializer=tf.zeros_initializer())
    outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
    if output_size == 1:
        outputs = tf.squeeze(outputs, -1)
    return outputs


def batch_gather(emb, indices):
    batch_size = shape(emb, 0)
    seqlen = shape(emb, 1)
    if len(emb.get_shape()) > 2:
        emb_size = shape(emb, 2)
    else:
        emb_size = 1
    flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
    offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]
    gathered = tf.gather(flattened_emb, indices + offset)  # [batch_size, num_indices, emb]
    if len(emb.get_shape()) == 2:
        gathered = tf.squeeze(gathered, 2)  # [batch_size, num_indices]
    return gathered
