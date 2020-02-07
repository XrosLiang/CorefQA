#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: file_based_input_fn_builder.py
@time: 2019/12/22
@contact: wu.wei@pku.edu.cn

build input_fn
"""
import tensorflow as tf


def file_based_input_fn_builder(input_file, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        'doc_idx': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'sentence_map': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'subtoken_map': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'flattened_input_ids': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'flattened_input_mask': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'span_starts': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'span_ends': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'cluster_ids': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn
