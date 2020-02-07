#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: model_fn_builder.py
@time: 2019/12/22
@contact: wu.wei@pku.edu.cn
"""
from corefqa_model import CorefQAModel
import tensorflow as tf

from bert import modeling
from optimization import create_custom_optimizer


def model_fn_builder(config):
    """Returns `model_fn` closure for TPUEstimator."""
    init_checkpoint = config.init_checkpoint
    coref_model = CorefQAModel(config)

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        predictions, losses = coref_model.forward(features, is_training)
        total_loss = tf.reduce_sum(losses)
        top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = predictions
        tvars = tf.trainable_variables()
        initialized_variables = {}
        scaffold_fn = None
        if init_checkpoint:
            assignment_map, initialized_variables = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if config.use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ", *INIT_FROM_CKPT*" if var.name in initialized_variables else ""
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = create_custom_optimizer(total_loss, config)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(loss):
                return {"eval_loss": tf.metrics.mean(loss)}

            eval_metrics = (metric_fn, [losses])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"top_span_starts": top_span_starts, "top_span_ends": top_span_ends, "loss": total_loss,
                             "top_antecedents": top_antecedents, "top_antecedent_scores": top_antecedent_scores},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn
