#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: train.py
@time: 2020/02/10
@contact: wu.wei@pku.edu.cn
"""
import os
import yaml
import tensorflow as tf

from model_fn_builder import model_fn_builder
from file_based_input_fn_builder import file_based_input_fn_builder

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("config_name", "spanbert_base", "config name used to train the model.")
tf.flags.DEFINE_bool("output_dir", "data", "The output directory of the model training.")
tf.flags.DEFINE_bool("do_train", False, "Whether to run training.")
tf.flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
tf.flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")
tf.flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")
tf.flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
tf.flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training. This should be either the name "
                       "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
tf.flags.DEFINE_string("tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in. If not "
                       "specified, we will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string("gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project. If not "
                       "specified, we will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
tf.flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class Config(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yml')
    config_dict = yaml.safe_load(open(config_path))
    config = Config(config_dict[FLAGS.config_name])
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    tf.gfile.MakeDirs(FLAGS.output_dir)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=config.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = config.num_docs * config.num_epochs
    model_fn = model_fn_builder(config)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=1,
        eval_batch_size=1,
        predict_batch_size=1)

    if FLAGS.do_train:
        train_input_fn = file_based_input_fn_builder(
            input_file=config.train_file,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


if __name__ == '__main__':
    tf.app.run()
