# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time, sys, argparse
import tensorflow as tf
import numpy as np

from . import cifar10_layer
from . import cifar10_inference
from .cifar10_args import *

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = cifar10_layer.distorted_inputs()
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    
    logits = cifar10_layer.inference(images)
    
    # Calculate loss.
    loss = cifar10_layer.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10_layer.train(loss, global_step)
    
    #Evaluate Graph
    '''
    train_data = False
    train_images, train_labels = cifar10_layer.inputs(eval_data=train_data)
    train_logits = cifar10_layer.inference(train_images)
    top_1_train_op = tf.nn.in_top_k(train_logits, train_labels, 1)
    
    eval_data = Arguments.eval_data == 'test'
    eval_images, eval_labels = cifar10_layer.inputs(eval_data=eval_data)
    eval_logits = cifar10_layer.inference(eval_images)
    top_1_eval_op = tf.nn.in_top_k(eval_logits, eval_labels, 1)
    '''
    
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % Arguments.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = Arguments.log_frequency * Arguments.batch_size / duration
          sec_per_batch = float(duration / Arguments.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=Arguments.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=Arguments.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=Arguments.log_device_placement)) as mon_sess:
      i = 0
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
        
        '''
        i += 1
        if i%Arguments.eval_frequency == 0:
          do_eval(mon_sess, top_1_train_op, global_step, 'train')
          do_eval(mon_sess, top_1_eval_op, global_step, 'eval')
          i = 0
        '''

def do_eval(sess, eval_op, global_step, name = 'eval'):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = 20
  num_examples = steps_per_epoch * Arguments.batch_size
  for step in xrange(steps_per_epoch):
    predictions = sess.run(eval_op)
    true_count += np.sum(predictions)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1 %s: %0.04f' %
        (num_examples, true_count, name, precision))
  summary = tf.Summary()
  summary_writer = tf.summary.FileWriter(Arguments.eval_dir)
  #summary.ParseFromString(sess.run(eval_op))
  summary.value.add(tag='Precision @ 1 '+name, simple_value=precision)
  g_step = global_step.eval(session = sess)
  summary_writer.add_summary(summary, g_step)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10_layer.maybe_download_and_extract()
  if tf.gfile.Exists(Arguments.train_dir):
    tf.gfile.DeleteRecursively(Arguments.train_dir)
  tf.gfile.MakeDirs(Arguments.train_dir)
  train()


#if __name__ == '__main__':
#  tf.app.run()
