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

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
from pprint import pprint
import numpy as np

from six.moves import urllib
import tensorflow as tf

from . import cifar10_input
from .cifar10_args import * 
from .cifar10_inference import *

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if Arguments.use_fp16 else tf.float32
    
    #check if graph varible is exit, reuse it if exit
    try:
      var = tf.get_variable(name, shape,initializer=initializer, dtype=dtype)
    except ValueError:
      tf.get_variable_scope().reuse_variables()
      var = tf.get_variable(name)

  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if Arguments.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not Arguments.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(Arguments.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=Arguments.batch_size)
  if Arguments.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not Arguments.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(Arguments.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=Arguments.batch_size)
  if Arguments.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def conv_layer(input, size_in, size_out, name="conv", f_size=3, stride=1, pad="SAME"):
  with tf.variable_scope(name) as scope:
    W = _variable_with_weight_decay('W', shape=[f_size, f_size, size_in, size_out], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding=pad)
    b = _variable_on_cpu('b', [size_out], tf.constant_initializer(0.01))
    conv_b = tf.nn.bias_add(conv, b)
    conv_b_relu = tf.nn.relu(conv_b, name=scope.name)
    _activation_summary(conv_b_relu)
    return conv_b_relu

def res_conv_layer(input, res, size_in, size_out, name="res_conv", f_size=3, stride=1, pad="SAME"):
  with tf.variable_scope(name) as scope:
    W = _variable_with_weight_decay('W', shape=[f_size, f_size, size_in, size_out], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(input, W, strides=[stride, stride, stride, stride], padding=pad)
    b = _variable_on_cpu('b', [size_out], tf.constant_initializer(0.01))
    conv_b = tf.nn.bias_add(conv, b)
    
    diff = conv_b.get_shape().as_list()[3] - res.get_shape().as_list()[3]
    diff_tensor = tf.cast(tf.convert_to_tensor([[0,0],[0,0],[0,0],[0, diff]]),tf.int32)

    if(diff == 0):
      res_conv_b = tf.add(conv_b,res)
    else:
      #res = tf.pad(res, diff_tensor, "CONSTANT")
      res = conv_layer(res, res.get_shape().as_list()[3], conv_b.get_shape().as_list()[3],'res_conv_pad',1)
      res_conv_b = tf.add(conv_b,res)

    res_conv_b_relu = tf.nn.relu(res_conv_b, name=scope.name)
    _activation_summary(res_conv_b_relu)
    return res_conv_b_relu

def fc_layer(input, size_in, size_out, name="fc"):
  with tf.variable_scope(name) as scope:
    W = _variable_with_weight_decay('W', shape=[size_in, size_out], stddev=0.04, wd=0.004)
    b = _variable_on_cpu('b', [size_out], tf.constant_initializer(0.01))
    fc = tf.nn.relu(tf.add(tf.matmul(input, W), b), name=scope.name)
    _activation_summary(fc)
    return fc

def bn_layer(input, name='bn'):
  with tf.variable_scope(name) as scope:
    mean, var = tf.nn.moments(input,[0,1,2])
    gamma = _variable_on_cpu('gamma', [input.get_shape()[3]], tf.constant_initializer(0.01))
    beta = _variable_on_cpu('beta', [input.get_shape()[3]], tf.constant_initializer(0.01))
    bn = tf.nn.batch_normalization(input, mean, var, beta, gamma, variance_epsilon=1e-10, name=scope.name)
    _activation_summary(bn)
    return bn

def vgg_layer(input, size_out, n_conv, name='vgg'):
  for i in range(n_conv):
    input = conv_layer(input,input.get_shape()[3],size_out,name=name+"-conv"+str(i))
  bn = bn_layer(input, name=name+"norm")
  pool = tf.nn.max_pool(bn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name=name+'pool')
  return pool

def res_layer(input, size_out, name="res"):
  with tf.variable_scope(name) as scope:
    # compute dim different
    diff = input.get_shape().as_list()[3] - size_out
    # conv1
    if(diff == 0):
      conv1 = conv_layer(input,input.get_shape()[3],size_out,name=name+"-conv1")
    else:
      conv1 = conv_layer(input,input.get_shape()[3],size_out,name=name+"-conv1_dim_down",stride=2)
    
    # conv2
    W = _variable_with_weight_decay('W', shape=[3, 3, size_out, size_out], stddev=5e-2, wd=0.0)
    conv2 = tf.nn.conv2d(conv1, W, strides=[1, 1, 1, 1], padding='SAME')
    b = _variable_on_cpu('b', [size_out], tf.constant_initializer(0.01))
    conv_b = tf.nn.bias_add(conv2, b)

    if(diff == 0):
      res_conv_b = tf.add(conv_b,input)
    else:
      res = conv_layer(input, input.get_shape().as_list()[3], size_out,'res_dim_up',f_size=1,stride=2)
      res_conv_b = tf.add(conv_b,res)

    res_conv_b_relu = tf.nn.relu(res_conv_b, name=scope.name)
    _activation_summary(res_conv_b_relu)
    bn = bn_layer(res_conv_b_relu, name=name+"norm")
    return bn

def resnet_layer(input, size_out, name = 'resnet'):
  conv1 = conv_layer(input,input.get_shape()[3],size_out,name=name+"-conv")
  conv2 = res_conv_layer(conv1,input,conv1.get_shape()[3],size_out,name=name+"-res_conv")
  bn = bn_layer(conv2, name=name+"norm")
  return bn

def inference(images):
  if Arguments.inference == "resnet3":
    return resnet3(images)
  elif Arguments.inference == "resnet1":
    return resnet1(images)
  elif Arguments.inference == "resnet5":
    return resnet5(images)
  elif Arguments.inference == "vggA":
    return vggA(images)
  elif Arguments.inference == "vggB":
    return vggB(images)
  elif Arguments.inference == "vggD":
    return vggD(images)
  elif Arguments.inference == "vggE":
    return vggE(images)
  

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / Arguments.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    #opt = tf.train.AdamOptimizer(lr)
    opt = tf.train.MomentumOptimizer(lr,0.9)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = Arguments.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def resnet1(images):
  
  #resnet
  conv1 = conv_layer(images,images.get_shape()[3],64,"conv1",7)
  bn = bn_layer(conv1, "norm1")
  pool1 = tf.nn.max_pool(bn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')

  res1 = res_layer(pool1, 64, name='res1')
  res2 = res_layer(res1, 64, name='res2') 
  res3 = res_layer(res2, 64, name='res3')
  res4 = res_layer(res3, 128, name='res4')
  res5 = res_layer(res4, 128, name='res5')
  res6 = res_layer(res5, 128, name='res6')
  res7 = res_layer(res6, 128, name='res7')
  res8 = res_layer(res7, 128, name='res8')
  res9 = res_layer(res8, 256, name='res9')
  res10 = res_layer(res9, 256, name='res10')
  res11 = res_layer(res10, 256, name='res11')
  res12 = res_layer(res11, 256, name='res12')
  res13 = res_layer(res12, 256, name='res13')
  
  pool2 = tf.nn.avg_pool(res13, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],padding='SAME', name='pool2')
  
  # local3 (fully connected layer)
  reshape = tf.reshape(pool2, [Arguments.batch_size, -1])
  local4 = fc_layer(reshape, reshape.get_shape()[1].value, 1000, 'local4')

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  softmax_linear = fc_layer(local4, 1000, NUM_CLASSES, 'softmax_linear')
  
  softmax = tf.nn.softmax(softmax_linear,name='softmax')
  _activation_summary(softmax)
 
  return softmax_linear

def resnet3(images):
  
  #resnet
  conv1 = conv_layer(images,images.get_shape()[3],32,"conv1",3)
  bn = bn_layer(conv1, "norm1")
  #pool1 = tf.nn.max_pool(bn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')

  res1 = resnet_layer(bn, 32, name='res1')
  res2 = resnet_layer(res1, 32, name='res2') 
  res3 = resnet_layer(res2, 32, name='res3')
  res4 = resnet_layer(res3, 64, name='res4')
  res5 = resnet_layer(res4, 64, name='res5')
  res6 = resnet_layer(res5, 64, name='res6')
  res7 = resnet_layer(res6, 128, name='res7')
  res8 = resnet_layer(res7, 128, name='res8')
  res9 = resnet_layer(res8, 256, name='res9')
  res10 = resnet_layer(res9, 256, name='res10')
  res11 = resnet_layer(res10, 512, name='res11')
  res12 = resnet_layer(res11, 512, name='res12')
  res13 = resnet_layer(res12, 512, name='res13')
  pool2 = tf.nn.avg_pool(res13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool2')
  
  # local3 (fully connected layer)
  reshape = tf.reshape(pool2, [Arguments.batch_size, -1])
  dim = reshape.get_shape()[1].value
  #local3 = fc_layer(reshape, dim, 4096, 'local3')
    
  # local4
  #local4 = fc_layer(local3, 4096, 1000, 'local4')
  local4 = fc_layer(reshape, dim, 1000, 'local4')

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  softmax_linear = fc_layer(local4, 1000, NUM_CLASSES, 'softmax_linear')
  
  softmax = tf.nn.softmax(softmax_linear,name='softmax')
  _activation_summary(softmax)
 
  return softmax_linear

def resnet5(images):
  
  #resnet
  conv1 = conv_layer(images,images.get_shape()[3],64,"conv1",7)
  bn = bn_layer(conv1, "norm1")
  pool1 = tf.nn.max_pool(bn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')

  res1 = res_layer(pool1, 64, name='res1')
  res2 = res_layer(res1, 64, name='res2') 
  res3 = res_layer(res2, 64, name='res3')
  res4 = res_layer(res3, 64, name='res4') 
  res5 = res_layer(res4, 64, name='res5')
  res6 = res_layer(res5, 128, name='res6')
  res7 = res_layer(res6, 128, name='res7')
  res8 = res_layer(res7, 128, name='res8')
  res9 = res_layer(res8, 128, name='res9')
  res10 = res_layer(res9, 128, name='res10')
  res11 = res_layer(res10, 256, name='res11')
  res12 = res_layer(res11, 256, name='res12')
  res13 = res_layer(res12, 256, name='res13')
  res14 = res_layer(res13, 256, name='res14')
  res15 = res_layer(res14, 256, name='res15')
  
  pool2 = tf.nn.avg_pool(res15, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],padding='SAME', name='pool2')
  
  # local3 (fully connected layer)
  reshape = tf.reshape(pool2, [Arguments.batch_size, -1])
  local4 = fc_layer(reshape, reshape.get_shape()[1].value, 1000, 'local4')

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  softmax_linear = fc_layer(local4, 1000, NUM_CLASSES, 'softmax_linear')
  
  softmax = tf.nn.softmax(softmax_linear,name='softmax')
  _activation_summary(softmax)
 
  return softmax_linear

def vggA(images):

  vgg1 = vgg_layer(images, 64, 1, name='vgg1')
  vgg2 = vgg_layer(vgg1, 128, 1, name='vgg2')
  vgg3 = vgg_layer(vgg2, 256, 2, name='vgg3')
  vgg4 = vgg_layer(vgg3, 512, 2, name='vgg4')
  vgg5 = vgg_layer(vgg4, 512, 2, name='vgg5')

  
  # local3 (fully connected layer)
  reshape = tf.reshape(vgg5, [Arguments.batch_size, -1])
  dim = reshape.get_shape()[1].value
  local3 = fc_layer(reshape, dim, 4096, 'local3')
    
  # local4
  local4 = fc_layer(local3, 4096, 1000, 'local4')

  # linear layer(WX + b),
  softmax_linear = fc_layer(local4, 1000, NUM_CLASSES, 'softmax_linear')
  
  # softmax,
  softmax = tf.nn.softmax(softmax_linear,name='softmax')
  _activation_summary(softmax)
 
  #however, tf do softmax when it compute softmax loss, therefore return linear layer
  return softmax_linear

def vggB(images):

  vgg1 = vgg_layer(images, 64, 2, name='vgg1')
  vgg2 = vgg_layer(vgg1, 128, 2, name='vgg2')
  vgg3 = vgg_layer(vgg2, 256, 2, name='vgg3')
  vgg4 = vgg_layer(vgg3, 512, 2, name='vgg4')
  vgg5 = vgg_layer(vgg4, 512, 2, name='vgg5')

  
  # local3 (fully connected layer)
  reshape = tf.reshape(vgg5, [Arguments.batch_size, -1])
  dim = reshape.get_shape()[1].value
  local3 = fc_layer(reshape, dim, 4096, 'local3')
    
  # local4
  local4 = fc_layer(local3, 4096, 1000, 'local4')

  # linear layer(WX + b),
  softmax_linear = fc_layer(local4, 1000, NUM_CLASSES, 'softmax_linear')
  
  # softmax,
  softmax = tf.nn.softmax(softmax_linear,name='softmax')
  _activation_summary(softmax)
 
  #however, tf do softmax when it compute softmax loss, therefore return linear layer
  return softmax_linear

def vggD(images):

  vgg1 = vgg_layer(images, 64, 2, name='vgg1')
  vgg2 = vgg_layer(vgg1, 128, 2, name='vgg2')
  vgg3 = vgg_layer(vgg2, 256, 3, name='vgg3')
  vgg4 = vgg_layer(vgg3, 512, 3, name='vgg4')
  vgg5 = vgg_layer(vgg4, 512, 3, name='vgg5')

  
  # local3 (fully connected layer)
  reshape = tf.reshape(vgg5, [Arguments.batch_size, -1])
  dim = reshape.get_shape()[1].value
  local3 = fc_layer(reshape, dim, 4096, 'local3')
    
  # local4
  local4 = fc_layer(local3, 4096, 1000, 'local4')

  # linear layer(WX + b),
  softmax_linear = fc_layer(local4, 1000, NUM_CLASSES, 'softmax_linear')
  
  # softmax,
  softmax = tf.nn.softmax(softmax_linear,name='softmax')
  _activation_summary(softmax)
 
  #however, tf do softmax when it compute softmax loss, therefore return linear layer
  return softmax_linear

def vggE(images):

  vgg1 = vgg_layer(images, 64, 2, name='vgg1')
  vgg2 = vgg_layer(vgg1, 128, 2, name='vgg2')
  vgg3 = vgg_layer(vgg2, 256, 4, name='vgg3')
  vgg4 = vgg_layer(vgg3, 512, 4, name='vgg4')
  vgg5 = vgg_layer(vgg4, 512, 4, name='vgg5')

  
  # local3 (fully connected layer)
  reshape = tf.reshape(vgg5, [Arguments.batch_size, -1])
  dim = reshape.get_shape()[1].value
  local3 = fc_layer(reshape, dim, 4096, 'local3')
    
  # local4
  local4 = fc_layer(local3, 4096, 1000, 'local4')

  # linear layer(WX + b),
  softmax_linear = fc_layer(local4, 1000, NUM_CLASSES, 'softmax_linear')
  
  # softmax,
  softmax = tf.nn.softmax(softmax_linear,name='softmax')
  _activation_summary(softmax)
 
  #however, tf do softmax when it compute softmax loss, therefore return linear layer
  return softmax_linear