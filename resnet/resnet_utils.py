# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

More variants were introduced in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks. arXiv: 1603.05027, 2016

We can obtain different ResNet variants by changing the network depth, width,
and form of residual unit. This module implements the infrastructure for
building them. Concrete ResNet units and full ResNet networks are implemented in
the accompanying resnet_v1.py and resnet_v2.py modules.

Compared to https://github.com/KaimingHe/deep-residual-networks, in the current
implementation we subsample the output activations in the last residual unit of
each block, instead of subsampling the input activations in the first residual
unit of each block. The two implementations give identical results but our
implementation is more memory efficient.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import tensorflow.compat.v1 as tf1

import tf_slim as slim

# Added by No@
from tf_slim.ops.arg_scope import add_arg_scope
from tensorflow.python.framework import ops
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
                                    Activation, MaxPool2D
from tf_slim.layers import utils
from icecream import ic

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing a ResNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """


def subsample(inputs, factor, scope=None):
  """Subsamples the input along the spatial dimensions.

  Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    scope: Optional variable_scope.

  Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return max_pool(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return conv(inputs, num_outputs, kernel_size, stride=1, 
                rate=rate, padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return conv(inputs, num_outputs, kernel_size, stride=stride,
                rate=rate, padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, multi_grid, output_stride=None,
                       outputs_collections=None):
  """Stacks ResNet `Blocks` and controls output feature density.

  First, this function creates scopes for the ResNet in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the ResNet
  output_stride, which is the ratio of the input to output spatial resolution.
  This is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Most ResNets consist of 4 ResNet blocks and subsample the activations by a
  factor of 2 when transitioning between consecutive ResNet blocks. This results
  to a nominal ResNet output_stride equal to 8. If we set the output_stride to
  half the nominal network stride (e.g., output_stride=4), then we compute
  responses twice.

  Control of the output feature density is implemented by atrous convolution.

  Args:
    net: A `Tensor` of size [batch, height, width, channels].
    blocks: A list of length equal to the number of ResNet `Blocks`. Each
      element is a ResNet `Block` object describing the units in the `Block`.
    output_stride: If `None`, then the output will be computed at the nominal
      network stride. If output_stride is not `None`, it specifies the requested
      ratio of input to output spatial resolution, which needs to be equal to
      the product of unit strides from the start up to some level of the ResNet.
      For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
      then valid values for the output_stride are 1, 2, 6, 24 or None (which
      is equivalent to output_stride=24).
    outputs_collections: Collection to add the ResNet block outputs.

  Returns:
    net: Output tensor with stride equal to the specified output_stride.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  for block in blocks:
    with tf.name_scope(block.scope) as sc:

      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')
        
        with tf.name_scope('unit_%d' % (i + 1)):# as sc_:
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            # Only uses atrous convolutions with multi-graid rates in the last (block4) block
            if block.scope == "block4":
              net = block.unit_fn(net, rate=rate * multi_grid[i], **dict(unit, stride=1))
            else:
              net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
            rate *= unit.get('stride', 1)
          else:
            net = block.unit_fn(net, rate=1, **unit)
            current_stride *= unit.get('stride', 1)
      net = slim.utils.collect_named_outputs(outputs_collections, sc, net)

  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net

def resnet_arg_scope(weight_decay=0.0001,
                     is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn="relu",
                     use_batch_norm=True):
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': None,
      'is_training': is_training,
      'fused': True,  # Use fused batch norm if possible.
  }
  with slim.arg_scope(
      [conv],
      kernel_regularizer = tf.keras.regularizers.L2(weight_decay),
      kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0),
      activation_fn=activation_fn,
      normalizer_fn=batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      with slim.arg_scope([max_pool], padding='SAME') as arg_sc:
        return arg_sc

# Added by No@
@add_arg_scope
def max_pool(inputs,
             kernel_size,
             stride=2,
             padding='SAME',
             outputs_collections=None,
             scope=None):
    
    with tf.name_scope(scope) as sc:
      outputs = MaxPool2D(pool_size=kernel_size, 
                          strides=stride, 
                          padding=padding)(inputs)      
      return utils.collect_named_outputs(outputs_collections, sc, outputs)

@add_arg_scope
def conv(inputs,
         filters,
         kernel_size,
         stride=(1, 1),
         padding='same',
         data_format=None,
         rate=(1, 1),
         activation_fn=None,
         normalizer_fn=None,
         normalizer_params=None,
         kernel_initializer='glorot_uniform',
         bias_initializer='zeros',
         kernel_regularizer=None,
         bias_regularizer=None,
         outputs_collections=None,
         trainable=True,
         reuse=None,
         scope=None):

    with tf.name_scope(scope) as sc:
      outputs = Conv2D(filters,
                      kernel_size,
                      strides=stride,
                      padding=padding,
                      data_format=data_format,
                      dilation_rate=rate,
                      activation=activation_fn,
                      use_bias=not normalizer_fn and bias_initializer,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer)(inputs)

      if normalizer_fn is not None:
        normalizer_params = normalizer_params or {}
        outputs = normalizer_fn(outputs, **normalizer_params)

      if activation_fn is not None:
        outputs = Activation(activation_fn)(outputs)  

      return utils.collect_named_outputs(outputs_collections, sc, outputs)

@add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               outputs_collections=None,
               is_training=True,
               fused=None,
               scope=None,
               reuse=None):
          
    if scope is None: scope="Batchnorm"
    with tf.name_scope(scope) as sc:
      outputs = BatchNormalization(momentum=decay, epsilon=epsilon, scale=scale,
                                fused=fused)(inputs, training=is_training)

      if activation_fn is not None:
        outputs = Activation(activation_fn)(outputs)  
      
      return utils.collect_named_outputs(outputs_collections, sc, outputs)
