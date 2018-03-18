"""Models for gene expression from DNA."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class ProteinBindingConv(t2t_model.T2TModel):
  """Protein binding conv net.

  """

  def body(self, features):
    inputs = features["inputs"]
    inputs.get_shape().assert_has_rank(4)

    hp = self._hparams

    out = inputs
    out = common_layers.flatten4d3d(out)

    # Conv layers
    assert hp.num_conv_layers == len(hp.pooling_windows)
    for i in xrange(hp.num_conv_layers):
      out = conv_layer(
          out,
          hp.hidden_size,
          hp.kernel_width,
          hp.stride,
          hp.pooling_windows[i],
          hp.dropout,
          dilation_rate=1,
          gated_linear=hp.gated_linear,
          name="conv_%d" % (i + 1))

    # Dense dilated conv layers
    for i in xrange(hp.num_dconv_layers):
      dilation_rate = 2**(i + 1)
      dconv_out = conv_layer(
          out,
          hp.hidden_size,
          hp.kernel_width,
          stride=1,
          pooling_window=0,
          dropout_rate=hp.dropout,
          dilation_rate=dilation_rate,
          gated_linear=hp.gated_linear,
          name="dconv_%d" % (i + 1))
      out = tf.concat([out, dconv_out], axis=2)

    # Fully connected layer
    out = fc_layer(out, hp.hidden_size, hp.dropout, name="fc")

    out.get_shape().assert_has_rank(3)
    out = tf.expand_dims(out, 2)
    return out


def conv_layer(x,
               hidden_size,
               kernel_size,
               stride,
               pooling_window,
               dropout_rate,
               dilation_rate,
               gated_linear=False,
               name="conv"):
  with tf.variable_scope(name):
    out = x
    out = common_layers.conv1d_block(
        out,
        hidden_size, [(dilation_rate, kernel_size)],
        strides=stride,
        first_relu=False,
        padding="same")
    out = tf.nn.relu(out)
    if gated_linear:
      gate = common_layers.conv1d_block(
          out,
          hidden_size, [(dilation_rate, kernel_size)],
          strides=stride,
          first_relu=False,
          padding="same")
      out = out * tf.nn.sigmoid(out)
    if pooling_window:
      out = tf.layers.max_pooling1d(
          out, pooling_window, pooling_window, padding="same")
    out = tf.layers.dropout(out, dropout_rate)
    return out


def fc_layer(x, num_out, dropout_rate, name="fc"):
  with tf.variable_scope(name):
    out = x
    out = tf.layers.dense(out, num_out)
    out = tf.contrib.layers.layer_norm(out)
    out = tf.nn.relu(out)
    out = tf.layers.dropout(out, dropout_rate)
    return out


@registry.register_hparams
def pb_conv_base():
  """Hparams for ProteinBindingConv model."""
  hparams = common_hparams.basic_params1()

  hparams.batch_size = 64

  hparams.dropout = 0.1
  hparams.add_hparam("num_conv_layers", 4)
  hparams.add_hparam("num_dconv_layers", 0)
  # Whether to use gated-linear convolutions from
  # https://arxiv.org/pdf/1612.08083.pdf
  hparams.add_hparam("gated_linear", True)
  # The product of these pooling windows should match
  # input_length/target_length.
  hparams.add_hparam("pooling_windows", [2, 2, 2, 4])

  hparams.hidden_size = 925
  hparams.kernel_width = 8
  hparams.add_hparam("stride", 1)
  return hparams
  
