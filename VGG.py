from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

def vgg_16_base(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      net = layers_lib.repeat(
          inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
      end_points = utils.convert_collection_to_dict(end_points_collection)
      end_points[sc.name+'/p5']=net
      return net,end_points
vgg_16.default_image_size = 224

def vgg_16(images,
                 trainable=True,
                 is_training=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 scope="vgg_16"):
  """Builds an VGG16 subgraph for image embeddings.
  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the vgg16 submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.
  Returns:
    end_points: A dictionary of activations from vgg16 layers.
  """
  # Only consider the vgg model to be in training mode if it's trainable.
  is_vgg_model_training = trainable and is_training


  # DEPRECATED
  # if use_batch_norm:
  #   # Default parameters for batch normalization.
  #   if not batch_norm_params:
  #     batch_norm_params = {
  #         "is_training": is_vgg_model_training,
  #         "trainable": trainable,
  #         # Decay for the moving averages.
  #         "decay": 0.9997,
  #         # Epsilon to prevent 0s in variance.
  #         "epsilon": 0.001,
  #         # Collection containing the moving mean and moving variance.
  #         "variables_collections": {
  #             "beta": None,
  #             "gamma": None,
  #             "moving_mean": ["moving_vars"],
  #             "moving_variance": ["moving_vars"],
  #         }
  #     }
  # else:
  #   batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, "vgg_16", [images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      # DEPRECATED
      # with slim.arg_scope(
      #     [slim.conv2d],
      #     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
      #     activation_fn=tf.nn.relu,
      #     normalizer_fn=slim.batch_norm,
      #     normalizer_params=batch_norm_params):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm):
        net, end_points = vgg_16_base(images, scope=scope)
        with tf.variable_scope("logits"):
          shape = net.get_shape()
          net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
          net = slim.dropout(
              net,
              keep_prob=dropout_keep_prob,
              is_training=is_vgg_model_training,
              scope="dropout")
          net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  return net