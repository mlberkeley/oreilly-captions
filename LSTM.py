from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

import configuration

def build_model(image_embeddings, seq_embeddings, target_seqs, input_mask, mode='train'):
    """Builds the model.
    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).

    config = configuration.ModelConfig()

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
        num_units=config.num_lstm_units, state_is_tuple=True)
    if mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=config.lstm_dropout_keep_prob,
          output_keep_prob=config.lstm_dropout_keep_prob)

    with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      zero_state = lstm_cell.zero_state(
          batch_size=image_embeddings.get_shape()[0], dtype=tf.float32)
      _, initial_state = lstm_cell(image_embeddings, zero_state)

      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()

      
      # Run the batch of sequence embeddings through the LSTM.
      sequence_length = tf.reduce_sum(input_mask, 1)
      lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                          inputs=seq_embeddings,
                                          sequence_length=sequence_length,
                                          initial_state=initial_state,
                                          dtype=tf.float32,
                                          scope=lstm_scope)

    # Stack batches vertically.
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    with tf.variable_scope("logits") as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=lstm_outputs,
          num_outputs=config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_scope)

    targets = tf.reshape(target_seqs, [-1])
    weights = tf.to_float(tf.reshape(input_mask, [-1]))

    # Compute losses.
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                            logits=logits)
    batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                        tf.reduce_sum(weights),
                        name="batch_loss")
    tf.losses.add_loss(batch_loss)
    total_loss = tf.losses.get_total_loss()

    # Add summaries.
    tf.summary.scalar("losses/batch_loss", batch_loss)
    tf.summary.scalar("losses/total_loss", total_loss)
    for var in tf.trainable_variables():
      tf.summary.histogram("parameters/" + var.op.name, var)

    total_loss = total_loss
    target_cross_entropy_losses = losses  # Used in evaluation.
    target_cross_entropy_loss_weights = weights  # Used in evaluation.