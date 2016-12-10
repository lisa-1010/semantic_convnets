# -*- coding: utf-8 -*-
"""
TODOs:
- add confidence score predictor for both coarse and fine

"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf

# Convolutional network building
def build_network(n_classes, get_hidden_reps=False):
    #assert n_output_units is not None, \
    #    "You need to specify how many tokens are in the output classification sequence."
    # n_classes represents the total number of classes

    # input tensor is prefeature_size x n_output_units
    n_output_units = 2
    prefeature_embedding_size = 512
    single_output_token_size = (100 + 20 + 1)

    net = input_data(shape=[None, prefeature_embedding_size])

    # Basically, this repeats the input several times to be fed into the LSTM
    net = tf.tile(net, [1, n_output_units])
    net = tf.reshape(net, [-1, n_output_units, prefeature_embedding_size])

    net = tflearn.lstm(net, single_output_token_size, return_seq=True) # This returns [# of samples, # of timesteps, output dim]

    fine_network, coarse_network = net
    fine_network = fully_connected(fine_network, single_output_token_size, activation='softmax')
    coarse_network = fully_connected(coarse_network, single_output_token_size, activation='softmax')

    stacked_coarse_and_fine_net = tf.concat(1, [coarse_network, fine_network])

    #net = tf.reshape(net, [-1, n_output_units * prefeature_embedding_size]) # This reshapes so that the outputs are stacked
    #net = fully_connected(net, 2*single_output_token_size, activation='softmax')

    # We need to split this and attach different losses
    target_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, n_output_units*single_output_token_size))

    def coarse_and_fine_joint_loss(incoming, placeholder):
        coarse_pred = incoming[:, :single_output_token_size]
        coarse_target = placeholder[:, :single_output_token_size]
        coarse_loss = tflearn.categorical_crossentropy(coarse_pred, coarse_target)

        fine_pred = incoming[:, single_output_token_size:]
        fine_target = placeholder[:, single_output_token_size:]
        fine_loss = tflearn.categorical_crossentropy(fine_pred, fine_target)

        return coarse_loss + fine_loss

    def coarse_and_fine_accuracy(y_pred, y_true, x):
        coarse_pred = y_pred[:, :single_output_token_size]
        fine_pred = y_pred[:, single_output_token_size:]
        coarse_target = y_true[:, :single_output_token_size]
        fine_target = y_true[:, single_output_token_size:]

        coarse_acc = tflearn.metrics.accuracy_op(coarse_pred, coarse_target)
        fine_acc = tflearn.metrics.accuracy_op(fine_pred, fine_target)

        # rounded_coarse_acc = tf.to_float(tf.round(coarse_acc * 1000) * 100000)
        # return tf.add(rounded_coarse_acc, fine_acc)
        return (fine_acc + coarse_acc)/2.0

    net = regression(stacked_coarse_and_fine_net , placeholder=target_placeholder, optimizer='adam',
                             loss=coarse_and_fine_joint_loss,
                             #metric=coarse_and_fine_accuracy,
                             learning_rate=0.001)

    return net