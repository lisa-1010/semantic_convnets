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
def build_network(output_dims=[20, 100], get_hidden_reps=False, get_fc_softmax_activations=False):


    assert (len(output_dims) == 2), "output_dims needs to be of length 2, containing coarse_dim and fine_dim."
    # outputdims is a list of num_classes

    coarse_dim, fine_dim = tuple(output_dims)
    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    network = input_data(shape=[None, 32, 32, 3],
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)


    coarse_network = conv_2d(network, 64, 3, activation='relu', name="unique_conv_1_coarse")
    coarse_network = conv_2d(coarse_network, 64, 3, activation='relu', name="unique_conv_2_coarse")
    coarse_network = max_pool_2d(coarse_network, 2)

    coarse_network = fully_connected(coarse_network, 512, activation='relu', name="unique_fc_1_coarse")
    coarse_hidden_reps = coarse_network
    coarse_network = dropout(coarse_network, 0.5)
    coarse_network = fully_connected(coarse_network, coarse_dim, activation='softmax', name="unique_fc_2_coarse")

    fine_network = conv_2d(network, 64, 3, activation='relu', name="unique_conv_1_fine")
    fine_network = conv_2d(fine_network, 64, 3, activation='relu', name="unique_conv_2_fine")
    fine_network = max_pool_2d(fine_network, 2)
    fine_network = fully_connected(fine_network, 512, activation='relu', name="unique_fc_1_fine")
    fine_hidden_reps = fine_network
    fine_network = dropout(fine_network, 0.5)
    fine_network = fully_connected(fine_network, fine_dim, activation='softmax', name="unique_fc_2_fine")

    if get_hidden_reps:
        return (coarse_hidden_reps, fine_hidden_reps)

    if get_fc_softmax_activations:
        # return the last layer of the coarse and the fine branch.
        # needed so we can write a custom function which takes the activations, computes the confidence scores and
        # decides at what level of the hierarchy to predict.
        return (coarse_network, fine_network)

    # coarse_confidence =
    # fine_confidence =

    stacked_coarse_and_fine_net = tf.concat(1, [coarse_network, fine_network])

    target_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, coarse_dim + fine_dim))

    def coarse_and_fine_joint_loss(incoming, placeholder):
        # coarse_dim = 20
        # fine_dim = 40
        # has access to coarse_dim and fine_dim
        coarse_pred = incoming[:, :coarse_dim]
        fine_pred = incoming[:, coarse_dim:]
        coarse_target = placeholder[:, :coarse_dim]
        fine_target = placeholder[:, coarse_dim:]

        coarse_loss = tflearn.categorical_crossentropy(coarse_pred, coarse_target)
        fine_loss = tflearn.categorical_crossentropy(fine_pred, fine_target)

        return coarse_loss + fine_loss

    def coarse_and_fine_accuracy(y_pred, y_true, x):
        coarse_pred = y_pred[:, :coarse_dim]
        fine_pred = y_pred[:, coarse_dim:]
        coarse_target = y_true[:, :coarse_dim]
        fine_target = y_true[:, coarse_dim:]

        coarse_acc = tflearn.metrics.accuracy_op(coarse_pred, coarse_target)
        fine_acc = tflearn.metrics.accuracy_op(fine_pred, fine_target)

        # rounded_coarse_acc = tf.to_float(tf.round(coarse_acc * 1000) * 100000)
        # return tf.add(rounded_coarse_acc, fine_acc)
        return fine_acc


    joint_network = regression(stacked_coarse_and_fine_net, placeholder=target_placeholder, optimizer='adam',
                             loss=coarse_and_fine_joint_loss,
                             metric=coarse_and_fine_accuracy,
                             learning_rate=0.001)
    return joint_network