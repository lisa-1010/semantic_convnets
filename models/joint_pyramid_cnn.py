# -*- coding: utf-8 -*-
"""
A slightly modified tflearn example. Reference: https://github.com/tflearn/tflearn

Convolutional network applied to CIFAR-10 dataset classification task.
References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
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
def build_network(output_dims=[20, 100]):


    assert (len(output_dims) == 2), "output_dims needs to be of length 2, containing coarse_dim and fine_dim."
    # outputdims is a list of num_classes
    # Real-time data preprocessing

    coarse_dim, fine_dim = tuple(output_dims)

    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    network = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)


    coarse_network = conv_2d(network, 64, 3, activation='relu', name="unique/conv_1_coarse")
    coarse_network = conv_2d(coarse_network, 64, 3, activation='relu', name="unique/conv_2_coarse")
    coarse_network = max_pool_2d(coarse_network, 2)
    coarse_network = fully_connected(coarse_network, 512, activation='relu', name="unique/fc_1_coarse")
    coarse_network = dropout(coarse_network, 0.5)
    coarse_network = fully_connected(coarse_network, coarse_dim, activation='softmax', name="unique/fc_2_coarse")

    fine_network = conv_2d(network, 64, 3, activation='relu', name="unique/conv_1_fine")
    fine_network = conv_2d(fine_network, 64, 3, activation='relu', name="unique/conv_2_fine")
    fine_network = max_pool_2d(fine_network, 2)
    fine_network = fully_connected(fine_network, 512, activation='relu', name="unique/fc_1_fine")
    fine_network = dropout(fine_network, 0.5)
    fine_network = fully_connected(fine_network, fine_dim, activation='softmax', name="unique/fc_2_fine")

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