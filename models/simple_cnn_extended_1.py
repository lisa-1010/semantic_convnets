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


# Convolutional network building
def build_network(output_dims=None, get_hidden_reps=False, detatch_final_layer=False):
    # outputdims is a list of num_classes
    # Real-time data preprocessing


    # img_prep = ImagePreprocessing()
    # img_prep.add_featurewise_zero_center()
    # img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    network = input_data(shape=[None, 32, 32, 3],
                         # data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 64, 3, activation='relu', name="unique_Conv2D_3")
    network = conv_2d(network, 64, 3, activation='relu', name="unique_Conv2D_4")
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 512, activation='relu', name="unique_FullyConnected")
    network = fully_connected(network, 512, activation='relu', name="unique_FullyConnected_1")
    if get_hidden_reps:
        return network
    network = dropout(network, 0.5)

    networks = []
    for output_dim in output_dims:
        cur_network = fully_connected(network, output_dim, activation='softmax', name="unique_FullyConnected_output_dim_{}".format(output_dim))
        cur_network = regression(cur_network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.00005)
        networks.append(cur_network)

    if len(networks) == 1:
        return networks[0]
    return networks