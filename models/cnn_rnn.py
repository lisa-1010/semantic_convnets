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
import tflearn.helpers.summarizer as tf_summarizer


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
        raw_predictions = y_pred
        raw_predictions = tf.reshape(raw_predictions, (-1, 2, single_output_token_size))
        raw_predictions = tf.argmax(raw_predictions, 2)

        raw_corrects = y_true
        raw_corrects = tf.reshape(raw_corrects, (-1, 2, single_output_token_size))
        raw_corrects = tf.argmax(raw_corrects, 2)

        hierarchical_pred = simplifyToHierarchicalFormat(raw_predictions)
        hierarchical_target = simplifyToHierarchicalFormat(raw_corrects)

        hierarchical_corrects = tf.equal(hierarchical_pred, hierarchical_target)
        hierarchical_acc = tf.reduce_mean(tf.cast(hierarchical_corrects, tf.float32), name="Hierarchical_accuracy")

        return hierarchical_acc

        ##### To calculate the average accuracy, do:
        #####
        # coarse_pred = y_pred[:, :single_output_token_size]
        # fine_pred = y_pred[:, single_output_token_size:]
        # coarse_target = y_true[:, :single_output_token_size]
        # fine_target = y_true[:, single_output_token_size:]
        #
        # coarse_acc = tflearn.metrics.accuracy_op(coarse_pred, coarse_target)
        # fine_acc = tflearn.metrics.accuracy_op(fine_pred, fine_target)
        # return (fine_acc + coarse_acc) / 2.0
        #####
        #####

        ## LISA --- All this code isn't required to be here.... but after taking 6 hours to figure out how to make this
        ##          work... it felt wrong to delete all of this... lol

        # rounded_coarse_acc = tf.to_float(tf.round(coarse_acc * 1000) * 100000)
        # return tf.add(rounded_coarse_acc, fine_acc)
        #with tf.Graph().as_default():
        #import tflearn.helpers.summarizer as s
        #s.summarize(coarse_acc, 'scalar', "test_summary")
        #summaries.get_summary(type, name, value, summary_collection)
        #tflearn.summaries.get_summary("scalar", "coarse_acc", coarse_acc, "test_summary_collection")
        #tflearn.summaries.get_summary("scalar", "fine_acc", fine_acc, "test_summary_collection")
        #summaries.add_gradients_summary(grads, "", "", summary_collection)
        #sum1 = tf.scalar_summary("coarse_acc", coarse_acc)
        #tf.merge_summary([sum1])

        #tf.merge_summary(tf.get_collection("test_summary_collection"))
        #tflearn.summaries.get_summary("scalar", "coarse_acc", coarse_acc)
        #tflearn.summaries.get_summary("scalar", "fine_acc", fine_acc)
        #tf.scalar_summary("fine_acc", fine_acc)

        #tf_summarizer.summarize(coarse_acc, "scalar", "Coarse_accuracy")
        #tf_summarizer.summarize(fine_acc, "scalar", "Fine_accuracy")


    #test_const = tf.constant(32.0, name="custom_constant")
    #sum1 = tf.scalar_summary("dumb_contant", test_const)
    #tf.merge_summary([sum1])

    # Class predictions are in the form [[A, B], ...], where A=Coarse, B=Fine
    def simplifyToHierarchicalFormat(class_labels, end_token=120):
        fine_labels = class_labels[:, 1]
        coarse_labels = class_labels[:, 0]

        coarse_labels_mask = tf.cast(tf.equal(fine_labels, end_token), tf.int64)
        coarse_labels_to_permit = coarse_labels * coarse_labels_mask

        fine_labels_mask = tf.cast(tf.not_equal(fine_labels, end_token), tf.int64)
        fine_labels_to_permit = fine_labels * fine_labels_mask

        hierarchical_labels = coarse_labels_to_permit + fine_labels_to_permit

        return hierarchical_labels

    with tf.name_scope('Accuracy'):
        with tf.name_scope('Coarse_Accuracy'):
            coarse_pred = coarse_network
            coarse_target = target_placeholder[:, :single_output_token_size]
            correct_coarse_pred = tf.equal(tf.argmax(coarse_pred, 1), tf.argmax(coarse_target, 1))
            coarse_acc_value = tf.reduce_mean(tf.cast(correct_coarse_pred, tf.float32))

        with tf.name_scope('Fine_Accuracy'):
            fine_pred = fine_network
            fine_target = target_placeholder[:, single_output_token_size:]
            correct_fine_pred = tf.equal(tf.argmax(fine_pred, 1), tf.argmax(fine_target, 1))
            fine_acc_value = tf.reduce_mean(tf.cast(correct_fine_pred, tf.float32))

    with tf.name_scope('Combination_Accuracies'):
        with tf.name_scope('Both_Correct_Accuracy'):
            both_correct_acc_value = tflearn.metrics.accuracy_op(stacked_coarse_and_fine_net, target_placeholder)
        with tf.name_scope('Average_Accuracy'):
            avg_acc_value = (coarse_acc_value+fine_acc_value) / 2
        with tf.name_scope('Hierarchical_Accuracy'):
            raw_predictions = stacked_coarse_and_fine_net
            raw_predictions = tf.reshape(raw_predictions, (-1, 2, single_output_token_size))
            raw_predictions = tf.argmax(raw_predictions, 2)

            raw_corrects = target_placeholder
            raw_corrects = tf.reshape(raw_corrects, (-1, 2, single_output_token_size))
            raw_corrects = tf.argmax(raw_corrects, 2)

            hierarchical_pred = simplifyToHierarchicalFormat(raw_predictions)
            hierarchical_target = simplifyToHierarchicalFormat(raw_corrects)

            hierarchical_corrects = tf.equal(hierarchical_pred, hierarchical_target)
            hierarchical_acc = tf.reduce_mean(tf.cast(hierarchical_corrects, tf.float32))
            # Reduce the prediction and the actual to ONE dimension


    net = regression(stacked_coarse_and_fine_net, placeholder=target_placeholder, optimizer='adam',
                             loss=coarse_and_fine_joint_loss,
                             metric=coarse_and_fine_accuracy,
                             validation_monitors=[coarse_acc_value, fine_acc_value, both_correct_acc_value, avg_acc_value, hierarchical_acc],
                             learning_rate=0.0001)


    return net