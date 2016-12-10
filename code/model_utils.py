# -*- coding: utf-8 -*-

# model_utils.py
# @author: Lisa Wang
# @created: Nov 12 2016
#
#===============================================================================
# DESCRIPTION:
#
# Exports train and test functions for all network models and data sets.
# Can be run from the command line.
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE:
# from model_utils import *
#===============================================================================

from __future__ import division, print_function, absolute_import

import os, sys, getopt
import datetime
import pickle

import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from sklearn.metrics import accuracy_score
import numpy as np

from constants import *
from utils import *
from data_utils import *

sys.path.append("../") # so we can import models.
from models import *
#===============================================================================

def get_weights_to_preload_function(model_id, checkpoint_model_id, is_training):
    def variable_name_map_func(existing_var_op_name):
        if not is_training:  # In test, always preload ALL weights
            return existing_var_op_name
        if checkpoint_model_id == model_id:  # If we're preloading the same model, then obviously preload ALL weights!
            return existing_var_op_name

        # Otherwise, we're in training and we can simply
        print(existing_var_op_name)
        if 'unique' in existing_var_op_name:
            return None
        else:
            return existing_var_op_name

    return variable_name_map_func


def save_features(X_train_joint, y_train_joint, X_train_gate, y_train_gate, fine_or_coarse_train_gate, \
    X_test, y_test, fine_or_coarse_test, checkpoint_model_id, dataset):
    feature_set_storage_dir = "../data/feature_sets"
    check_if_path_exists_or_create("/" + feature_set_storage_dir)

    feature_set_pickle_name = "{}/{}_prefeaturized".format(feature_set_storage_dir, dataset)

    dataset_contents = [X_train_joint, y_train_joint, X_train_gate, y_train_gate, fine_or_coarse_train_gate, \
    X_test, y_test, fine_or_coarse_test]

    pickle.dump(dataset_contents, open(feature_set_pickle_name, "wb"))


def load_model(model_id, n_classes=10, pyramid_output_dims=None, is_training=False, checkpoint_model_id=None, get_hidden_reps=False):
    # should be used for all models

    assert (is_training ^ get_hidden_reps), "If you train, you can't get hidden reps and vice versa. "
    print ('Loading model...')

    model_dict = ALL_MODEL_DICTS[model_id]
    network_type = model_dict['network_type']

    tensorboard_dir = '../tensorboard_logs/' + model_id + '/'
    checkpoint_path = '../checkpoints/' + model_id + '/'
    best_checkpoint_path = '../best_checkpoints/' + model_id + '/'

    print (tensorboard_dir)
    print (checkpoint_path)
    print (best_checkpoint_path)

    check_if_path_exists_or_create(tensorboard_dir)
    check_if_path_exists_or_create(checkpoint_path)
    check_if_path_exists_or_create(best_checkpoint_path)

    network = load_network(network_type=network_type, n_classes=n_classes, pyramid_output_dims=pyramid_output_dims,
                           get_hidden_reps=get_hidden_reps)

    if is_training:
        model = tflearn.DNN(network, tensorboard_verbose=2, tensorboard_dir=tensorboard_dir,
                            checkpoint_path=checkpoint_path, best_checkpoint_path=best_checkpoint_path, max_checkpoints=3)
    else:
        model = tflearn.DNN(network)

    if checkpoint_model_id:
        start_checkpoint_path = '../checkpoints/' + checkpoint_model_id + '/'
        checkpoint = tf.train.latest_checkpoint(start_checkpoint_path)  # can be none of no checkpoint exists
        if checkpoint and os.path.isfile(checkpoint):
            variable_name_map_func = get_weights_to_preload_function(model_id, checkpoint_model_id, is_training)
            model.load(checkpoint, weights_only=True, verbose=True, variable_name_map=variable_name_map_func)
            print('Checkpoint loaded.')
        else:
            print('No checkpoint found. ')

    print ('Model loaded.')
    return model


def load_network(network_type='simple_cnn', n_classes=10, pyramid_output_dims=None, get_hidden_reps=False):
    network = None

    if network_type == 'simple_cnn':
        network = simple_cnn.build_network([n_classes], get_hidden_reps=get_hidden_reps)
    elif network_type == 'lenet_cnn':
        network = lenet_cnn.build_network([n_classes])
    elif network_type == 'lenet_small_cnn':
        network = lenet_small_cnn.build_network([n_classes])
    elif network_type == 'vggnet_cnn':
        network = vggnet_cnn.build_network([n_classes])
    elif network_type == 'simple_cnn_extended1':
        network = simple_cnn_extended1.build_network([n_classes], get_hidden_reps=get_hidden_reps)
    elif network_type == 'pyramid':
        assert (pyramid_output_dims != None), "If you try to load the pyramid model, you need to provide the " \
                                              "pyramid_output_dims, which is a list [coarse_dim, fine_dim]"
        network = joint_pyramid_cnn.build_network(pyramid_output_dims, get_hidden_reps=get_hidden_reps )
    elif network_type == "cnn_rnn":
        network = cnn_rnn.build_network(n_classes, get_hidden_reps=get_hidden_reps)
    else:
        print("Model {} not found. ".format(network_type))
        sys.exit()
    return network


def train_model(model_id='simple_cnn', dataset='cifar10', checkpoint_model_id=None):

    print ("Training model {} with dataset {}".format(model_id, dataset))

    n_classes = DATASET_TO_N_CLASSES[dataset]

    date_time_string = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_id = "{}_{}".format(model_id, date_time_string)
    # Train using classifier
    model = load_model(model_id=model_id, n_classes=n_classes, is_training=True, checkpoint_model_id=checkpoint_model_id)

    X, Y, X_test, Y_test = load_data(dataset)

    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=0.1,
              show_metric=True, batch_size=128, run_id=run_id, snapshot_step=100)


def train_pyramid_model(model_id='pyramid_cifar100', dataset='cifar100_joint',  checkpoint_model_id=None):
    coarse_dim = 20
    fine_dim = 100
    X_train_joint, y_train_joint = load_data_pyramid(dataset=dataset, return_subset='joint_only')

    X_train_joint, y_train_joint = shuffle(X_train_joint, y_train_joint)
    y_train_fine, y_train_coarse = y_train_joint[:, 0], y_train_joint[:, 1]
    y_train_fine, y_train_coarse = to_categorical(y_train_fine, fine_dim), to_categorical(y_train_coarse, coarse_dim)

    y_train_joint = np.concatenate((y_train_coarse, y_train_fine), axis=1)

    model = load_model(model_id, pyramid_output_dims=[coarse_dim, fine_dim], is_training=True, checkpoint_model_id=checkpoint_model_id)

    date_time_string = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_id = "{}_{}".format(model_id, date_time_string)

    model.fit(X_train_joint, y_train_joint, n_epoch=50, shuffle=True, validation_set=0.1,
              show_metric=True, batch_size=128, run_id=run_id, snapshot_step=100)


def form_y_for_cnn_rnn(y, fine_or_coarse_gate, coarse_dim, fine_dim):
    coarse_indexes, fine_indexes = y[:, 1] + fine_dim, y[:, 0]
    total_dim = coarse_dim + fine_dim + 1

    locations_of_end_token = fine_or_coarse_gate * -2 + 1  # 1 for not end token, -1 for end token
    fine_indexes = fine_indexes * locations_of_end_token  # any end token now has negative index
    fine_indexes[fine_indexes < 0] = total_dim - 1  # swap negative for end token

    coarse_one_hots = to_categorical(coarse_indexes, total_dim)
    fine_one_hots = to_categorical(fine_indexes, total_dim)

    y_joint = np.concatenate((coarse_one_hots, fine_one_hots), axis=1)

    return y_joint

def train_cnn_rnn_model(model_id='cnn_rnn_cifar100', dataset='cifar100_joint_prefeaturized',  checkpoint_model_id=None):
    coarse_dim = 20
    fine_dim = 100
    n_classes = coarse_dim + fine_dim + 1 # add 1 for the end token

    X_train_gate, y_train_gate, fine_or_coarse_train_gate = load_data_pyramid(dataset=dataset, return_subset="gate_only")
    X_test, y_test, fine_or_coarse_test = load_data_pyramid(dataset=dataset, return_subset="test_only")

    y_train_gate = form_y_for_cnn_rnn(y_train_gate, fine_or_coarse_train_gate, coarse_dim=coarse_dim, fine_dim=fine_dim)
    y_test = form_y_for_cnn_rnn(y_test, fine_or_coarse_test, coarse_dim=coarse_dim, fine_dim=fine_dim)

    model = load_model(model_id, n_classes=n_classes, is_training=True, checkpoint_model_id=checkpoint_model_id)

    date_time_string = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_id = "{}_{}".format(model_id, date_time_string)

    print("Shapes of X and Y:")
    print(np.asarray(X_train_gate).shape)
    print(y_train_gate.shape)
    print("Example: ")
    print(X_train_gate[0])
    print(y_train_gate[0])

    # Need to also tile this...
    #np.tile(b, 2)

    print("\n\n\nFitting these now...")
    model.fit(X_train_gate, y_train_gate, n_epoch=50, shuffle=True, validation_set=0.1,
              show_metric=True, batch_size=128, run_id=run_id, snapshot_step=100)


def test_model(model_id='simple_cnn', dataset='cifar10'):
    print("Testing model {} with dataset {}".format(model_id, dataset))

    X, Y, X_test, Y_test = load_data(dataset)
    n_classes = DATASET_TO_N_CLASSES[dataset]

    # Test using classifier
    model = load_model(model_id, n_classes=n_classes, checkpoint_model_id=model_id, is_training=False)
    # pred_train_probs = model.predict(X)
    # pred_train = np.argmax(pred_train_probs, axis=1)
    # train_acc = accuracy_score(pred_train, np.argmax(Y, axis=1))
    pred_test_probs = model.predict(X_test)
    pred_test = np.argmax(pred_test_probs, axis=1)
    test_acc = accuracy_score(pred_test, np.argmax(Y_test, axis=1))
    print("Test acc: {}".format( test_acc))