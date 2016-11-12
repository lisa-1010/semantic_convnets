# -*- coding: utf-8 -*-

# pipeline.py
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
# In another python script:
# from pipeline import *
#
# Commandline:
# python pipeline.py -t <train_or_test_mode> -m <model_id> -d <dataset>
# or to get help:python pipeline.py -h
#
#===============================================================================

from __future__ import division, print_function, absolute_import

import os, sys, getopt

import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from sklearn.metrics import accuracy_score

import numpy as np
from utils import *
from data_utils import *

sys.path.append("../") # so we can import models.
from models import *
#===============================================================================


def load_model(model_id, load_checkpoint=False, is_training=False):
    # should be used for all models
    print ('Loading model...')

    tensorboard_dir = '../tensorboard_logs/' + model_id + '/'
    checkpoint_path = '../checkpoints/' + model_id + '/'
    best_checkpoint_path = '../best_checkpoints/' + model_id + '/'

    print (tensorboard_dir)
    print (checkpoint_path)
    print (best_checkpoint_path)

    check_if_path_exists_or_create(tensorboard_dir)
    check_if_path_exists_or_create(checkpoint_path)
    check_if_path_exists_or_create(best_checkpoint_path)

    network = load_network(model_id=model_id)

    if is_training:
        model = tflearn.DNN(network, tensorboard_verbose=2, tensorboard_dir=tensorboard_dir, \
                            checkpoint_path=checkpoint_path, best_checkpoint_path=best_checkpoint_path, max_checkpoints=3)
    else:
        model = tflearn.DNN(network)

    if load_checkpoint:
        checkpoint = tf.train.latest_checkpoint(checkpoint_path)  # can be none of no checkpoint exists
        if checkpoint and os.path.isfile(checkpoint):
            # model.load(checkpoint, weights_only=True, verbose=True)
            model.load(checkpoint, verbose=True)
            print ('Checkpoint loaded.')
        else:
            print ('No checkpoint found. ')

    print ('Model loaded.')
    return model


def load_network(model_id='simple_cnn'):
    network = None
    if model_id == 'simple_cnn':
        network = simple_cnn.build_network()
    else:
        print("Model {} not found. ".format(model_id))
        sys.exit()
    return network


def load_data(dataset='cifar10'):
    X, Y, X_test, Y_test = None, None, None, None
    if dataset == 'cifar10':
        X, Y, X_val, Y_val, X_test, Y_test = load_cifar(num_training=50000, num_validation=0, num_test=10000,
                                                    dataset='cifar10')
    else:
        print ("Dataset {} not found. ".format(dataset))
        sys.exit()

    X, Y = shuffle(X, Y)
    Y = to_categorical(Y, 10)
    Y_test = to_categorical(Y_test, 10)
    return X, Y, X_test, Y_test


def train_model(model_id='simple_cnn', dataset='cifar10', load_checkpoint=False):

    print ("Training model {} with dataset {}".format(model_id, dataset))

    X, Y, X_test, Y_test = load_data(dataset)

    # Train using classifier
    model = load_model(model_id=model_id, load_checkpoint=load_checkpoint, is_training=True)
    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
              show_metric=True, batch_size=96, run_id='cifar10_cnn', snapshot_step=100)


def test_model(model_id='simple_cnn', dataset='cifar10'):
    print("Testing model {} with dataset {}".format(model_id, dataset))

    X, Y, X_test, Y_test = load_data(dataset)

    # Train using classifier
    model = load_model(model_id, load_checkpoint=True, is_training=False)
    pred_train_probs = model.predict(X)
    pred_train = np.argmax(pred_train_probs, axis=1)
    pred_test_probs = model.predict(X_test)
    pred_test = np.argmax(pred_test_probs, axis=1)
    train_acc = accuracy_score(pred_train, np.argmax(y_train, axis=1))
    test_acc = accuracy_score(pred_test, np.argmax(y_test, axis=1))
    print("Train acc: {}\t Test acc: {}".format(train_acc, test_acc))


def read_commandline_args():
    def usage():
        print("Usage: python pipeline.py -t <train_or_test_mode> -m <model_id> -d <dataset>")
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ht:m:d:", ["help", "train_or_test_mode", "model_id", "dataset"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print (str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    mode, model_id, dataset = None, None, None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-t", "--train_or_test_mode"):
            mode = a
        elif o in ("-m", "--model_id"):
            model_id = a
        elif o in ("-d", "--dataset"):
            dataset = a
        else:
            assert False, "unhandled option"

    if mode == None:
        mode = 'train'
        print ("Using default mode: train. To test, please specify with commandline arg '-t test ")

    if model_id == None:
        model_id = 'simple_cnn'

    if dataset == None:
        dataset = 'cifar10'
    return mode, model_id, dataset


def main():
    mode, model_id, dataset = read_commandline_args()
    if mode == 'train':
        train_model(model_id, dataset, load_checkpoint=True)
    elif mode == 'test':
        test_model(model_id, dataset)


if __name__ == '__main__':
    main()