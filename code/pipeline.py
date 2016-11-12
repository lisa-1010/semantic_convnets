# -*- coding: utf-8 -*-

# pipeline.py
# @author: Lisa Wang
# @created: Nov 12 2016
#
#===============================================================================
# DESCRIPTION:
#
# Exports
#
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE:
# In another python script:
# from pipeline import *
#
# Commandline:
# python pipeline.py -m <model_id> -d <dataset>
# or to get help:python pipeline.py -h
#
#===============================================================================

from __future__ import division, print_function, absolute_import

import os, sys, getopt

import tflearn
from tflearn.data_utils import shuffle, to_categorical

from data_utils import *

sys.path.append("../") # so we can import models.
from models import *
#===============================================================================


def load_network(model_id='simple_cnn', load_checkpoint=False):
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


def train_model(model_id='simple_cnn', dataset='cifar10'):

    print ("Training model {} with dataset {}".format(model_id, dataset))

    network = load_network(model_id=model_id)
    X, Y, X_test, Y_test = load_data(dataset)

    # Train using classifier
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
              show_metric=True, batch_size=96, run_id='cifar10_cnn')


def test
def read_commandline_args():
    def usage():
        print("Usage: python pipeline.py -m <model_id> -d <dataset>")
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hm:d:", ["help", "model_id", "dataset"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print (str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    model_id, dataset = None, None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-m", "--model_id"):
            model_id = a
        elif o in ("-d", "--dataset"):
            dataset = a
        else:
            assert False, "unhandled option"
    if model_id == None:
        model_id = 'simple_cnn'

    if dataset == None:
        dataset = 'cifar10'
    return model_id, dataset


def main():
    model_id, dataset = read_commandline_args()
    train_model(model_id, dataset)


if __name__ == '__main__':
    main()