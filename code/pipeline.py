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
import datetime

from model_utils import *

sys.path.append("../") # so we can import models.

#===============================================================================


def read_commandline_args():
    def usage():
        print("Usage: python pipeline.py -t <train_or_test_mode> -m <model_id> -c <ckpt_model_id>")
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ht:m:d:c:", ["help", "train_or_test_mode", "model_id", "ckpt_model_id"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print (str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    mode, model_id, checkpoint_model_id = None, None, None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-t", "--train_or_test_mode"):
            mode = a
        elif o in ("-m", "--model_id"):
            model_id = a
        elif o in ("-c", "--ckpt_model_id"):
            checkpoint_model_id = a
        else:
            assert False, "unhandled option"

    if mode == None:
        mode = 'train'
        print ("Using default mode: train. To test, please specify with commandline arg '-t test ")

    if model_id == None:
        model_id = 'simple_cnn'

    return mode, model_id, checkpoint_model_id


def main():
    mode, model_id, checkpoint_model_id = read_commandline_args()

    dataset = ALL_MODEL_DICTS[model_id]["dataset"]
    if mode == 'train':
        if model_id == 'pyramid_cifar100':
            train_pyramid_model(model_id, dataset, checkpoint_model_id=checkpoint_model_id)
        elif model_id == 'cnn_rnn_cifar100':
            train_cnn_rnn_model(model_id, dataset, checkpoint_model_id=checkpoint_model_id)
        else:
            train_model(model_id, dataset, checkpoint_model_id=checkpoint_model_id)
    elif mode == 'test':
        test_model(model_id, dataset)


if __name__ == '__main__':
    main()