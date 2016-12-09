# -*- coding: utf-8 -*-

# feature_extractor.py
# @author: Ajay Sohmshetty
# @created: Dec 4 2016
#
#===============================================================================
# DESCRIPTION:
#
# Runs pretrained model as feature extractor.
# Can be run from the command line.
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE:
# Commandline:
# To get help:python feature_extractor.py -h
#
#===============================================================================

from __future__ import division, print_function, absolute_import

from model_utils import *

sys.path.append("../") # so we can import models.

#===============================================================================


def read_commandline_args():
    def usage():
        print("Usage: python feature_extractor.py -d <dataset> -c <ckpt_model_id>")
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ht:m:d:c:", ["help", "dataset", "ckpt_model_id"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print (str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    dataset, checkpoint_model_id = None, None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-d", "--dataset"):
            dataset = a
        elif o in ("-c", "--ckpt_model_id"):
            checkpoint_model_id = a
        else:
            assert False, "unhandled option"

    assert dataset is not None and checkpoint_model_id is not None

    return dataset, checkpoint_model_id


def main():
    dataset, checkpoint_model_id = read_commandline_args()

    print("Using checkpoint_model {} as feature extractor for pyramid image dataset {}".format(checkpoint_model_id, dataset))

    # Encodes X, and X_test using specified model

    # Define a new DNN that goes until the penultimate point (right before the final classification layer).
    X_train_joint, y_train_joint, X_train_gate, y_train_gate, fine_or_coarse_train_gate, \
    X_test, y_test, fine_or_coarse_test = load_data_pyramid(dataset=dataset, return_subset='all')

    # Test using classifier
    model = load_model(checkpoint_model_id, checkpoint_model_id=checkpoint_model_id, is_training=False, get_hidden_reps=True)

    X_train_joint = model.predict(X_train_joint)
    X_train_gate = model.predict(X_train_gate)
    X_test = model.predict(X_test)


    save_features(X_train_joint, y_train_joint, X_train_gate, y_train_gate, fine_or_coarse_train_gate, \
    X_test, y_test, fine_or_coarse_test, checkpoint_model_id, dataset)

    print("DONE.")


if __name__ == '__main__':
    main()