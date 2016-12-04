from __future__ import division, print_function, absolute_import

import os, sys, getopt
import datetime

import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from sklearn.metrics import accuracy_score
import numpy as np

from utils import *
from model_utils import *
from data_utils import *

sys.path.append("../") # so we can import models.
from models import *

class PyramidWrapper(object):
    def __init__(self, checkpoint_model_id):
        coarse_net, fine_net = joint_pyramid_cnn.build_network(output_dims=[20, 100],  get_fc_softmax_activations=True)
        self.checkpoint_model_id = checkpoint_model_id
        self.coarse_model = tflearn.DNN(coarse_net)
        self.fine_model = tflearn.DNN(fine_net)
        print ("models loaded")
        self.load_checkpoint()

    def load_checkpoint(self):
        start_checkpoint_path = '../checkpoints/' + self.checkpoint_model_id + '/'
        checkpoint = tf.train.latest_checkpoint(start_checkpoint_path)  # can be none of no checkpoint exists
        if checkpoint and os.path.isfile(checkpoint):
            variable_name_map_func = get_weights_to_preload_function(model_id=None,
                                                                     checkpoint_model_id=self.checkpoint_model_id,
                                                                     is_training=False)
            self.coarse_model.load(checkpoint, weights_only=True, verbose=True, variable_name_map=variable_name_map_func)
            self.fine_model.load(checkpoint, weights_only=True, verbose=True, variable_name_map=variable_name_map_func)
            print('Checkpoint loaded.')
        else:
            print('No checkpoint found. ')


    def predict_both_fine_and_coarse(self, X):
        coarse_pred_probs = self.coarse_model.predict(X)
        fine_pred_probs = self.fine_model.predict(X)
        coarse_pred_classes = np.argmax(coarse_pred_probs, axis=1)
        fine_pred_classes = np.argmax(fine_pred_probs, axis=1)
        return fine_pred_classes, coarse_pred_classes


    def predict_fine_or_coarse(self, X, confid_threshold=0.5):
        n_samples = X.shape[0]
        fine_pred_classes, coarse_pred_classes = self.predict_both_fine_and_coarse(X)
        fine_confidence_score = np.ones(n_samples) # TODO: actually compute the score for each sample given the prediction probs.

        final_pred_classes = [] # list for final predictions, each prediction is EITHER coarse OR fine.
        for i in xrange(n_samples):
            if fine_confidence_score[i] > confid_threshold:
                final_pred_classes.append(fine_pred_classes[i])
            else:
                final_pred_classes.append(coarse_pred_classes[i])
        return np.array(final_pred_classes)


def compute_accuracy_predict_fine_or_coarse(final_pred_classes, Y_fine_coarse, fine_or_coarse):
    n_samples = final_pred_classes.shape[0]
    true_classes = []
    for i in xrange(n_samples):
        # true_classes.append(Y_fine_coarse[i, fine_or_coarse[i]])
        if fine_or_coarse[i] == 0:  # should predict fine
            true_classes.append(Y_fine_coarse[i, 0])
        else:  # shoud predict coarse
            true_classes.append(Y_fine_coarse[i, 1])
    acc = accuracy_score(np.argmax(true_classes, axis=1), final_pred_classes)
    return acc


def evaluate_predictions(model, X, Y, fine_or_coarse, confid_threshold=0.5):
    # expects model to be an instance of PyramidWrapper
    final_pred_classes = model.predict_fine_or_coarse(X, confid_threshold=confid_threshold)
    fine_or_coarse_acc = compute_accuracy_predict_fine_or_coarse(final_pred_classes, Y, fine_or_coarse)
    print("Accuracy for predict coarse OR fine: {}".format(fine_or_coarse_acc))

    fine_pred_classes, coarse_pred_classes = model.predict_both_fine_and_coarse(X)

    Y_fine, Y_coarse = Y[:,0], Y[:,1]
    fine_acc = accuracy_score(np.argmax(Y_fine), fine_pred_classes)
    coarse_acc = accuracy_score(np.argmax(Y_coarse), coarse_pred_classes)

    print("Accuracy for coarse predictions: {}".format(coarse_acc))
    print("Accuracy for fine predictions: {}".format(fine_acc))


if __name__ == "__main__":
    pyramid_model = PyramidWrapper(checkpoint_model_id="pyramid_cifar100")
    X, Y, fine_or_coarse = load_data_pyramid(return_subset='test_only')
    evaluate_predictions(pyramid_model, X, Y, fine_or_coarse, confid_threshold=0.5)
