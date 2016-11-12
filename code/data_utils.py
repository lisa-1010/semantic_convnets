# data_utils.py
# @author: Lisa Wang
# @created:  2016
#
#===============================================================================
# DESCRIPTION: Code based on data_utils.py from loopy_nets project.
#===============================================================================
# CURRENT STATUS:
#===============================================================================
# USAGE: from data_utils import *
#

import os, struct
import numpy as np

from array import array as pyarray
from numpy import append, array, int8, uint8, zeros, int32, float32

import cPickle as pickle
from scipy.misc import imread


def change_to_array(M, H, W):
    N = len(M[0])
    X = np.array(M[0], dtype=float32).reshape((N,1,H,W))
    y = np.array(M[1], dtype=int32)
    return X, y



def load_cifar(num_training=49000, num_validation=1000, num_test=1000, dataset='cifar10'):
    """
    WARNING: Needs to be run from code directory, otherwise relative path
    will not work.
    Load the CIFAR-10 dataset from disk.
    Returns train, validation and test sets.
    Note that num_training, num_validation and num_test have to be > 0.
    Adapted from CS231N assignment 1.
    """
    # Load the raw CIFAR-10 data
    if dataset == 'cifar10':
        cifar10_dir = '../data/cifar-10-batches-py'
        X_train, y_train, X_test, y_test = _load_cifar10(cifar10_dir)
    elif dataset == 'cifar100':
        cifar100_dir = '../data/cifar-100-python'
        X_train, y_fine_train, y_coarse_train, X_test, y_fine_test, y_coarse_test = _load_cifar100(cifar100_dir)
        y_train = np.stack((y_fine_train, y_coarse_train)).swapaxes(0,1)
        y_test = np.stack((y_fine_test, y_coarse_test)).swapaxes(0,1)
    else:
        print "dataset not known. "
        return

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows

    X_train = X_train.transpose((0, 3, 1, 2))
    X_val = X_val.transpose((0, 3, 1, 2))
    X_test = X_test.transpose((0, 3, 1, 2))

    X_train = X_train.astype(np.int32)
    X_val = X_val.astype(np.int32)
    X_test = X_test.astype(np.int32)

    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)
    if dataset == 'cifar10':
        return X_train, y_train, X_val, y_val, X_test, y_test
    elif dataset == 'cifar100':
        return X_train, y_train, X_val, y_val, X_test, y_test



def _load_cifar100(ROOT):
    Xtr, Y_fine_tr, Y_coarse_tr = _load_cifar100_batch(os.path.join(ROOT, 'train'))
    Xte, Y_fine_te, Y_coarse_te  = _load_cifar100_batch(os.path.join(ROOT, 'test'))
    return Xtr, Y_fine_tr, Y_coarse_tr, Xte, Y_fine_te, Y_coarse_te


def _load_cifar100_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        batch_label = datadict['batch_label']
        print ("loading cifar batch {}".format(batch_label))
        X = datadict['data']
        Y_fine = datadict['fine_labels']
        Y_coarse = datadict['coarse_labels']
        num_samples = 0
        if batch_label == 'training batch 1 of 1':
            num_samples = 50000

        elif batch_label == 'testing batch 1 of 1':
            num_samples = 10000
        X = X.reshape(num_samples, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y_fine = np.array(Y_fine)
        Y_coarse = np.array(Y_coarse)
        return X, Y_fine, Y_coarse


def _load_cifar10(ROOT):
    """ load all of cifar, adapted from CS231N assignment 1 """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = _load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = _load_cifar10_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def _load_cifar10_batch(filename):
    """ load single batch of cifar, adapted from CS231N assignment 1"""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        batch_label = datadict['batch_label']
        print ("loading cifar batch {}".format(batch_label))
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y


if __name__=='__main__':
    # X_train, y_train, X_val, y_val, X_test, y_test = load_cifar(num_training=49000, num_validation=1, num_test=1000, dataset='cifar10')
    # print 'X_train shape: {}'.format(X_train.shape)
    # print 'X_val shape: {}'.format(X_val.shape)
    # print 'X_test shape: {}'.format(X_test.shape)

    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar(num_training=49000, num_validation=1, num_test=1000,
                                                                dataset='cifar100')
    print 'X_train shape: {}'.format(X_train.shape)
    print 'y_train shape: {}'.format(y_train.shape)
    print 'X_val shape: {}'.format(X_val.shape)
    print 'X_test shape: {}'.format(X_test.shape)

