# -*- coding: utf-8 -*-
# vis.py
# @author: Lisa Wang
# @created: Nov 29 2016
#===============================================================================
# DESCRIPTION:
# Exports functions to visualize network embeddings
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE:
# from vis import *
# # To see how this module is used, please refer to examples in
# tsne_experiments.ipynb notebook.
#===============================================================================

from model_utils import *
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# from matplotlib import style
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def visualize_embeddings_with_tsne(model_id, X, Y, label_names):
    # Y has to be dense (not one-hot)
    n_classes = len(label_names)
    hidden_reps = None
    graph_to_use = tf.Graph()
    with graph_to_use.as_default():
        hidden_rep_model = load_model(model_id, n_classes=n_classes, checkpoint_model_id=model_id, is_training=False, get_hidden_reps=True)
        hidden_reps = np.array(hidden_rep_model.predict(X))

    y = np.array(np.argmax(Y, axis=1),dtype="int")

    pca = PCA(n_components=5)
    pca_results = pca.fit_transform(hidden_reps)
    tsne_model = TSNE(n_components=2, perplexity=35, random_state=0)
    tsne_results = tsne_model.fit_transform(pca_results)
    dim_0 = np.reshape(tsne_results[:,0], tsne_results.shape[0])
    dim_1 = np.reshape(tsne_results[:,1], tsne_results.shape[0])

    sns.set_palette("Set2", 10)
    colors = sns.color_palette("cubehelix",10)

    plt.figure(figsize=(6,6))
    for label in xrange(n_classes):
        points_0, points_1 = [], []
        for i in xrange(len(dim_0)):
            if y[i] == label:
                points_0.append(dim_0[i])
                points_1.append(dim_1[i])

        plt.scatter(points_0, points_1, c=colors[label], label=label_names[label])
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.show()
