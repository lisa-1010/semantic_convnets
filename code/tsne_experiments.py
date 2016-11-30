from pipeline import *
from utils import *
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import style
# import pandas as pd
from sklearn.decomposition import PCA



model_id='simple_cnn'
dataset='cifar10'

X, Y, X_test, Y_test = load_data(dataset, num_training=100, num_test=500)
n_classes = DATASET_TO_N_CLASSES[dataset]


hidden_reps = None
graph_to_use = tf.Graph()
with graph_to_use.as_default():
    print "loading model"
    hidden_rep_model = load_model(model_id, n_classes=n_classes, checkpoint_model_id=model_id, is_training=False, get_hidden_reps=True)
    hidden_reps = np.array(hidden_rep_model.predict(X_test))

print hidden_reps.shape

y_test = np.argmax(Y_test, axis=1)
pca = PCA(n_components=4)
pca_results = pca.fit_transform(hidden_reps)
tsne_model = TSNE(n_components=2, random_state=0)
tsne_results = tsne_model.fit_transform(pca_results)
dim_0 = np.reshape(tsne_results[:,0], tsne_results.shape[0])
dim_1 = np.reshape(tsne_results[:,1], tsne_results.shape[0])
print dim_0
print dim_1
plt.scatter(dim_0, dim_1)
plt.show()


