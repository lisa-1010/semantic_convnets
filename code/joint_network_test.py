from pipeline import *
import tflearn
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np

import datetime

from utils import *
from data_utils import *

sys.path.append("../") # so we can import models.
from models import *



def _load_model(model_id, network, n_classes=10, load_checkpoint=False, is_training=False):
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

    if is_training:
        model = tflearn.DNN(network, tensorboard_verbose=2, tensorboard_dir=tensorboard_dir,
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


networks = simple_cnn.build_network([20, 100])

coarseModel = networks[0]
fineModel = networks[1]


model_id_coarse = "coarse_simple_cnn"
model_id_fine = "fine_simple_cnn"

coarse_model = _load_model(model_id_coarse, coarseModel, n_classes=20, load_checkpoint=False, is_training=True)
date_time_string = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
run_id = "{}_{}".format(model_id_coarse, date_time_string)
# Train using classifier


X_coarse, Y_coarse, X_test_coarse, Y_test_coarse = load_data(dataset="cifar100_coarse")
print (X_coarse.shape)
print (Y_coarse.shape)

# print (Y_coarse[:10])
X_fine, Y_fine, X_test_fine, Y_test_fine = load_data(dataset="cifar100_fine")
print (Y_fine.shape)
# print (Y_fine[:10])

coarse_model.fit(X_coarse, Y_coarse, n_epoch=50, shuffle=True, validation_set=(X_test_coarse, Y_test_coarse),
                 show_metric=True, batch_size=96, run_id=run_id, snapshot_step=100)