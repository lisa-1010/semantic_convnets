MEAN_PIXEL_CIFAR = 121.5

N_COARSE_CIFAR = 20
N_FINE_CIFAR = 100


DATASET_TO_N_CLASSES = {
    'cifar10' : 10,
    'cifar100_coarse': 20,
    'cifar100_fine': 100,
    'cifar100_joint_fine_only': 100
}


ALL_MODEL_DICTS = {
    'simple_cnn': {'network_type': 'simple_cnn', 'dataset': 'cifar10'},
    'simple_cnn_cifar100_joint': {'network_type': 'simple_cnn', 'dataset': None},  # TODO joint
    'simple_cnn_cifar100_coarse': {'network_type': 'simple_cnn', 'dataset': 'cifar100_coarse'},
    'simple_cnn_cifar100_fine': {'network_type': 'simple_cnn', 'dataset': 'cifar100_fine'},
    'simple_cnn_extended1_cifar100_fine': {'network_type': 'simple_cnn_extended1', 'dataset': 'cifar100_fine'},

    'lenet_cnn_cifar100_coarse': {'network_type': 'lenet_cnn', 'dataset': 'cifar100_coarse'},
    'vggnet_cnn_cifar100_coarse': {'network_type': 'vggnet_cnn', 'dataset': 'cifar100_coarse'},

    'lenet_cnn_cifar100_fine': {'network_type': 'lenet_cnn', 'dataset': 'cifar100_fine'},
    'vggnet_cnn_cifar100_fine': {'network_type': 'vggnet_cnn', 'dataset': 'cifar100_fine'},

    'pyramid_cifar100': {'network_type': 'pyramid', 'dataset': 'cifar100_joint'},

    # Prefeaturization models
    'simple_cnn_cifar100_fine_for_featurization': {'network_type': 'simple_cnn', 'dataset': 'cifar100_joint_fine_only'},
    'simple_cnn_extended_1_cifar100_fine_for_featurization': {'network_type': 'simple_cnn_extended_1', 'dataset': 'cifar100_joint_fine_only'},

    # CNN_RNN models
    'cnn_rnn_cifar100': {'network_type': 'cnn_rnn', 'dataset': 'cifar100_joint_prefeaturized'},
    'cnn_rnn_end_to_end_cifar100': {'network_type': 'cnn_rnn_end_to_end', 'dataset': 'cifar100_joint'}
}