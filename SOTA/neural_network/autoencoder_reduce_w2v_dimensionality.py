__author__ = 'lqrz'

import cPickle
import numpy as np

from autoencoder import Autoencoder
from utils.utils import NeuralNetwork
from data import get_w2v_training_data_vectors

if __name__ == '__main__':

    print '...Getting activation vectors'

    original_vectors_dict = cPickle.load(open(get_w2v_training_data_vectors('googlenews_representations_train_True_valid_True_test_False.p'), 'rb'))
    original_vectors = np.array(original_vectors_dict.values())
    n_in = original_vectors.shape[1]
    n_hidden = 50

    params = {
        'n_in': n_in,
        'n_hidden': n_hidden,
        'hidden_function': NeuralNetwork.linear_activation_function,
        'x_train': original_vectors,
        'output_path': get_w2v_training_data_vectors,
        'regularization': True
    }

    autoencoder = Autoencoder(**params)

    hidden_activations = autoencoder.train(max_epochs=300)

    assert hidden_activations.shape[0] == original_vectors.shape[0]

    resized_vectors = dict(zip(original_vectors_dict.keys(), hidden_activations))

    cPickle.dump(resized_vectors, open(get_w2v_training_data_vectors('googlenews_representations_train_True_valid_True_test_False_50.p'),'wb'))

    print 'End.'
