__author__ = 'root'

import theano
import theano.tensor as T
from utils import utils
from collections import OrderedDict
import numpy as np

theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.warn_float64='ignore'
theano.config.floatX='float64'

class Autoencoder(object):

    def __init__(self, n_in, n_hidden):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = self.n_in

        self.params = OrderedDict()

    def train(self, learning_rate, batch_size=512, max_epochs=20):

        w1 = theano.shared(value=utils.NeuralNetwork.initialize_weights(self.n_in, self.n_hidden, function='linear').astype(dtype=theano.config.floatX), name='w1', borrow=True)
        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(self.n_hidden, self.n_out, function='linear').astype(dtype=theano.config.floatX), name='w2', borrow=True)
        b1 = theano.shared(value=np.zeros(self.n_hidden), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out), name='b2', borrow=True)


if __name__=='__main__':
    n_in = 20
    n_hidden = 10
