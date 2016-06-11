__author__ = 'root'

import theano
import theano.tensor as T
from utils import utils
from collections import OrderedDict
import numpy as np
import cPickle

from trained_models import get_cwnn_path
from utils.utils import NeuralNetwork

theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.warn_float64='ignore'
theano.config.floatX='float64'

np.random.seed(1234)

class Autoencoder(object):

    def __init__(self, n_in, n_hidden, hidden_function, x_train, output_path, regularization):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = self.n_in

        self.hidden_function = hidden_function

        self.x_train = np.array(x_train).astype(dtype=theano.config.floatX)

        self.output_path = output_path

        self.regularization = regularization

        self.params = OrderedDict()

    def train(self, learning_rate=0.01, batch_size=512, max_epochs=20,
              save_matrix=True, save_hidden=True, **kwargs):

        w1 = theano.shared(value=utils.NeuralNetwork.initialize_weights(self.n_in, self.n_hidden, function='linear').astype(dtype=theano.config.floatX), name='w1', borrow=True)
        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(self.n_hidden, self.n_out, function='linear').astype(dtype=theano.config.floatX), name='w2', borrow=True)
        b1 = theano.shared(value=np.zeros(self.n_hidden), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out), name='b2', borrow=True)

        param_names = ['w1', 'w2']
        params = [w1, w2]
        self.params.update(zip(param_names, params))

        train_x = theano.shared(value=np.array(self.x_train).astype(dtype=theano.config.floatX),
                                name='train_x')

        h = self.hidden_function(T.dot(train_x, w1))
        out = T.dot(h, w2)

        L2_w1 = T.sum(w1 ** 2)
        L2_w2 = T.sum(w2 ** 2)
        L2 = L2_w1 + L2_w2

        if self.regularization:
            error = T.sum(T.sqr(out-train_x)) + L2
        else:
            error = T.sum(T.sqr(out-train_x))

        grads = [T.grad(cost=error, wrt=param) for param in self.params.values()]

        # adagrad
        accumulated_grads = []
        for name, param in zip(param_names, params):
            eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grads.append(theano.shared(value=eps, borrow=True))

        updates = []
        for param, grad, accum_grad in zip(params, grads, accumulated_grads):
            accum = accum_grad + T.sqr(grad)
            updates.append((param, param - learning_rate * grad / (T.sqrt(accum) + 10 ** -5)))
            updates.append((accum_grad, accum))

        train = theano.function(inputs=[],
                                outputs=error,
                                updates=updates)

        compute_activations = theano.function(inputs=[],
                                outputs=h,
                                updates=[])

        compute_weights = theano.function(inputs=[],
                                          outputs=[L2_w1, L2_w2])

        for epoch in range(max_epochs):
            epoch_cost = train()
            l2_w1, l2_w2 = compute_weights()

            print 'Epoch %d Cost: %f W1_sum: %f W2_sum:%f' % (epoch, epoch_cost, l2_w1, l2_w2)

        if save_matrix:
            print '...Saving autoencoded weight matrix'
            cPickle.dump(w1.get_value(), open(self.output_path('autoencoded_w1.p'), 'wb'))

        h = compute_activations()

        return h


if __name__ == '__main__':

    print '...Getting activation vectors'
    activations = cPickle.load(open(get_cwnn_path('hidden_activations.p'), 'rb'))

    n_in = activations.shape[1]
    n_hidden = 39

    params = {
        'n_in': n_in,
        'n_hidden': n_hidden,
        'hidden_function': NeuralNetwork.linear_activation_function,
        'x_train': activations,
        'output_path': get_cwnn_path
    }

    autoencoder = Autoencoder(**params)

    autoencoder.train(max_epochs=50)



