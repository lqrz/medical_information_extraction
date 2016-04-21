__author__ = 'root'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/SOTA')

import theano
import theano.tensor as T
import lasagne
import numpy as np
import time

from SOTA.neural_network.hidden_Layer_Context_Window_Net import Hidden_Layer_Context_Window_Net
from SOTA.neural_network.train_neural_network import load_w2v_model_and_vectors_cache
from utils.utils import NeuralNetwork

if __name__ == '__main__':
    crf_training_data_filename = 'handoverdata.zip'
    test_data_filename = 'handover-set2.zip'

    add_tags = ['<PAD>']
    n_window = 5
    max_epochs = 50

    x_train, y_train, x_test, y_test, word2index, \
    index2word, label2index, index2label = \
        Hidden_Layer_Context_Window_Net.get_data(crf_training_data_filename, test_data_filename,
                                                add_tags, x_idx=None, n_window=n_window)

    args = {
        'w2v_vectors_cache': 'w2v_googlenews_representations.p',
    }
    w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache(args)

    w = Hidden_Layer_Context_Window_Net.initialize_w(w2v_dims, word2index.keys(),
                                                 w2v_vectors=w2v_vectors, w2v_model=w2v_model)
    w = w.astype(dtype=theano.config.floatX)

    n_words, n_dims = w.shape
    n_out = label2index.keys().__len__()

    assert w2v_dims == n_dims, 'Error in embeddings dimensions computation.'

    x = T.imatrix()
    l_in = lasagne.layers.InputLayer((n_window, 1))

    l1 = lasagne.layers.EmbeddingLayer(l_in, input_size=n_words, output_size=n_dims, W=w)

    # x is the input layer (i only have one). l1 is the output expression.
    # l1_indexed = lasagne.layers.get_output(l1, x)
    l1_reshaped = lasagne.layers.ReshapeLayer(l1, shape=(1,n_window*n_dims))

    b1 = lasagne.layers.BiasLayer(l1_reshaped, b=np.zeros(shape=(n_window*n_dims,), dtype=theano.config.floatX))

    h = lasagne.layers.NonlinearityLayer(b1, nonlinearity=lasagne.nonlinearities.tanh)

    # h_output = lasagne.layers.get_output(h, x)
    # test_hidden_layer = theano.function([x], h_output)
    # test_hidden_layer(np.matrix(x_train[0]).astype(dtype='int32'))

    l2 = lasagne.layers.DenseLayer(h, num_units=n_out,
                                   W=NeuralNetwork.initialize_weights(n_window*n_dims,n_out,'softmax').astype(
                                       dtype=theano.config.floatX
                                   ),
                                   b=np.zeros(shape=(n_out,), dtype=theano.config.floatX),
                                   nonlinearity=lasagne.nonlinearities.softmax)

    # test embedding reshape.
    l2_output = lasagne.layers.get_output(l2, x)
    # test_out_layer = theano.function([x], l2_output)
    # test_out_layer(np.matrix(x_train[0]).astype(dtype='int32'))

    y = T.ivector()

    cost = T.mean(lasagne.objectives.categorical_crossentropy(l2_output, y))
    l2_penalty = lasagne.regularization.l2(l1.W) + lasagne.regularization.l2(l2.W)
    cost = cost + l2_penalty * 0.01

    pred = T.argmax(l2_output, axis=1)
    errors = T.neq(pred, y)

    test = theano.function(inputs=[x,y], outputs=[pred,errors])
    test(np.matrix(x_train[0]).astype('int32'), [y_train[0]])

    params = lasagne.layers.get_all_params(l2, trainable=True)
    updates = lasagne.updates.adagrad(cost,params,learning_rate=0.01,epsilon=10**-5)

    # test = theano.function(inputs=[x,y], outputs=[updates])
    # test(np.matrix(x_train[0]).astype('int32'), [y_train[0]])

    train_fn = theano.function(inputs=[x, y], outputs=[cost,errors], updates=updates)

    for epoch in range(max_epochs):
        train_cost = 0
        train_error = 0
        start = time.time()

        for x_sample, y_sample in zip(x_train,y_train):
            cost, error = train_fn(np.matrix(x_sample).astype('int32'), [y_sample])
            if np.asscalar(error) != 0 and np.asscalar(error) != 1:
                print 'whut'
            train_cost += np.asscalar(cost)
            train_error += np.asscalar(error)
        end = time.time()
        print 'Epoch: %d Train_cost: %f Train_errors: %d Took: %f' % (epoch, train_cost, train_error, end-start)
    # lasagne.utils.one_hot