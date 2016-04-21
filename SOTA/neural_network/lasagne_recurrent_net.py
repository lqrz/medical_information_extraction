__author__ = 'root'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import theano
import theano.tensor as T
import lasagne
import numpy as np
import time
from itertools import chain
import cPickle

from SOTA.neural_network.recurrent_net import Recurrent_net
from SOTA.neural_network.hidden_Layer_Context_Window_Net import Hidden_Layer_Context_Window_Net
from SOTA.neural_network.train_neural_network import load_w2v_model_and_vectors_cache
from utils.metrics import Metrics

np.random.seed(1234)

if __name__ == '__main__':
    crf_training_data_filename = 'handoverdata.zip'
    test_data_filename = 'handover-set2.zip'

    # add_tags = ['<PAD>']
    add_tags = []
    n_window = 1
    max_epochs = 50
    bidirectional = True
    train_in_hid_weight = False

    if n_window == 1:
        x_train, y_train, x_test, y_test, word2index, \
        index2word, label2index, index2label = \
            Recurrent_net.get_data(crf_training_data_filename, test_data_filename,
                                                    add_tags, x_idx=None, n_window=n_window)
    elif n_window > 1:
        x_train, y_train, x_test, y_test, word2index, \
        index2word, label2index, index2label = \
            Hidden_Layer_Context_Window_Net.get_data(crf_training_data_filename, test_data_filename,
                                                    add_tags, x_idx=None, n_window=n_window)

    args = {
        'w2v_vectors_cache': 'w2v_googlenews_representations.p',
    }
    w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache(args)

    w = Recurrent_net.initialize_w(w2v_dims, word2index.keys(),
                                                 w2v_vectors=w2v_vectors, w2v_model=w2v_model)
    w = w.astype(dtype=theano.config.floatX)

    n_words, n_dims = w.shape
    n_out = label2index.keys().__len__()

    assert w2v_dims == n_dims, 'Error in embeddings dimensions computation.'

    x = T.imatrix()

    l_idxs = lasagne.layers.InputLayer(shape=(1,None), input_var=x, name='idxs')

    l1 = lasagne.layers.EmbeddingLayer(l_idxs, input_size=n_words, output_size=n_dims, W=w, name='embeddings_layer')

    # l1_output = lasagne.layers.get_output(l1)
    # test = theano.function([x], l1_output)
    # test(x_train[0])

    l_forward = lasagne.layers.RecurrentLayer(
        l1, n_dims, mask_input=None, grad_clipping=0,
        # W_in_to_hid=lasagne.init.HeUniform(),
        W_in_to_hid=np.identity(n=n_dims, dtype=theano.config.floatX),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh,
        name='forward_hidden_layer')

    l_forward_slice = lasagne.layers.SliceLayer(l_forward, indices=-1, axis=0)

    if bidirectional:
        #TODO: share weights?
        l_backward = lasagne.layers.RecurrentLayer(
            l1, n_dims, mask_input=None, grad_clipping=0,
            # W_in_to_hid=lasagne.init.HeUniform(),
            W_in_to_hid=np.identity(n=n_dims, dtype=theano.config.floatX),
            W_hid_to_hid=lasagne.init.HeUniform(),
            # W_in_to_hid=l_forward.W_in_to_hid,
            # W_hid_to_hid=l_forward.W_hid_to_hid,
            nonlinearity=lasagne.nonlinearities.tanh,
            backwards=True,
            name='backwards_hidden_layer')

    # l_f_out = lasagne.layers.get_output(l_forward)
    # test = theano.function([x], l_f_out)
    # out_forw = test(np.matrix(x_train[0]).astype('int32'))

    # For the backwards layer, the first index actually corresponds to the
    # final output of the network, as it processes the sequence backwards.
        l_backward_slice = lasagne.layers.SliceLayer(l_backward, indices=-1, axis=0)

    # flip the order of the backward slice, so the concatenation matches.
        l_backward_slice_inv = lasagne.layers.SliceLayer(l_backward_slice, indices=slice(None,None,-1),axis=0)

    # l_f_out = lasagne.layers.get_output(l_backward_slice)
    # test = theano.function([x], l_f_out)
    # out_back = test(np.matrix(x_train[0]).astype('int32'))
    # l_f_out = lasagne.layers.get_output(l_backward_slice_inv)
    # test = theano.function([x], l_f_out)
    # out_back_inv = test(np.matrix(x_train[0]).astype('int32'))

    # Now, we'll concatenate the outputs to combine them.
        l_sum = lasagne.layers.ConcatLayer([l_forward_slice, l_backward_slice], axis=1)

        l_out = lasagne.layers.DenseLayer(
            l_sum, num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax,
            name='output_layer'
        )

    # l_f_out = lasagne.layers.get_output(l_sum)
    # test = theano.function([x], l_f_out)
    # test(np.matrix(x_train[0]).astype('int32'))
    else:
        l_out = lasagne.layers.DenseLayer(
            l_forward_slice, num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax,
            name='output_layer')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)
    # test = theano.function([x], network_output)
    # test(np.matrix(x_train[0]).astype('int32'))

    y = T.ivector()

    # The value we care about is the final value produced for each sequence
    predicted_values = T.argmax(network_output, axis=1)

    errors = T.sum(T.neq(predicted_values,y))

    # test = theano.function([x,y], errors)
    # test(np.matrix(x_train[0]).astype('int32'),y_train[0])

    cost = lasagne.objectives.categorical_crossentropy(network_output,y).mean()

    l2_penalty = T.sum([lasagne.regularization.l2(param) for param in lasagne.layers.get_all_params(l_out, regularizable=True)])

    cost = cost + 0.01 * l2_penalty
    # test = theano.function([x,y], [network_output,y,cost])
    # test(np.matrix(x_train[0]).astype('int32'),y_train[0])

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    if not train_in_hid_weight:
        all_params.remove(l_forward.W_in_to_hid)
        all_params.remove(l_backward.W_in_to_hid)

    # print all_params

    updates = lasagne.updates.adagrad(cost, all_params, learning_rate=0.01, epsilon=10**-5)

    train_fn = theano.function(inputs=[x, y], outputs=[cost,errors], updates=updates)
    train_predict_fn = theano.function(inputs=[x, y], outputs=[cost,errors], updates=[])
    predict_fn = theano.function(inputs=[x], outputs=predicted_values, updates=[])

    train_costs_list = []
    train_errors_list = []
    test_costs_list = []
    test_errors_list = []

    print 'Training net...'
    for epoch in range(max_epochs):
        train_cost = 0
        train_error = 0
        test_cost = 0
        test_error = 0
        start = time.time()

        for x_sample, y_sample in zip(x_train,y_train):
            cost, error = train_fn(np.matrix(x_sample).astype('int32'), y_sample)
            train_cost += np.asscalar(cost)
            train_error += np.asscalar(error)

        for x_sample, y_sample in zip(x_test,y_test):
            cost, error = train_predict_fn(np.matrix(x_sample).astype('int32'), y_sample)
            test_cost += cost
            test_error += error
        end = time.time()
        train_costs_list.append(train_cost)
        train_errors_list.append(train_error)
        test_costs_list.append(test_cost)
        test_errors_list.append(test_error)
        print 'Epoch: %d Train_cost: %f Train_errors: %d Test_cost: %f Test_errors: %d Took: %f' % \
              (epoch, train_cost, train_error, test_cost, test_error, end-start)

    print 'Predicting...'
    flat_predictions = []
    for x_sample in x_test:
        y_preds = predict_fn(np.matrix(x_sample).astype('int32'))
        flat_predictions.extend(y_preds)

    flat_true = list(chain(*y_test))

    print 'MACRO results'
    print Metrics.compute_all_metrics(flat_true,flat_predictions,average='macro')

    cPickle.dump(train_costs_list, open('train_costs.p', 'wb'))
    cPickle.dump(train_errors_list, open('train_errors.p', 'wb'))
    cPickle.dump(test_costs_list, open('test_costs.p', 'wb'))
    cPickle.dump(test_errors_list, open('test_errors.p', 'wb'))

    print 'End'