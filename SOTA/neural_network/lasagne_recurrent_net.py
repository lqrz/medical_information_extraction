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

from SOTA.neural_network.recurrent_net import Recurrent_net
from SOTA.neural_network.recurrent_Context_Window_net import Recurrent_Context_Window_net
from SOTA.neural_network.A_neural_network import A_neural_network
from SOTA.neural_network.train_neural_network import load_w2v_model_and_vectors_cache
from utils.metrics import Metrics

np.random.seed(1234)

if __name__ == '__main__':
    add_tags = []
    n_window = 1
    max_epochs = 50
    alpha_l2 = 0.01
    bidirectional = True
    shared_params = True
    train_in_hid_weight = False

    if n_window == 1:
        x_train, y_train, x_train_feats, \
        x_valid, y_valid, x_valid_feats, \
        x_test, y_test, x_test_feats, \
        word2index, index2word, \
        label2index, index2label, \
        features_indexes = \
            Recurrent_net.get_data(clef_training=True, clef_validation=True, clef_testing=True,
                                   add_words=[], add_tags=[], add_feats=[], x_idx=None,
                                   n_window=n_window,
                                   feat_positions=None,
                                   lowercase=True)
    elif n_window > 1:
        x_train, y_train, x_train_feats, \
        x_valid, y_valid, x_valid_feats, \
        x_test, y_test, x_test_feats, \
        word2index, index2word, \
        label2index, index2label, \
        features_indexes = \
            Recurrent_Context_Window_net.get_data(clef_training=True, clef_validation=True, clef_testing=True,
                                                  add_words=[], add_tags=[], add_feats=[], x_idx=None,
                                                  n_window=n_window,
                                                  feat_positions=None,
                                                  lowercase=True)

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

    l_idxs = lasagne.layers.InputLayer(shape=(1, None), input_var=x, name='idxs')

    l1 = lasagne.layers.EmbeddingLayer(l_idxs, input_size=n_words, output_size=n_dims, W=w, name='embeddings_layer')

    # l1_output = lasagne.layers.get_output(l1)
    # test = theano.function([x], l1_output)
    # test(x_train[0])

    w_fw = lasagne.init.HeUniform()

    l_forward = lasagne.layers.RecurrentLayer(
        l1, n_dims, mask_input=None, grad_clipping=0,
        # W_in_to_hid=lasagne.init.HeUniform(),
        W_in_to_hid=np.identity(n=n_dims, dtype=theano.config.floatX),
        W_hid_to_hid=w_fw,
        nonlinearity=lasagne.nonlinearities.tanh,
        name='forward_hidden_layer')

    l_forward_slice = lasagne.layers.SliceLayer(l_forward, indices=-1, axis=0)

    if bidirectional:

        if shared_params:
            w_bw = w_fw
        else:
            w_bw = lasagne.init.HeUniform()

        l_backward = lasagne.layers.RecurrentLayer(
            l1, n_dims, mask_input=None, grad_clipping=0,
            # W_in_to_hid=lasagne.init.HeUniform(),
            W_in_to_hid=np.identity(n=n_dims, dtype=theano.config.floatX),
            W_hid_to_hid=w_bw,
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

    cross_entropy = lasagne.objectives.categorical_crossentropy(network_output,y).mean()

    l2_penalty = T.sum([lasagne.regularization.l2(param) for param in lasagne.layers.get_all_params(l_out, regularizable=True)])

    cost = cross_entropy + alpha_l2 * l2_penalty
    # test = theano.function([x,y], [network_output,y,cost])
    # test(np.matrix(x_train[0]).astype('int32'),y_train[0])

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    if not train_in_hid_weight:
        all_params.remove(l_forward.W_in_to_hid)
        all_params.remove(l_backward.W_in_to_hid)

    # print all_params

    updates = lasagne.updates.adagrad(cost, all_params, learning_rate=0.01, epsilon=10**-5)

    train_fn = theano.function(inputs=[x, y], outputs=[cost, cross_entropy, errors], updates=updates)
    train_predict_fn = theano.function(inputs=[x, y], outputs=[cost, cross_entropy, errors, predicted_values], updates=[])
    predict_fn = theano.function(inputs=[x], outputs=predicted_values, updates=[])

    train_costs_list = []
    train_cross_entropy_list = []
    train_errors_list = []
    valid_costs_list = []
    valid_cross_entropy_list = []
    valid_errors_list = []
    l2_penalty_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []

    valid_flat_true = list(chain(*y_valid))

    print 'Training net...'
    for epoch in range(max_epochs):
        train_cost = 0
        train_cross_entropy = 0
        train_error = 0
        valid_cost = 0
        valid_cross_entropy = 0
        valid_error = 0
        start = time.time()

        valid_predictions = []

        for x_sample, y_sample in zip(x_train, y_train):
            cost_output, cross_entropy_output, error_output = train_fn(np.matrix(x_sample).astype('int32'), y_sample)
            train_cost += np.asscalar(cost_output)
            train_cross_entropy += np.asscalar(cross_entropy_output)
            train_error += np.asscalar(error_output)

        for x_sample, y_sample in zip(x_valid, y_valid):
            cost_output, cross_entropy_output, error_output, pred_output = train_predict_fn(np.matrix(x_sample).astype('int32'), y_sample)
            valid_cost += cost_output
            valid_cross_entropy += cross_entropy_output
            valid_error += error_output
            valid_predictions.extend(pred_output)

        assert valid_flat_true.__len__() == valid_predictions.__len__()
        results = Metrics.compute_all_metrics(y_true=valid_flat_true, y_pred=valid_predictions, average='macro')
        f1_score = results['f1_score']
        precision = results['precision']
        recall = results['recall']

        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

        train_costs_list.append(train_cost)
        train_costs_list.append(train_cross_entropy)
        train_errors_list.append(train_error)
        valid_costs_list.append(valid_cost)
        valid_costs_list.append(valid_cross_entropy)
        valid_errors_list.append(valid_error)

        end = time.time()

        print('Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f Took: %f'
                    % (epoch + 1, train_cost, train_error, valid_cost, valid_error, f1_score, end - start))

        # print 'Predicting...'
        # flat_predictions = []
        # for x_sample in x_test:
        #     y_preds = predict_fn(np.matrix(x_sample).astype('int32'))
        #     flat_predictions.extend(y_preds)


    actual_time = str(time.time())
    rnn = Recurrent_net(hidden_activation_f=None, out_activation_f=None)
    rnn.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                      valid_errors_list,
                                      actual_time)
    rnn.plot_scores(precision_list, recall_list, f1_score_list, actual_time)
    # self.plot_penalties(l2_w1_list=l2_w1_list, l2_w2_list=l2_w2_list, l2_ww_fw_list=l2_ww_list,
    #                     actual_time=actual_time)

    rnn.plot_cross_entropies(train_cross_entropy=train_cross_entropy_list,
                              valid_cross_entropy=valid_cross_entropy_list,
                              actual_time=actual_time)


    # print 'MACRO results'
    # print Metrics.compute_all_metrics(flat_true,flat_predictions,average='macro')

    # cPickle.dump(train_costs_list, open('train_costs.p', 'wb'))
    # cPickle.dump(train_errors_list, open('train_errors.p', 'wb'))
    # cPickle.dump(test_costs_list, open('test_costs.p', 'wb'))
    # cPickle.dump(test_errors_list, open('test_errors.p', 'wb'))

    print 'End'