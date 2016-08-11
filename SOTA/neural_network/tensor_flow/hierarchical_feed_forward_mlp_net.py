__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from utils.utils import NeuralNetwork
from utils.metrics import Metrics
from data import load_w2v_vectors
from SOTA.neural_network.tensor_flow.feed_forward_mlp_net import Neural_Net
from SOTA.neural_network.A_neural_network import A_neural_network
from utils.huffman_encoding import Huffman_encoding

import time
import tensorflow as tf
import numpy as np
import argparse


class Hierarchical_feed_forward_neural_net(Neural_Net):

    def __init__(self, log_reg, n_hidden, na_tag, **kwargs):

        super(Hierarchical_feed_forward_neural_net, self).__init__(log_reg, n_hidden, na_tag, **kwargs)

        self.graph = tf.Graph()

        self.log_reg = log_reg
        self.n_hidden = n_hidden

        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.w3 = None
        self.b3 = None

        # dynamic assignment of funtion. Depends on architecture
        self.forward_function = None
        self.hidden_activations = None

        self.na_tag = na_tag

        # parameters to get L2
        self.regularizables = []

        # split into training and fine-tuning
        self.training_params = []
        self.fine_tuning_params = []

        self.initialize_plotting_lists()

        self.convert_y_datasets()

    def convert_y_datasets(self):
        if np.any(np.equal(self.y_test, None)):
            # there are no values for the test dataset. Do not include it.
            encoding = Huffman_encoding(np.concatenate([self.y_train, self.y_valid])).encode()
        else:
            # include the test dataset
            encoding = Huffman_encoding(np.concatenate([self.y_train, self.y_valid, self.y_test])).encode()
            self.y_test = map(lambda x: encoding[x], self.y_test)

        self.y_train = map(lambda x: encoding[x], self.y_train)
        self.y_valid = map(lambda x: encoding[x], self.y_valid)

        return True

    def hidden_activations_one_hidden_layer(self, w1_x_r):
        return tf.tanh(tf.nn.bias_add(w1_x_r, self.b1))

    def forward_one_hidden_layer(self, w1_x_r):
        # forward pass
        h = self.hidden_activations(w1_x_r)
        out = tf.matmul(h, self.w2) + self.b2

        return out

    def hidden_activations_two_hidden_layer(self, w1_x_r):
        h1 = tf.tanh(tf.nn.bias_add(w1_x_r, self.b1))
        a2 = tf.matmul(h1, self.w2)
        h2 = tf.tanh(tf.nn.bias_add(a2, self.b2))

        return h2

    def forward_two_hidden_layer(self, w1_x_r):
        h2 = self.hidden_activations(w1_x_r)
        a3 = tf.matmul(h2, self.w3)
        out = tf.nn.bias_add(a3, self.b3)

        return out

    def lookup_and_reshape(self, idxs):
        w1_x = tf.nn.embedding_lookup(self.w1, idxs)
        w1_x_r = tf.reshape(w1_x, shape=[-1, self.n_window * self.n_emb])

        return w1_x_r

    def compute_output_layer_logits(self, idxs):
        # embedding lookup
        w1_x_r = self.lookup_and_reshape(idxs)
        out = self.forward_function(w1_x_r)

        return out

    def compute_predictions(self, out_logits):

        predictions = tf.to_int32(tf.arg_max(tf.nn.softmax(out_logits), 1))

        return predictions

    def compute_errors(self, predictions, labels):

        n_errors = tf.reduce_sum(tf.to_int32(tf.not_equal(predictions, labels)))

        return n_errors

    def train_graph(self, minibatch_size, max_epochs,
                    learning_rate_train, learning_rate_tune, lr_decay,
                    plot,
                    alpha_l2=0.001,
                    alpha_na=None,
                    **kwargs):

        with self.graph.as_default():
            tf.set_random_seed(1234)

            # Input data
            idxs = tf.placeholder(tf.int32, name='idxs')
            labels = tf.placeholder(tf.int32, name='labels')

            na_label = tf.placeholder(tf.int32, name='na_label')

            out_logits = self.compute_output_layer_logits(idxs)

            # note: tf.log computes the natural logarithm
            cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(out_logits, labels))

            # l2 regularization
            l2_regularizers = tf.add_n([tf.nn.l2_loss(param) for param in self.regularizables])

            if alpha_na is not None:
                na_prob = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(out_logits, na_label))
                cost = tf.reduce_sum(cross_entropy + alpha_l2 * l2_regularizers + alpha_na * na_prob)
            else:
                cost = tf.reduce_sum(cross_entropy + alpha_l2 * l2_regularizers)

            # they both do the same: split the learning rate. But the 2nd one is slightly more performant

            # optimizer_fine_tune = tf.train.AdagradOptimizer(learning_rate=learning_rate_tune).\
            #     minimize(cost, var_list=self.fine_tuning_params)
            #
            # optimizer_train = tf.train.AdagradOptimizer(learning_rate=learning_rate_train).\
            #     minimize(cost, var_list=self.training_params)
            # optimizer = tf.group(optimizer_fine_tune, optimizer_train)

            optimizer_fine_tune = tf.train.AdagradOptimizer(learning_rate=learning_rate_tune)
            optimizer_train = tf.train.AdagradOptimizer(learning_rate=learning_rate_train)
            grads = tf.gradients(cost, self.fine_tuning_params + self.training_params)
            fine_tuning_grads = grads[:len(self.fine_tuning_params)]
            training_grads = grads[-len(self.training_params):]
            fine_tune_op = optimizer_fine_tune.apply_gradients(zip(fine_tuning_grads, self.fine_tuning_params))
            train_op = optimizer_train.apply_gradients(zip(training_grads, self.training_params))
            optimizer = tf.group(fine_tune_op, train_op)

            predictions = self.compute_predictions(out_logits)

            n_errors = self.compute_errors(predictions, labels)

        with tf.Session(graph=self.graph) as session:
            session.run(tf.initialize_all_variables())
            print("Initialized")

            n_batches = np.int(np.ceil(self.x_train.shape[0] / minibatch_size))

            for epoch_ix in range(max_epochs):
                start = time.time()
                train_cost = 0
                train_xentropy = 0
                train_errors = 0

                for batch_ix in range(n_batches):
                    x_sample = self.x_train[batch_ix * minibatch_size:(batch_ix + 1) * minibatch_size]
                    y_sample = self.y_train[batch_ix * minibatch_size:(batch_ix + 1) * minibatch_size]

                    if alpha_na is not None:
                        na_sample = [self.na_tag] * x_sample.shape[0]
                        feed_dict_train = {idxs: x_sample, labels: y_sample, na_label: na_sample}
                        feed_dict_valid = {idxs: self.x_valid, labels: self.y_valid, na_label: [self.na_tag]*self.x_valid.shape[0]}
                    else:
                        feed_dict_train = {idxs: x_sample, labels: y_sample}
                        feed_dict_valid = {idxs: self.x_valid, labels: self.y_valid}

                    _, cost_val, xentropy, pred, errors = session.run(
                        [optimizer, cost, cross_entropy, predictions, n_errors], feed_dict=feed_dict_train)
                    train_cost += cost_val
                    train_xentropy += xentropy
                    train_errors += errors

                # session.run([out, y_valid, cross_entropy, tf.reduce_sum(cross_entropy)], feed_dict=feed_dict)
                valid_cost, valid_xentropy, pred, valid_errors = session.run(
                    [cost, cross_entropy, predictions, n_errors], feed_dict=feed_dict_valid)

                precision, recall, f1_score = self.compute_scores(self.y_valid, pred)

                if plot:
                    epoch_l2_w1, epoch_l2_w2, epoch_l2_w3 = self.compute_parameters_sum()

                    self.update_monitoring_lists(train_cost, train_xentropy, train_errors,
                                                 valid_cost, valid_xentropy, valid_errors,
                                                 epoch_l2_w1, epoch_l2_w2, epoch_l2_w3,
                                                 precision, recall, f1_score)

                print 'epoch: %d train_cost: %f train_errors: %d valid_cost: %f valid_errors: %d F1: %f took: %f' \
                      % (
                      epoch_ix, train_cost, train_errors, valid_cost, valid_errors, f1_score, time.time() - start)

            self.saver = tf.train.Saver(self.training_params + self.fine_tuning_params)
            self.saver.save(session, self.get_output_path('params.model'))

        if plot:
            print 'Making plots'
            self.make_plots()

    def predict(self, on_training_set=False, on_validation_set=False, on_testing_set=False, **kwargs):

        results = dict()

        if on_training_set:
            x_test = self.x_train
            y_test = self.y_train
        elif on_validation_set:
            x_test = self.x_valid
            y_test = self.y_valid
        elif on_testing_set:
            x_test = self.x_test
            y_test = self.y_test
        else:
            raise Exception

        with self.graph.as_default():

            # Input data
            # idxs = tf.placeholder(tf.int32)
            # out_logits = self.compute_output_layer_logits(idxs)
            idxs = self.graph.get_tensor_by_name(name='idxs:0')
            out_logits = self.compute_output_layer_logits(idxs)

            predictions = self.compute_predictions(out_logits)

            # init = tf.initialize_all_variables()

        with tf.Session(graph=self.graph) as session:
            # init.run()

            self.saver.restore(session, self.get_output_path('params.model'))

            feed_dict = {idxs: x_test}
            pred = session.run(predictions, feed_dict=feed_dict)

        results['flat_trues'] = y_test
        results['flat_predictions'] = pred

        return results

    def make_plots(self):
        actual_time = str(time.time())

        self.plot_training_cost_and_error(self.train_costs_list, self.train_errors_list,
                                          self.valid_costs_list, self.valid_errors_list,
                                          actual_time=actual_time)

        self.plot_scores(self.precision_list, self.recall_list, self.f1_score_list,
                         actual_time=actual_time)

        self.plot_cross_entropies(self.train_cross_entropy_list, self.valid_cross_entropy_list,
                                  actual_time=actual_time)

        plot_data_dict = dict()

        if self.w1 is not None:
            plot_data_dict['w1'] = self.epoch_l2_w1_list

        if self.w2 is not None:
            plot_data_dict['w2'] = self.epoch_l2_w2_list

        if self.w3 is not None:
            plot_data_dict['w3'] = self.epoch_l2_w3_list

        self.plot_penalties_general(plot_data_dict, actual_time=actual_time)

        return

    def compute_parameters_sum(self):
        w1_sum = 0
        w2_sum = 0
        w3_sum = 0

        if self.w1 is not None:
            w1_sum = tf.reduce_sum(tf.square(self.w1)).eval()

        if self.w2 is not None:
            w2_sum = tf.reduce_sum(tf.square(self.w2)).eval()

        if self.w3 is not None:
            w3_sum = tf.reduce_sum(tf.square(self.w3)).eval()

        return w1_sum, w2_sum, w3_sum

    def update_monitoring_lists(self,
                                train_cost, train_xentropy, train_errors,
                                valid_cost, valid_xentropy, valid_errors,
                                epoch_l2_w1, epoch_l2_w2, epoch_l2_w3,
                                precision, recall, f1_score):

        # training
        self.train_costs_list.append(train_cost)
        self.train_cross_entropy_list.append(train_xentropy)
        self.train_errors_list.append(train_errors)

        # validation
        self.valid_costs_list.append(valid_cost)
        self.valid_cross_entropy_list.append(valid_xentropy)
        self.valid_errors_list.append(valid_errors)

        # weights
        self.epoch_l2_w1_list.append(epoch_l2_w1)
        self.epoch_l2_w2_list.append(epoch_l2_w2)
        self.epoch_l2_w3_list.append(epoch_l2_w3)

        # scores
        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.f1_score_list.append(f1_score)

        return

    def compute_scores(self, true_values, predictions):
        results = Metrics.compute_all_metrics(y_true=true_values, y_pred=predictions, average='macro')
        f1_score = results['f1_score']
        precision = results['precision']
        recall = results['recall']

        return precision, recall, f1_score

    def to_string(self):
        return '[Tensorflow] Neural network'

    def get_hidden_activations(self, on_training_set, on_validation_set, on_testing_set, **kwargs):

        hidden_activations = None

        if on_training_set:
            x_test = self.x_train
        elif on_validation_set:
            x_test = self.x_valid
        elif on_testing_set:
            x_test = self.x_test
        else:
            raise Exception

        with self.graph.as_default():
            idxs = self.graph.get_tensor_by_name(name='idxs:0')
            w1_x_r = self.lookup_and_reshape(idxs)
            h = self.hidden_activations(w1_x_r)

        with tf.Session(graph=self.graph) as session:
            # init.run()

            self.saver.restore(session, self.get_output_path('params.model'))

            feed_dict = {idxs: x_test}
            hidden_activations = session.run(h, feed_dict=feed_dict)

        return hidden_activations

    def get_output_logits(self, on_training_set, on_validation_set, on_testing_set, **kwargs):

        output_logits = None

        if on_training_set:
            x_test = self.x_train
        elif on_validation_set:
            x_test = self.x_valid
        elif on_testing_set:
            x_test = self.x_test
        else:
            raise Exception

        with self.graph.as_default():
            idxs = self.graph.get_tensor_by_name(name='idxs:0')
            out_logits = self.compute_output_layer_logits(idxs)

        with tf.Session(graph=self.graph) as session:
            # init.run()

            self.saver.restore(session, self.get_output_path('params.model'))

            feed_dict = {idxs: x_test}
            output_logits = session.run(out_logits, feed_dict=feed_dict)

        return output_logits

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural net trainer')

    parser.add_argument('--net', type=str, action='store', required=True,
                        choices=['single_cw', 'hidden_cw', 'vector_tag', 'last_tag', 'rnn', 'cw_rnn', 'multi_hidden_cw',
                                 'two_hidden_cw'],
                        help='NNet type')
    parser.add_argument('--window', type=int, action='store', required=True,
                        help='Context window size. 1 for RNN')
    parser.add_argument('--epochs', type=int, action='store', required=True,
                        help='Nr of training epochs.')

    group_w2v = parser.add_mutually_exclusive_group(required=True)
    group_w2v.add_argument('--w2vvectorscache', action='store', type=str)
    group_w2v.add_argument('--w2vmodel', action='store', type=str, default=None)
    group_w2v.add_argument('--w2vrandomdim', action='store', type=int, default=None)

    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--lr', action='store', type=float, default=.1)
    parser.add_argument('--lrdecay', action='store_true', default=False)
    parser.add_argument('--hidden', action='store', type=int, default=False)

    # parse arguments
    arguments = parser.parse_args()

    args = dict()
    args['nn_name'] = arguments.net
    args['window_size'] = arguments.window
    args['max_epochs'] = arguments.epochs
    args['w2v_vectors_cache'] = arguments.w2vvectorscache
    args['w2v_model_name'] = arguments.w2vmodel
    args['w2v_random_dim'] = arguments.w2vrandomdim
    args['plot'] = arguments.plot
    args['learning_rate'] = arguments.lr
    args['n_hidden'] = arguments.hidden
    args['lr_decay'] = arguments.lrdecay

    return args


def initialize_embedding_matrix(args, word2index):
    embeddings = None

    word_vectors_name = args['w2v_vectors_cache']
    random_dim = args['w2v_random_dim']
    if word_vectors_name is not None:
        # use the pretrained w2v embeddings
        word_vectors = load_w2v_vectors(word_vectors_name)
        w2v_dims = word_vectors.values()[0].shape[0]
        embeddings = A_neural_network.initialize_w(w2v_dims=w2v_dims, unique_words=word2index.keys(), w2v_model=None,
                                                   w2v_vectors=word_vectors)
    elif random_dim is not None:
        embeddings = NeuralNetwork.initialize_weights(word2index.keys().__len__(), random_dim, function='tanh')
    else:
        raise Exception

    assert embeddings is not None

    return embeddings

def train_graph(n_window, x_train, y_train, x_valid, y_valid, word2index, label2index, epochs, alpha_l2=0.01):
    '''
    This is a simpler version (unstructured) of the neural network class above. For debugging purposes.
    '''
    n_unique_words = word2index.__len__()
    emb_size = 300
    n_out = label2index.__len__()

    graph = tf.Graph()

    with graph.as_default():

        idxs = tf.placeholder(dtype=tf.int32)
        labels = tf.placeholder(dtype=tf.int32)

        with tf.device('/cpu:0'):

            w1 = tf.Variable(initial_value=NeuralNetwork.initialize_weights(n_in=n_unique_words, n_out=emb_size, function='tanh'),
                             dtype=tf.float32, trainable=True, name='w1')

            b1 = tf.Variable(tf.zeros([emb_size * n_window]), dtype=tf.float32, name='b1')

            w2 = tf.Variable(NeuralNetwork.initialize_weights(n_in=emb_size * n_window, n_out=n_out, function='softmax'),
                             dtype=tf.float32, trainable=True, name='w2')

            b2 = tf.Variable(tf.zeros([n_out]), dtype=tf.float32, name='b2')

            w1_x = tf.nn.embedding_lookup(w1, idxs)
            w1_x_r = tf.reshape(w1_x, shape=[-1, n_window * emb_size])

            out = tf.nn.bias_add(tf.matmul(tf.tanh(tf.nn.bias_add(w1_x_r, b1)), w2), b2)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(out,labels)

            l2 = tf.reduce_sum(tf.nn.l2_loss(w1_x)) + tf.reduce_sum(tf.nn.l2_loss(w2))
            cost = tf.reduce_sum(cross_entropy + alpha_l2 * l2)

            optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(cost)

            predictions = tf.to_int32(tf.arg_max(tf.nn.softmax(out), 1))

            n_errors = tf.reduce_sum(tf.to_int32(tf.not_equal(predictions, labels)))

            init = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        init.run()
        for epoch_ix in range(epochs):
            start = time.time()
            train_cost_acc = 0
            train_errors_acc = 0
            for x_sample, y_sample in zip(x_train, y_train):
                _, train_cost, train_errors = session.run([optimizer, cost, n_errors], feed_dict={idxs: x_sample, labels: [y_sample]})
                train_cost_acc += train_cost
                train_errors_acc += train_errors

            valid_cost, pred, valid_errors = session.run([cost, predictions, n_errors], feed_dict={idxs: x_valid, labels: y_valid})

            f1_score = Metrics.compute_f1_score(y_valid, pred, average='macro')

            print 'Epoch %d train_cost: %f train_errors: %d valid_cost: %f valid_errors: %d F1: %f took: %f' % \
                  (epoch_ix, train_cost_acc, train_errors_acc, valid_cost, valid_errors, f1_score, time.time()-start)


if __name__ == '__main__':

    # args = parse_arguments()
    # n_window = args['window_size']

    n_window = 3

    x_train, y_train, x_valid, y_valid, x_test, y_test, word2index, label2index = get_dataset(n_window=n_window, add_words=['<PAD>'])

    # embeddings = initialize_embedding_matrix(args, word2index)

    train_graph(n_window, x_train, y_train, x_valid, y_valid, word2index, label2index, epochs=20)