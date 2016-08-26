__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Check this tutorial:
# https://danijar.com/variable-sequence-lengths-in-tensorflow/

import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
import numpy as np
from itertools import chain
import time
import cPickle

from SOTA.neural_network.A_neural_network import A_neural_network
from data.dataset import Dataset
from utils.metrics import Metrics
from utils import utils

class Recurrent_net(A_neural_network):

    def __init__(self, grad_clip, pad_tag, bidirectional, rnn_cell_type, **kwargs):
        super(Recurrent_net, self).__init__(**kwargs)

        self.graph = tf.Graph()

        self.bidirectional = bidirectional
        self.rnn_cell_type = rnn_cell_type

        self.w1 = None
        self.w2 = None
        self.b2 = None
        self.ww = None

        # parameters to get L2
        self.regularizables = []

        self.grad_clip = isinstance(grad_clip, int)

        if grad_clip:
            self.max_length = grad_clip
        else:
            self.max_length = self._determine_datasets_max_len()

        self.pad_tag = pad_tag
        self._add_padding_representation_to_embeddings()
        self.filling_ix = self._determing_padding_representation_ix()

        if self.grad_clip:
            self._grad_clip_dataset()
        else:
            self._pad_datasets_to_max_length()

        self.initialize_plotting_lists()

        self.initialize_parameters()

    def train(self, **kwargs):

        minibatch_size = kwargs['batch_size']
        learning_rate = kwargs['learning_rate_train']

        if self.bidirectional:
            print('Using a bidirectional RNN cells: %s minibatch: %d lr: %d clipping_grad: %d' % \
                  (self.rnn_cell_type, minibatch_size, learning_rate, self.max_length))

            self.instantiate_cells = self.instantiate_bidirectional_cells
            self.compute_output_layer_logits = self.compute_output_layer_logits_bidirectional
            self.compute_inner_weights_sum = self.compute_inner_weights_sum_bidirectional

            self._train_graph(minibatch_size=minibatch_size,
                                 learning_rate=learning_rate,
                                 **kwargs)

        else:
            print('Using an unidirectional RNN cells %s minibatch: %d lr: %d clipping_grad: %d' % \
                  (self.rnn_cell_type, minibatch_size, learning_rate, self.max_length))

            self.instantiate_cells = self.instantiate_unidirectional_cells
            self.compute_output_layer_logits = self.compute_output_layer_logits_unidirectional
            self.compute_inner_weights_sum = self.compute_inner_weights_sum_unidirectional

            self._train_graph(minibatch_size=minibatch_size,
                                 learning_rate=learning_rate,
                                 **kwargs)

        return True

    def instantiate_bidirectional_cells(self, keep_prob):

        if self.rnn_cell_type == 'normal':
            self.cell_fw = rnn_cell.BasicRNNCell(num_units=self.n_emb * self.n_window, input_size=None)
            self.cell_bw = rnn_cell.BasicRNNCell(num_units=self.n_emb * self.n_window, input_size=None)
        elif self.rnn_cell_type == 'lstm':
            self.cell_fw = rnn_cell.BasicLSTMCell(num_units=self.n_emb * self.n_window)
            self.cell_bw = rnn_cell.BasicLSTMCell(num_units=self.n_emb * self.n_window)
        elif self.rnn_cell_type == 'gru':
            self.cell_fw = rnn_cell.GRUCell(num_units=self.n_emb * self.n_window)
            self.cell_bw = rnn_cell.GRUCell(num_units=self.n_emb * self.n_window)
        else:
            raise Exception()

        self.cell_fw = rnn_cell.DropoutWrapper(self.cell_fw, output_keep_prob=keep_prob)
        self.cell_bw = rnn_cell.DropoutWrapper(self.cell_bw, output_keep_prob=keep_prob)

        return True

    def instantiate_unidirectional_cells(self, keep_prob):
        if self.rnn_cell_type == 'normal':
            self.cell = rnn_cell.BasicRNNCell(num_units=self.n_emb * self.n_window, input_size=None)
        elif self.rnn_cell_type == 'lstm':
            self.cell = rnn_cell.BasicLSTMCell(num_units=self.n_emb * self.n_window)
        elif self.rnn_cell_type == 'gru':
            self.cell = rnn_cell.GRUCell(num_units=self.n_emb * self.n_window)
        else:
            raise Exception()

        self.cell = rnn_cell.DropoutWrapper(self.cell, output_keep_prob=keep_prob)

        return True

    def initialize_parameters(self):
        with self.graph.as_default():
            tf.set_random_seed(1234)

            self.w1 = tf.Variable(initial_value=self.pretrained_embeddings, dtype=tf.float32, trainable=True, name='w1')

            if self.bidirectional:
                # forward and backward
                self.w2 = tf.Variable(tf.truncated_normal([2 * self.n_emb * self.n_window, self.n_out], stddev=0.1))
            else:
                self.w2 = tf.Variable(tf.truncated_normal([self.n_emb * self.n_window, self.n_out], stddev=0.1))

            self.b2 = tf.Variable(tf.constant(0.1, shape=[self.n_out]))

        return

    def perform_lookup_and_reshape(self, idxs):
        w_x = tf.nn.embedding_lookup(self.w1, idxs)
        return tf.reshape(w_x, shape=[-1, self.n_window * self.n_emb])

    def compute_output_layer_logits_unidirectional(self, idxs):

        w_x_r = self.perform_lookup_and_reshape(idxs)

        input_data = tf.reshape(w_x_r, shape=[-1, self.max_length, self.n_window * self.n_emb])

        hidden_activations, _ = tf.nn.dynamic_rnn(self.cell, input_data, dtype=tf.float32,
                                                  sequence_length=self.length(input_data))

        hidden_activations_flat = tf.reshape(hidden_activations, [-1, self.n_emb * self.n_window])

        return tf.matmul(hidden_activations_flat, self.w2) + self.b2

    def compute_output_layer_logits_bidirectional(self, idxs):
        w_x_r = self.perform_lookup_and_reshape(idxs)

        input_data = tf.reshape(w_x_r, shape=[-1, self.max_length, self.n_window * self.n_emb])

        # """Split the single tensor of a sequence into a list of frames."""
        input_data_unp = tf.unpack(tf.transpose(input_data, perm=[1, 0, 2]))

        hidden_activations, _, _ = tf.nn.bidirectional_rnn(self.cell_fw, self.cell_bw, input_data_unp,
                                                           dtype=tf.float32,
                                                           sequence_length=self.length(input_data))

        # """Combine a list of the frames into a single tensor of the sequence."""
        hidden_activations_flat = tf.reshape(tf.transpose(tf.pack(hidden_activations), perm=[1, 0, 2]),
                                                [-1, 2 * self.n_emb * self.n_window])

        return tf.matmul(hidden_activations_flat, self.w2) + self.b2

    def compute_inner_weights_sum_unidirectional(self):

        if self.rnn_cell_type == 'normal':
            with tf.variable_scope("RNN"):
                with tf.variable_scope("BasicRNNCell"):
                    with tf.variable_scope("Linear"):
                        tf.get_variable_scope().reuse_variables()
                        self.ww = tf.get_variable("Matrix")
        elif self.rnn_cell_type == 'lstm':
            with tf.variable_scope("RNN"):
                with tf.variable_scope("BasicLSTMCell"):
                    with tf.variable_scope("Linear"):
                        tf.get_variable_scope().reuse_variables()
                        self.ww = tf.get_variable("Matrix")
        elif self.rnn_cell_type == 'gru':
            with tf.variable_scope("RNN"):
                with tf.variable_scope("GRUCell"):
                    with tf.variable_scope("Candidate"):
                        with tf.variable_scope("Linear"):
                            tf.get_variable_scope().reuse_variables()
                            ww_candidate = tf.get_variable("Matrix")

            with tf.variable_scope("RNN"):
                with tf.variable_scope("GRUCell"):
                    with tf.variable_scope("Gates"):
                        with tf.variable_scope("Linear"):
                            tf.get_variable_scope().reuse_variables()
                            ww_gates = tf.get_variable("Matrix")

            self.ww = tf.concat(1, [ww_candidate, ww_gates])

        else:
            raise Exception()

        return True

    def compute_inner_weights_sum_bidirectional(self):

        if self.rnn_cell_type == 'normal':

            with tf.variable_scope("BiRNN_FW"):
                with tf.variable_scope("BasicRNNCell"):
                    with tf.variable_scope("Linear"):
                        tf.get_variable_scope().reuse_variables()
                        ww_fw = tf.get_variable("Matrix")

            with tf.variable_scope("BiRNN_BW"):
                with tf.variable_scope("BasicRNNCell"):
                    with tf.variable_scope("Linear"):
                        tf.get_variable_scope().reuse_variables()
                        ww_bw = tf.get_variable("Matrix")

            self.ww = tf.concat(0, [ww_fw, ww_bw])

        elif self.rnn_cell_type == 'lstm':

            with tf.variable_scope("BiRNN_FW"):
                with tf.variable_scope("BasicLSTMCell"):
                    with tf.variable_scope("Linear"):
                        tf.get_variable_scope().reuse_variables()
                        ww_fw = tf.get_variable("Matrix")
            with tf.variable_scope("BiRNN_BW"):
                with tf.variable_scope("BasicLSTMCell"):
                    with tf.variable_scope("Linear"):
                        tf.get_variable_scope().reuse_variables()
                        ww_bw = tf.get_variable("Matrix")

            self.ww = tf.concat(0, [ww_fw, ww_bw])

        elif self.rnn_cell_type == 'gru':

            with tf.variable_scope("BiRNN_FW"):
                with tf.variable_scope("RNN"):
                    with tf.variable_scope("GRUCell"):
                        with tf.variable_scope("Candidate"):
                            with tf.variable_scope("Linear"):
                                tf.get_variable_scope().reuse_variables()
                                fw_ww_candidate = tf.get_variable("Matrix")

                with tf.variable_scope("RNN"):
                    with tf.variable_scope("GRUCell"):
                        with tf.variable_scope("Gates"):
                            with tf.variable_scope("Linear"):
                                tf.get_variable_scope().reuse_variables()
                                fw_ww_gates = tf.get_variable("Matrix")

            with tf.variable_scope("BiRNN_BW"):
                with tf.variable_scope("RNN"):
                    with tf.variable_scope("GRUCell"):
                        with tf.variable_scope("Candidate"):
                            with tf.variable_scope("Linear"):
                                tf.get_variable_scope().reuse_variables()
                                bw_ww_candidate = tf.get_variable("Matrix")

                with tf.variable_scope("RNN"):
                    with tf.variable_scope("GRUCell"):
                        with tf.variable_scope("Gates"):
                            with tf.variable_scope("Linear"):
                                tf.get_variable_scope().reuse_variables()
                                bw_ww_gates = tf.get_variable("Matrix")

            self.ww = tf.concat(1, [fw_ww_candidate, fw_ww_gates, bw_ww_candidate, bw_ww_gates])

        else:
            raise Exception()

        return True

    def _train_graph(self, minibatch_size, max_epochs, learning_rate, plot, **kwargs):

        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate

        with self.graph.as_default():
            with tf.device('/cpu:0'):

                if self.n_window > 1:
                    idxs = tf.placeholder(tf.int32, name='idxs', shape=[None, self.max_length, self.n_window])
                else:
                    idxs = tf.placeholder(tf.int32, name='idxs', shape=[None, self.max_length])

                true_labels = tf.placeholder(tf.int32, name='true_labels')

                keep_prob = tf.placeholder(tf.float32, name='keep_prob')

                with tf.variable_scope("cell"):
                    self.instantiate_cells(keep_prob)

                    out_logits = self.compute_output_layer_logits(idxs)

                    self.compute_inner_weights_sum()

                # cross_entropy = tf.reduce_sum(tf.slice(cros_entropies_list, begin=0, size=lengths))
                cross_entropies_list = tf.nn.sparse_softmax_cross_entropy_with_logits(out_logits, true_labels)
                # cross_entropy_unmasked = tf.reduce_sum(cross_entropies_list)

                mask = tf.sign(tf.to_float(tf.not_equal(true_labels, tf.constant(self.filling_ix))))
                cross_entropy_masked = cross_entropies_list * mask

                cross_entropy = tf.reduce_sum(cross_entropy_masked)
                # cross_entropy /= tf.reduce_sum(mask)
                # cross_entropy = tf.reduce_mean(cross_entropy)

                # self.regularizables = [w_x, self.w2]
                # l2_sum = tf.add_n([tf.nn.l2_loss(param) for param in self.regularizables])

                cost = cross_entropy

                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

                predictions = self.compute_predictions(out_logits)

                errors_list = tf.to_float(tf.not_equal(predictions, true_labels))
                errors = errors_list * mask
                errors = tf.reduce_sum(errors)

                # with vs.variable_scope("Linear"):
                #     matrix = vs.get_variable("Matrix", None)

        with tf.Session(graph=self.graph) as session:

            valid_mask = np.array(sorted(set(np.where(self.y_valid.reshape((-1)) != self.filling_ix)[0]).intersection(set(np.where(self.y_valid.reshape((-1)) != self.pad_tag)[0]))))
            flat_y_valid = self.y_valid.reshape((-1))

            session.run(tf.initialize_all_variables())
            print("Initialized")

            early_stopping_cnt_since_last_update = 0
            early_stopping_min_validation_cost = np.inf
            early_stopping_min_iteration = None
            model_update = None

            for epoch_ix in range(max_epochs):

                if self.early_stopping_threshold is not None:
                    if early_stopping_cnt_since_last_update > self.early_stopping_threshold:
                        assert early_stopping_min_iteration is not None
                        print('Training early stopped at iteration %d' % early_stopping_min_iteration)
                        break

                start = time.time()
                n_batches = np.int(np.ceil(self.x_train.shape[0] / float(minibatch_size)))
                train_cost = 0
                train_cross_entropy = 0
                train_errors = 0
                for batch_ix in range(n_batches):
                    feed_dict = {
                        # true_labels: y_train[batch_ix*batch_size:(batch_ix+1)*batch_size].astype(float),
                        idxs: self.x_train[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size],
                        true_labels: list(chain(*self.y_train[batch_ix * minibatch_size:(batch_ix + 1) * minibatch_size])),
                        keep_prob: 0.5
                    }

                    _, cost_output, cross_entropy_output, errors_output = session.run(
                        [optimizer, cost, cross_entropy, errors], feed_dict=feed_dict)
                    train_cost += cost_output
                    train_cross_entropy += cross_entropy_output
                    train_errors += errors_output

                feed_dict = {idxs: self.x_valid, true_labels: list(chain(*self.y_valid)), keep_prob: 1.}
                valid_predictions, valid_cost, valid_cross_entropy, valid_errors = session.run([predictions, cost,
                                                                                                cross_entropy, errors],
                                                                                               feed_dict)
                valid_true = flat_y_valid[valid_mask]
                valid_predictions = valid_predictions[valid_mask]
                precision, recall, f1_score = self.compute_scores(valid_true, valid_predictions)

                if plot:
                    epoch_l2_w1, epoch_l2_w2, epoch_l2_ww = self.compute_parameters_sum()

                    self.update_monitoring_lists(train_cost, train_cross_entropy, train_errors,
                                                 valid_cost, valid_cross_entropy, valid_errors,
                                                 epoch_l2_w1, epoch_l2_w2, epoch_l2_ww,
                                                 precision, recall, f1_score)


                if valid_cost < early_stopping_min_validation_cost:
                    self.saver = tf.train.Saver(tf.all_variables())
                    self.saver.save(session, self.get_output_path('params.model'), write_meta_graph=True)
                    early_stopping_min_iteration = epoch_ix
                    early_stopping_min_validation_cost = valid_cost
                    early_stopping_cnt_since_last_update = 0
                    model_update = True
                else:
                    early_stopping_cnt_since_last_update += 1
                    model_update = False

                assert model_update is not None

                print('epoch: %d train_cost: %f train_errors: %d valid_cost: %f valid_errors: %d F1: %f upd: %s took: %f' \
                      % (epoch_ix, train_cost, train_errors, valid_cost, valid_errors, f1_score, model_update, time.time() - start))

        if plot:
            print 'Making plots'
            self.make_plots()

        if self.pickle_lists:
            print('Pickling lists')
            self.pickle_lists()

        return True

    def pickle_lists(self):
        """
        This is to make some later plots.
        """
        output_filename = '_'.join([self.rnn_cell_type, str(self.minibatch_size), str(self.learning_rate), str(self.bidirectional), str(self.max_length)])

        cPickle.dump(self.valid_costs_list, open(self.get_output_path('validation_cost_list-'+output_filename+'.p'),'wb'))
        cPickle.dump(self.valid_cross_entropy_list, open(self.get_output_path('valid_cross_entropy_list-'+output_filename+'.p'),'wb'))
        cPickle.dump(self.train_costs_list, open(self.get_output_path('train_cost_list-'+output_filename+'.p'),'wb'))
        cPickle.dump(self.train_cross_entropy_list, open(self.get_output_path('train_cross_entropy_list-'+output_filename+'.p'),'wb'))
        cPickle.dump(self.f1_score_list, open(self.get_output_path('valid_f1_score_list-'+output_filename+'.p'),'wb'))

        return True

    def predict(self, on_training_set=False, on_validation_set=False, on_testing_set=False, **kwargs):

        assert on_training_set or on_validation_set or on_testing_set
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

            mask = np.array(sorted(set(np.where(y_test.reshape((-1)) != self.filling_ix)[0]).intersection(
                set(np.where(y_test.reshape((-1)) != self.pad_tag)[0]))))
            flat_y_test = y_test.reshape((-1))

            tf.train.import_meta_graph(self.get_output_path('params.model.meta'))

            idxs = self.graph.get_tensor_by_name(name='idxs:0')
            keep_prob = self.graph.get_tensor_by_name(name='keep_prob:0')

            with tf.variable_scope("cell") as scope:
                scope.reuse_variables()
            # with tf.variable_scope("RNN"):
            #     tf.get_variable_scope().reuse_variables()
                out_logits = self.compute_output_layer_logits(idxs)
                predictions = self.compute_predictions(out_logits)

        with tf.Session(graph=self.graph) as session:
            self.saver.restore(session, self.get_output_path('params.model'))

            feed_dict = {idxs: x_test, keep_prob: 1.}

            prediction_output = session.run(predictions, feed_dict=feed_dict)

        results['flat_trues'] = flat_y_test[mask]
        results['flat_predictions'] = prediction_output[mask]

        return results

    def compute_predictions(self, out_logits):
        return tf.to_int32(tf.arg_max(tf.nn.softmax(out_logits), 1))

    def update_monitoring_lists(self,
                                train_cost, train_xentropy, train_errors,
                                valid_cost, valid_xentropy, valid_errors,
                                epoch_l2_w1, epoch_l2_w2, epoch_l2_ww,
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
        self.epoch_l2_ww_list.append(epoch_l2_ww)

        # scores
        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.f1_score_list.append(f1_score)

        return

    def compute_parameters_sum(self):
        w1_sum = None
        w2_sum = None

        ww_sum = None

        if self.w1 is not None:
            w1_sum = tf.reduce_sum(tf.square(self.w1)).eval()

        if self.w2 is not None:
            w2_sum = tf.reduce_sum(tf.square(self.w2)).eval()

        if self.ww is not None:
            ww_sum = tf.reduce_sum(tf.square(self.ww)).eval()

        return w1_sum, w2_sum, ww_sum

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

        if self.ww is not None:
            plot_data_dict['ww'] = self.epoch_l2_ww_list

        self.plot_penalties_general(plot_data_dict, actual_time=actual_time)

        return

    def compute_scores(self, true_values, predictions):

        assert true_values.__len__() == predictions.__len__()

        results = Metrics.compute_all_metrics(y_true=true_values, y_pred=predictions, average='macro')
        f1_score = results['f1_score']
        precision = results['precision']
        recall = results['recall']

        return precision, recall, f1_score

    def initialize_plotting_lists(self):

        # plotting purposes
        self.train_costs_list = []
        self.train_errors_list = []
        self.valid_costs_list = []
        self.valid_errors_list = []
        self.precision_list = []
        self.recall_list = []
        self.f1_score_list = []
        self.epoch_l2_w1_list = []
        self.epoch_l2_w2_list = []
        self.epoch_l2_ww_list = []
        self.train_cross_entropy_list = []
        self.valid_cross_entropy_list = []

        return True

    @classmethod
    def _get_data_as_sentences(cls, clef_training=False, clef_validation=False, clef_testing=False,
                 add_words=[], add_tags=[], add_feats=[], x_idx=None, n_window=None, feat_positions=None,
                 lowercase=True, use_context_window=None):

        assert not use_context_window

        document_sentence_words = []
        document_sentence_tags = []

        x_train = None
        y_train = None
        y_valid = None
        x_valid = None
        y_test = None
        x_test = None
        features_indexes = None
        x_train_feats = None
        x_valid_feats = None
        x_test_feats = None

        if clef_training:
            train_features, _, train_document_sentence_words, train_document_sentence_tags = Dataset.get_clef_training_dataset()

            document_sentence_words.extend(train_document_sentence_words.values())
            document_sentence_tags.extend(train_document_sentence_tags.values())

        if clef_validation:
            valid_features, _, valid_document_sentence_words, valid_document_sentence_tags = Dataset.get_clef_validation_dataset()

            document_sentence_words.extend(valid_document_sentence_words.values())
            document_sentence_tags.extend(valid_document_sentence_tags.values())

        if clef_testing:
            test_features, _, test_document_sentence_words, test_document_sentence_tags = Dataset.get_clef_testing_dataset()

            document_sentence_words.extend(test_document_sentence_words.values())
            document_sentence_tags.extend(test_document_sentence_tags.values())

        word2index, index2word = A_neural_network._construct_index(add_words, document_sentence_words)
        label2index, index2label = A_neural_network._construct_index(add_tags, document_sentence_tags)

        if clef_training:
            x_train, y_train = cls.get_partitioned_data(x_idx=x_idx,
                                                      document_sentences_words=train_document_sentence_words,
                                                      document_sentences_tags=train_document_sentence_tags,
                                                      word2index=word2index,
                                                      label2index=label2index,
                                                      use_context_window=use_context_window,
                                                      n_window=n_window)

        if clef_validation:
            x_valid, y_valid = cls.get_partitioned_data(x_idx=x_idx,
                                                      document_sentences_words=valid_document_sentence_words,
                                                      document_sentences_tags=valid_document_sentence_tags,
                                                      word2index=word2index,
                                                      label2index=label2index,
                                                      use_context_window=use_context_window,
                                                      n_window=n_window)

        if clef_testing:
            x_test, y_test = cls.get_partitioned_data(x_idx=x_idx,
                                                    document_sentences_words=test_document_sentence_words,
                                                    document_sentences_tags=test_document_sentence_tags,
                                                    word2index=word2index,
                                                    label2index=label2index,
                                                    use_context_window=use_context_window,
                                                    n_window=n_window)

        # if i use context window, the senteces are padded when constructing the windows. If not, i have to do it here.
        x_train, y_train, x_valid, y_valid, x_test, y_test = cls._pad_sentences(x_train, y_train, x_valid, y_valid,
                                                                                x_test,
                                                                                y_test, word2index, label2index)

        return x_train, y_train, x_train_feats, \
               x_valid, y_valid, x_valid_feats, \
               x_test, y_test, x_test_feats, \
               word2index, index2word, \
               label2index, index2label, \
               features_indexes




    @classmethod
    def _get_data_as_sentences_of_windows(cls, clef_training=False, clef_validation=False, clef_testing=False,
                 add_words=[], add_tags=[], add_feats=[], x_idx=None, n_window=None, feat_positions=None,
                 lowercase=True, use_context_window=None):

        assert use_context_window

        document_sentence_words = []
        document_sentence_tags = []

        x_train = None
        y_train = None
        y_valid = None
        x_valid = None
        y_test = None
        x_test = None
        features_indexes = None
        x_train_feats = None
        x_valid_feats = None
        x_test_feats = None

        if clef_training:
            train_features, _, train_document_sentence_words, train_document_sentence_tags = Dataset.get_clef_training_dataset()

            document_sentence_words.extend(train_document_sentence_words.values())
            document_sentence_tags.extend(train_document_sentence_tags.values())

        if clef_validation:
            valid_features, _, valid_document_sentence_words, valid_document_sentence_tags = Dataset.get_clef_validation_dataset()

            document_sentence_words.extend(valid_document_sentence_words.values())
            document_sentence_tags.extend(valid_document_sentence_tags.values())

        if clef_testing:
            test_features, _, test_document_sentence_words, test_document_sentence_tags = Dataset.get_clef_testing_dataset()

            document_sentence_words.extend(test_document_sentence_words.values())
            document_sentence_tags.extend(test_document_sentence_tags.values())

        word2index, index2word = A_neural_network._construct_index(add_words, document_sentence_words)
        label2index, index2label = A_neural_network._construct_index(add_tags, document_sentence_tags)

        if clef_training:
            x_train, y_train = cls._get_partitioned_data_sentences_of_windows(x_idx=x_idx,
                                                                     document_sentences_words=train_document_sentence_words,
                                                                     document_sentences_tags=train_document_sentence_tags,
                                                                     word2index=word2index,
                                                                     label2index=label2index,
                                                                     use_context_window=use_context_window,
                                                                     n_window=n_window)

        if clef_validation:
            x_valid, y_valid = cls._get_partitioned_data_sentences_of_windows(x_idx=x_idx,
                                                                     document_sentences_words=valid_document_sentence_words,
                                                                     document_sentences_tags=valid_document_sentence_tags,
                                                                     word2index=word2index,
                                                                     label2index=label2index,
                                                                     use_context_window=use_context_window,
                                                                     n_window=n_window)

        if clef_testing:
            x_test, y_test = cls._get_partitioned_data_sentences_of_windows(x_idx=x_idx,
                                                                   document_sentences_words=test_document_sentence_words,
                                                                   document_sentences_tags=test_document_sentence_tags,
                                                                   word2index=word2index,
                                                                   label2index=label2index,
                                                                   use_context_window=use_context_window,
                                                                   n_window=n_window)

        return x_train, y_train, x_train_feats, \
               x_valid, y_valid, x_valid_feats, \
               x_test, y_test, x_test_feats, \
               word2index, index2word, \
               label2index, index2label, \
               features_indexes

    @classmethod
    def _get_partitioned_data_with_context_window_sentences_of_windows(cls, doc_sentences, n_window, item2index):
        x = []
        for sentence in doc_sentences:
            x.append([map(lambda x: item2index[x], sent_window)
                      for sent_window in utils.NeuralNetwork.context_window(sentence, n_window)])

        return x

    @classmethod
    def _get_partitioned_data_sentences_of_windows(cls, x_idx, document_sentences_words, document_sentences_tags,
                             word2index, label2index, use_context_window=False, n_window=None, **kwargs):
        """
        this is at sentence level.
        it partitions the training data according to the x_idx doc_nrs used for training and y_idx doc_nrs used for
        testing while cross-validating.

        :param x_idx:
        :param y_idx:
        :return:
        """
        x = []
        y = []

        for doc_nr, doc_sentences in document_sentences_words.iteritems():
            if (not x_idx) or (doc_nr in x_idx):
                # training set
                if use_context_window:
                    x_doc = cls._get_partitioned_data_with_context_window_sentences_of_windows(doc_sentences, n_window, word2index)
                else:
                    x_doc = cls._get_partitioned_data_without_context_window(doc_sentences, word2index)

                y_doc = [map(lambda x: label2index[x] if x else None, sent) for sent in
                         document_sentences_tags[doc_nr]]

                x.extend(x_doc)
                y.extend(y_doc)

        return x, y


    @classmethod
    def get_data(cls, clef_training=False, clef_validation=False, clef_testing=False,
                 add_words=[], add_tags=[], add_feats=[], x_idx=None, n_window=None, feat_positions=None,
                 lowercase=True):
        """
        overrides the inherited method.
        gets the training data and organizes it into sentences per document.
        RNN overrides this method, cause other neural nets dont partition into sentences.

        :param crf_training_data_filename:
        :return:
        """

        assert clef_training or clef_validation or clef_testing

        use_context_window = False if not n_window or n_window == 1 else True

        if use_context_window:
            x_train, y_train, x_train_feats, \
            x_valid, y_valid, x_valid_feats, \
            x_test, y_test, x_test_feats, \
            word2index, index2word, \
            label2index, index2label, \
            features_indexes = cls._get_data_as_sentences_of_windows(clef_training=clef_training, clef_validation=clef_validation,
                                                          clef_testing=clef_testing,
                 add_words=add_words, add_tags=add_tags, add_feats=add_feats, x_idx=x_idx, n_window=n_window,
                                                          feat_positions=feat_positions,
                 lowercase=lowercase, use_context_window=use_context_window)
        else:
            x_train, y_train, x_train_feats, \
            x_valid, y_valid, x_valid_feats, \
            x_test, y_test, x_test_feats, \
            word2index, index2word, \
            label2index, index2label, \
            features_indexes = cls._get_data_as_sentences(clef_training=clef_training, clef_validation=clef_validation,
                                                          clef_testing=clef_testing,
                 add_words=add_words, add_tags=add_tags, add_feats=add_feats, x_idx=x_idx, n_window=n_window,
                                                          feat_positions=feat_positions,
                 lowercase=lowercase, use_context_window=use_context_window)

        return x_train, y_train, x_train_feats, \
               x_valid, y_valid, x_valid_feats, \
               x_test, y_test, x_test_feats, \
               word2index, index2word, \
               label2index, index2label, \
               features_indexes

    def _pad_datasets_to_max_length(self):

        if self.n_window > 1:
            # i have to add entire windows of fillings
            filling = [[self.filling_ix] * self.n_window]

        else:
            filling = [self.filling_ix]

        assert isinstance(filling, list)

        self.x_train = np.array(map(lambda x: x + filling * (self.max_length - x.__len__()), self.x_train))
        self.y_train = np.array(map(lambda x: x + [self.filling_ix] * (self.max_length - x.__len__()), self.y_train))
        self.x_valid = np.array(map(lambda x: x + filling * (self.max_length - x.__len__()), self.x_valid))
        self.y_valid = np.array(map(lambda x: x + [self.filling_ix] * (self.max_length - x.__len__()), self.y_valid))
        self.x_test = np.array(map(lambda x: x + filling * (self.max_length - x.__len__()), self.x_test))
        self.y_test = np.array(map(lambda x: x + [self.filling_ix] * (self.max_length - x.__len__()), self.y_test))

        return True

    def _grad_clip_dataset(self):
        self.x_train, self.y_train = self._grad_clip_dataset_(self.x_train, self.y_train)
        self.x_valid, self.y_valid = self._grad_clip_dataset_(self.x_valid, self.y_valid)
        self.x_test, self.y_test = self._grad_clip_dataset_(self.x_test, self.y_test)

        return True

    def _grad_clip_dataset_(self, dataset_x, dataset_y):
        x = []
        y = []
        for sentence_x, sentence_y in zip(dataset_x, dataset_y):
            to_add = np.int(np.ceil(sentence_x.__len__() / float(self.max_length))* self.max_length - sentence_x.__len__())
            padded_sentence_x = sentence_x + [self.filling_ix] * to_add
            padded_sentence_y = sentence_y + [self.filling_ix] * to_add

            x.extend([padded_sentence_x[i*self.max_length:(i+1)*self.max_length] for i in range(np.int(np.ceil(padded_sentence_x.__len__() / float(self.max_length))))])
            y.extend([padded_sentence_y[i*self.max_length:(i+1)*self.max_length] for i in range(np.int(np.ceil(padded_sentence_y.__len__() / float(self.max_length))))])

        return np.array(x), np.array(y)

    def _determine_datasets_max_len(self):
        return np.max(map(len, list(self.x_train) + list(self.x_valid) + list(self.x_test)))

    def length(self, data):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def _add_padding_representation_to_embeddings(self):

        self.pretrained_embeddings = np.vstack((self.pretrained_embeddings, np.zeros(self.n_emb)))

    def _determing_padding_representation_ix(self):
        return self.pretrained_embeddings.shape[0] - 1

    @classmethod
    def _pad_sentences(cls, x_train, y_train, x_valid, y_valid, x_test, y_test, word2index, label2index):
        pad_word_ix = [word2index['<PAD>']]
        pad_tag_ix = [label2index['<PAD>']]

        x_train = map(lambda x: pad_word_ix+ x + pad_word_ix, x_train)
        x_valid = map(lambda x: pad_word_ix + x + pad_word_ix, x_valid)
        x_test = map(lambda x: pad_word_ix + x + pad_word_ix, x_test)

        y_train = map(lambda x: pad_tag_ix + x + pad_tag_ix, y_train)
        y_valid = map(lambda x: pad_tag_ix + x + pad_tag_ix, y_valid)
        y_test = map(lambda x: pad_tag_ix + x + pad_tag_ix, y_test)

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def to_string(self):
        return '[Tensorflow] Recurrent neural network'

    def get_hidden_activations(self, on_training_set, on_validation_set, on_testing_set, **kwargs):

       return []

    def get_output_logits(self, on_training_set, on_validation_set, on_testing_set, **kwargs):

        return []

if __name__ == '__main__':
    batch_size = 128
    max_length = 7     # unrolled up to this length
    n_window = 1
    learning_rate = 0.1
    alpha_l2 = 0.001
    max_epochs = 15
    mask_inputs_by_length = True

    # x_train, y_train, x_train_feats, \
    # x_valid, y_valid, x_valid_feats, \
    # x_test, y_test, x_test_feats, \
    # word2index, index2word, \
    # label2index, index2label, \
    # features_indexes = \
    #     Recurrent_net.get_data(clef_training=True, clef_validation=True, clef_testing=True,
    #                            add_words=['<PAD>'], add_tags=['<PAD>'], add_feats=[], x_idx=None,
    #                            n_window=n_window,
    #                            feat_positions=None,
    #                            lowercase=True)

    # args = {
    #     'w2v_vectors_cache': 'w2v_googlenews_representations.p',
    # }
    #
    # w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache(args)

    # pretrained_embeddings = A_neural_network.initialize_w(w2v_dims, word2index.keys(),
    #                                w2v_vectors=w2v_vectors, w2v_model=w2v_model)

    # n_words, n_emb_dims = pretrained_embeddings.shape
    # n_out = label2index.keys().__len__()

    # assert w2v_dims == n_emb_dims, 'Error in embeddings dimensions computation.'





    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_valid = np.array(x_valid)
    # y_valid = np.array(y_valid)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)



        # rnn.bidirectional_rnn(cell_fw=, cell_bw=, inputs=, initial_state_fw=, initial_state_bw=,
        #                       sequence_length=None, scope=None)