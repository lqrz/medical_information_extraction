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

from SOTA.neural_network.train_neural_network import load_w2v_model_and_vectors_cache
from SOTA.neural_network.A_neural_network import A_neural_network
from data.dataset import Dataset
from utils.metrics import Metrics


def get_data(clef_training=False, clef_validation=False, clef_testing=False,
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
        x_train, y_train = A_neural_network.get_partitioned_data(x_idx=x_idx,
                                                    document_sentences_words=train_document_sentence_words,
                                                    document_sentences_tags=train_document_sentence_tags,
                                                    word2index=word2index,
                                                    label2index=label2index,
                                                    use_context_window=False,
                                                    n_window=n_window)

    if clef_validation:
        x_valid, y_valid = A_neural_network.get_partitioned_data(x_idx=x_idx,
                                                    document_sentences_words=valid_document_sentence_words,
                                                    document_sentences_tags=valid_document_sentence_tags,
                                                    word2index=word2index,
                                                    label2index=label2index,
                                                    use_context_window=False,
                                                    n_window=n_window)

    if clef_testing:
        x_test, y_test = A_neural_network.get_partitioned_data(x_idx=x_idx,
                                                  document_sentences_words=test_document_sentence_words,
                                                  document_sentences_tags=test_document_sentence_tags,
                                                  word2index=word2index,
                                                  label2index=label2index,
                                                  use_context_window=False,
                                                  n_window=n_window)

    return x_train, y_train, x_train_feats, \
           x_valid, y_valid, x_valid_feats, \
           x_test, y_test, x_test_feats, \
           word2index, index2word, \
           label2index, index2label, \
           features_indexes

def pad_to_max_length(x_train, y_train, x_valid, y_valid, x_test, y_test, pad_ix):

    max_len = determine_datasets_max_len(x_train, x_valid, x_test)

    x_train = map(lambda x: x + [pad_ix] * (max_len - x.__len__()), x_train)
    y_train = map(lambda x: x + [pad_ix] * (max_len - x.__len__()), y_train)
    x_valid = map(lambda x: x + [pad_ix] * (max_len - x.__len__()), x_valid)
    y_valid = map(lambda x: x + [pad_ix] * (max_len - x.__len__()), y_valid)
    x_test = map(lambda x: x + [pad_ix] * (max_len - x.__len__()), x_test)
    y_test = map(lambda x: x + [pad_ix] * (max_len - x.__len__()), y_test)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def determine_datasets_max_len(x_train, x_valid, x_test):
    return np.max(map(len, x_train+x_valid+x_test))


def length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def pad_sentences(x_train, y_train, x_valid, y_valid, x_test, y_test, word2index, label2index):
    pad_word_ix = [word2index['<PAD>']]
    pad_tag_ix = [label2index['<PAD>']]

    x_train = map(lambda x: pad_word_ix+ x + pad_word_ix, x_train)
    x_valid = map(lambda x: pad_word_ix + x + pad_word_ix, x_valid)
    x_test = map(lambda x: pad_word_ix + x + pad_word_ix, x_test)

    y_train = map(lambda x: pad_tag_ix + x + pad_tag_ix, y_train)
    y_valid = map(lambda x: pad_tag_ix + x + pad_tag_ix, y_valid)
    y_test = map(lambda x: pad_tag_ix + x + pad_tag_ix, y_test)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

if __name__ == '__main__':
    batch_size = 1
    input_size = 10
    max_length = 7     # unrolled up to this length
    n_window = 1
    learning_rate = 0.1
    alpha_l2 = 0.001
    max_epochs = 20
    mask_inputs_by_length = True

    if n_window == 1:
        x_train, y_train, x_train_feats, \
        x_valid, y_valid, x_valid_feats, \
        x_test, y_test, x_test_feats, \
        word2index, index2word, \
        label2index, index2label, \
        features_indexes = \
            get_data(clef_training=True, clef_validation=True, clef_testing=True,
                                   add_words=['<PAD>'], add_tags=['<PAD>'], add_feats=[], x_idx=None,
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
            get_data(clef_training=True, clef_validation=True, clef_testing=True,
                                                  add_words=[], add_tags=[], add_feats=[], x_idx=None,
                                                  n_window=n_window,
                                                  feat_positions=None,
                                                  lowercase=True)

    args = {
        'w2v_vectors_cache': 'w2v_googlenews_representations.p',
    }

    x_train, y_train, x_valid, y_valid, x_test, y_test = pad_sentences(x_train, y_train, x_valid, y_valid, x_test, y_test, word2index, label2index)

    w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache(args)

    pretrained_embeddings = A_neural_network.initialize_w(w2v_dims, word2index.keys(),
                                   w2v_vectors=w2v_vectors, w2v_model=w2v_model)

    n_words, n_emb_dims = pretrained_embeddings.shape
    n_out = label2index.keys().__len__()

    assert w2v_dims == n_emb_dims, 'Error in embeddings dimensions computation.'

    pretrained_embeddings = np.vstack((pretrained_embeddings, np.zeros(n_emb_dims)))
    extra_ix_pad = pretrained_embeddings.shape[0] - 1

    x_train, y_train, x_valid, y_valid, x_test, y_test = pad_to_max_length(x_train, y_train, x_valid, y_valid, x_test,
                                                                           y_test, extra_ix_pad)

    max_length = determine_datasets_max_len(x_train, x_valid, x_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):

            tf.set_random_seed(1234)

            # word embeddings matrix and bias. Always needed
            w1 = tf.Variable(initial_value=pretrained_embeddings, dtype=tf.float32, trainable=True, name='w1')
            b1 = tf.Variable(tf.zeros([n_emb_dims * n_window]), dtype=tf.float32, name='b1')

            # data = tf.placeholder(tf.float32, [None, max_length, n_emb_dims])

            idxs = tf.placeholder(tf.int32, name='idxs', shape=[None, max_length])
            # true_labels = tf.placeholder(tf.float32, [None, max_length, index2label.__len__()])
            true_labels = tf.placeholder(tf.int32, name='true_labels')

            # embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
            w_x = tf.nn.embedding_lookup(w1, idxs)

            input_data = tf.reshape(w_x, shape=[-1, max_length, n_emb_dims])

            cell = rnn_cell.BasicRNNCell(num_units=n_emb_dims * n_window, input_size=None)

            # Initial state of the cell.
            # state = tf.zeros([batch_size, n_emb_dims * n_window])

            hidden_activations, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32, sequence_length=length(input_data))
            hidden_activations_flat = tf.reshape(hidden_activations, [-1, n_emb_dims * n_window])

            w2 = tf.Variable(tf.truncated_normal([n_emb_dims * n_window, label2index.__len__()], stddev=0.1))
            b2 = tf.Variable(tf.constant(0.1, shape=[label2index.__len__()]))

            out_logits = tf.matmul(hidden_activations_flat, w2) + b2
            # logits_reshape = tf.reshape(out_logits, [-1, max_length, label2index.__len__()])

            # cross_entropy = tf.reduce_sum(tf.slice(cros_entropies_list, begin=0, size=lengths))
            cross_entropies_list = tf.nn.sparse_softmax_cross_entropy_with_logits(out_logits, true_labels)
            # cross_entropy_unmasked = tf.reduce_sum(cross_entropies_list)

            mask = tf.sign(tf.to_float(tf.not_equal(true_labels, tf.constant(extra_ix_pad))))
            cross_entropy_masked = cross_entropies_list * mask

            cross_entropy = tf.reduce_sum(cross_entropy_masked)
            # cross_entropy /= tf.reduce_sum(mask)
            # cross_entropy = tf.reduce_mean(cross_entropy)

            regularizables = [w1, w2]

            l2_sum = tf.add_n([tf.nn.l2_loss(param) for param in regularizables])
            cost = cross_entropy + alpha_l2 * l2_sum

            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

            predictions = tf.to_int32(tf.arg_max(tf.nn.softmax(out_logits), 1))

            errors_list = tf.to_float(tf.not_equal(predictions, true_labels))
            errors = errors_list * mask
            errors = tf.reduce_sum(errors)

            with tf.Session() as session:
                session.run(tf.initialize_all_variables())
                print("Initialized")
                for epoch_ix in range(max_epochs):
                    start = time.time()
                    n_batches = np.int(np.ceil(x_train.shape[0] / batch_size))
                    train_cost = 0
                    train_cross_entropy = 0
                    train_errors = 0
                    for batch_ix in range(n_batches):
                        feed_dict = {
                                    # true_labels: y_train[batch_ix*batch_size:(batch_ix+1)*batch_size].astype(float),
                                    idxs: x_train[batch_ix*batch_size: (batch_ix+1)*batch_size],
                                    true_labels: list(chain(*y_train[batch_ix*batch_size:(batch_ix+1)*batch_size]))
                        }

                        _, cost_output, cross_entropy_output, errors_output = session.run([optimizer, cost, cross_entropy, errors], feed_dict=feed_dict)
                        train_cost += cost_output
                        train_cross_entropy += cross_entropy_output
                        train_errors += errors_output

                    feed_dict = {idxs: x_valid, true_labels: list(chain(*y_valid))}
                    valid_predictions, valid_cost, valid_errors = session.run([predictions, cost, errors], feed_dict)

                    valid_f1 = Metrics.compute_f1_score(y_true=list(chain(*y_valid)), y_pred=valid_predictions, average='macro')

                    end = time.time()

                    print 'Epoch %d - Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d Valid_f1: %f Took: %f' % \
                          (epoch_ix, train_cost, train_errors, valid_cost, valid_errors, valid_f1, end-start)

        # rnn.bidirectional_rnn(cell_fw=, cell_bw=, inputs=, initial_state_fw=, initial_state_bw=,
        #                       sequence_length=None, scope=None)