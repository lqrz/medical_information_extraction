__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from utils.utils import NeuralNetwork
from utils.metrics import Metrics
from data import load_w2v_vectors
from SOTA.neural_network.A_neural_network import A_neural_network

import time
import tensorflow as tf

def get_dataset(n_window, add_words=[], add_tags=[], feat_positions=[], add_feats=[]):
  x_train, y_train, x_train_feats, \
  x_valid, y_valid, x_valid_feats, \
  x_test, y_test, x_test_feats, \
  word2index, index2word, \
  label2index, index2label, \
  features_indexes = \
    A_neural_network.get_data(clef_training=True, clef_validation=True, clef_testing=True, add_words=add_words,
                      add_tags=add_tags, add_feats=add_feats, x_idx=None, n_window=n_window,
                      feat_positions=feat_positions)

  return x_train, y_train, x_valid, y_valid, x_test, y_test, word2index, label2index

# x_train, y_train, x_valid, y_valid, x_test, y_test, word2index, label2index = get_dataset(n_window=3, add_words=['<PAD>'])

def train_graph(n_window, x_train, y_train, x_valid, y_valid, word2index, label2index,
                epochs, embeddings=None, embedding_size=300, alpha_l2=0.001):
  graph = tf.Graph()

  n_unique_words = word2index.keys().__len__()
  n_out = label2index.keys().__len__()

  with graph.as_default():

    # Input data
    idxs = tf.placeholder(tf.int32)
    labels = tf.placeholder(tf.int32)

    with tf.device('/cpu:0'):

      # word embeddings matrix
      if embeddings is not None:
          print 'Embeddings: using param embeddings'
          w1 = tf.Variable(initial_value=embeddings, dtype=tf.float32)
      else:
          print 'Embeddings: randomly initialising'
          w1 = tf.Variable(
            initial_value=NeuralNetwork.initialize_weights(n_in=n_unique_words, n_out=embedding_size, function='tanh'),
            dtype=tf.float32)

      w2 = tf.Variable(
        initial_value=NeuralNetwork.initialize_weights(n_in=n_window*embedding_size, n_out=n_out, function='softmax'),
        dtype=tf.float32)

      b1 = tf.Variable(tf.zeros([embedding_size * n_window]))
      b2 = tf.Variable(tf.zeros([n_out]))

      # embedding lookup
      w1_x = tf.nn.embedding_lookup(w1, idxs)
      w1_x_r = tf.reshape(w1_x, shape=[-1, n_window * embedding_size])

      # forward pass
      h = tf.tanh(tf.nn.bias_add(w1_x_r, b1))
      out = tf.matmul(h, w2) + b2

    # note: tf.log computes the natural logarithm
    cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(out, labels))

    # l2 regularization
    l2_regularizers = (tf.nn.l2_loss(w1_x) + tf.nn.l2_loss(w2))

    cost = tf.reduce_sum(cross_entropy + alpha_l2 * l2_regularizers)

    optimizer = tf.train.AdagradOptimizer(learning_rate=.01).minimize(cost)

    predictions = tf.to_int32(tf.arg_max(tf.nn.softmax(out), 1))

    n_errors = tf.reduce_sum(tf.to_int32(tf.not_equal(predictions, labels)))

    init = tf.initialize_all_variables()

  with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    for epoch_ix in range(epochs):
      start = time.time()
      train_cost = 0
      train_xentropy = 0
      train_errors = 0
      for x_sample, y_sample in zip(x_train, y_train):
        feed_dict = {idxs: x_sample, labels: [y_sample]}
        # session.run([optimizer, out, tf.nn.softmax(out), predictions, n_errors], feed_dict={idxs: x_sample, labels: [y_sample]})
        _, cost_val, xentropy, pred, errors = session.run([optimizer, cost, cross_entropy, predictions, n_errors], feed_dict=feed_dict)
        train_cost += cost_val
        train_xentropy += xentropy
        train_errors += errors
      emb_sum = tf.reduce_sum(tf.square(w1)).eval()

      feed_dict = {idxs: x_valid, labels: y_valid}
      # session.run([out, y_valid, cross_entropy, tf.reduce_sum(cross_entropy)], feed_dict=feed_dict)
      valid_cost, valid_xentropy, pred, valid_errors = session.run([cost, cross_entropy, predictions, n_errors], feed_dict=feed_dict)

      valid_f1 = Metrics.compute_f1_score(y_true=y_valid, y_pred=pred, average='macro')

      print 'epoch: %d train_cost: %f train_errors: %d valid_cost: %f valid_errors: %d F1: %f took: %f' \
            % (epoch_ix, train_cost, train_errors, valid_cost, valid_errors, valid_f1, time.time()-start)


if __name__ == '__main__':

    n_window = 3
    x_train, y_train, x_valid, y_valid, x_test, y_test, word2index, label2index = get_dataset(n_window=n_window, add_words=['<PAD>'])

    word_vectors_name = 'googlenews_representations_train_True_valid_True_test_False.p'
    word_vectors = load_w2v_vectors(word_vectors_name)
    w2v_dims = word_vectors.values()[0].shape[0]
    embeddings = A_neural_network.initialize_w(w2v_dims=w2v_dims, unique_words=word2index.keys(), w2v_model=None, w2v_vectors=word_vectors)
    train_graph(n_window, x_train, y_train, x_valid, y_valid, word2index, label2index, embeddings=embeddings, epochs=20)