__author__ = 'lqrz'

from SOTA.neural_network.A_neural_network import A_neural_network
from utils.metrics import Metrics

import time
import tensorflow as tf
import numpy as np

class Convolutional_Neural_Net(A_neural_network):

    def __init__(self, pos_embeddings, n_filters, region_sizes, features_to_use, **kwargs):

        super(Convolutional_Neural_Net, self).__init__(**kwargs)

        self.graph = tf.Graph()

        # parameters
        self.w1_w2v = None
        self.w1_pos = None
        self.w2 = None
        self.b2 = None

        # embeddings
        self.pos_embeddings = pos_embeddings
        self.n_pos_emb = pos_embeddings.shape[1]

        # convolution filters params
        self.n_filters = n_filters
        self.region_sizes = region_sizes

        # w2v filters
        self.w2v_filters_weights = []
        self.w2v_filters_bias = []

        self.features_to_use = self.parse_features_to_use(features_to_use)

        self.n_hidden = self.determine_hidden_layer_size()

        # parameters to get L2
        self.regularizables = []

        self.fine_tuning_params = []
        self.training_params = []

        self.initialize_plotting_lists()

    def parse_features_to_use(self, features_to_use):
        return [self.__class__.FEATURE_MAPPING[feat] for feat in features_to_use]

    def determine_hidden_layer_size(self):
        size = 0

        for desc in self.features_to_use:

            feature = desc['name']
            convolve = desc['convolve']
            max_pool = desc['max_pool']
            c_window = desc['use_cw']
            feat_dim = self.get_features_dimensions()[feature]

            feat_extended = '_'.join([feature, 'c' if convolve else 'nc', 'm' if max_pool else 'nm'])

            if convolve:
                window = 1
                n_filters = self.determine_nr_filters(feat_extended)
                n_regions = self.determine_nr_region_sizes(feat_extended)
                filter_windows = np.sum([self.n_window - rs + 1 for rs in self.region_sizes])
                filter_window_width = self.determine_resulting_width(feat_extended, feat_dim)
            else:
                window = self.n_window
                n_regions = 1
                n_filters = 1
                filter_window_width = feat_dim
                filter_windows = 1

            if max_pool:
                feat_dimension = 1
                filter_windows = self.determine_nr_region_sizes(feat_extended)
            else:
                feat_dimension = feat_dim

            if not c_window:
                window = 1

            # size += feat_dimension * window * n_regions * n_filters
            size += filter_windows * filter_window_width * window * n_filters

        return size

    def get_features_dimensions(self):
        d = dict()
        d['w2v'] = self.n_emb
        # d['pos'] = self.n_pos_emb
        # d['ner'] = self.n_ner_emb
        # d['sent_nr'] = self.n_sent_nr_emb
        # d['tense'] = self.n_tense_emb

        return d

    def determine_nr_filters(self, feature):
        nr_filters = self.__class__.FEATURE_MAPPING[feature]['nr_filters']
        if not nr_filters:
            nr_filters = self.n_filters

        assert nr_filters is not None, 'Could not determine nr_filters for feature: %s' % feature

        return nr_filters

    def determine_nr_region_sizes(self, feature):
        nr_regions = self.__class__.FEATURE_MAPPING[feature]['nr_region_sizes']
        if not nr_regions:
            nr_regions = self.region_sizes.__len__()

        assert nr_regions is not None, 'Could not determine nr_region_sizes for feature: %s' % feature

        return nr_regions

    def determine_resulting_width(self, feature, feat_dim):
        resulting_width = None
        filter_width = self.__class__.FEATURE_MAPPING[feature]['filter_width']

        if filter_width is not None:
            resulting_width = feat_dim / filter_width
        else:
            resulting_width = 1

        assert resulting_width is not None, 'Could not determine nr_region_sizes for feature: %s' % feature

        return resulting_width

    def initialize_plotting_lists(self):
        pass

    def train(self, static, batch_size, **kwargs):
        self.initialize_parameters(static)

        if not batch_size:
            # train SGD
            minibatch_size = 1
        else:
            minibatch_size = batch_size

        self._train_graph(minibatch_size, **kwargs)

    def initialize_parameters(self, static):
        with self.graph.as_default():

            # word embeddings matrix. Always needed
            self.w1_w2v = tf.Variable(initial_value=self.pretrained_embeddings, dtype=tf.float32, trainable=not static,
                                      name='w1_w2v')

            self.regularizables.append(self.w1_w2v)
            self.fine_tuning_params.append(self.w1_w2v)

            self.w1_pos = tf.Variable(initial_value=self.pos_embeddings, dtype=tf.float32, trainable=not static, name='w1_pos')

            self.regularizables.append(self.w1_pos)

            self.w2 = tf.Variable(tf.truncated_normal(shape=[self.n_hidden, self.n_out], stddev=0.1))
            self.b2 = tf.Variable(tf.constant(0.1, shape=[self.n_out]))

            # this is for the w2v convolution
            for i, rs in enumerate(self.region_sizes):
                filter_shape = [rs, self.n_emb, 1, self.n_filters]
                w_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w_filter_shape'+str(rs))
                b_filter = tf.Variable(tf.constant(0.1, shape=[self.n_filters]), name='b_filter_shape'+str(rs))
                self.w2v_filters_weights.append(w_filter)
                self.w2v_filters_bias.append(b_filter)

        return

    def _train_graph(self, minibatch_size, max_epochs,
                     learning_rate_train, learning_rate_tune, lr_decay,
                     plot, alpha_l2=0.001,
                     **kwargs):

        with self.graph.as_default():

            w2v_idxs = tf.placeholder(tf.int32, shape=[None, self.n_window], name='w2v_idxs')
            pos_idxs = tf.placeholder(tf.int32, name='pos_idxs')
            labels = tf.placeholder(tf.int32, name='labels')
            keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            w1_x = tf.nn.embedding_lookup(self.w1_w2v, w2v_idxs)
            w1_x_expanded = tf.expand_dims(w1_x, -1)
            # w1_x_r = tf.reshape(w1_x, shape=[-1, self.n_window * self.n_emb])

            pooled_out = []
            for filter_weight, filter_bias, filter_size in zip(self.w2v_filters_weights, self.w2v_filters_bias, self.region_sizes):
            # filter_weight = self.w2v_filters_weights[0]
            # filter_bias = self.w2v_filters_bias[0]
            # filter_size = self.region_sizes[0]
                # "VALID" padding means that we slide the filter over our sentence without padding the edges
                conv = tf.nn.conv2d(input=w1_x_expanded, filter=filter_weight, strides=[1,1,1,1], padding='VALID')
                a = tf.nn.bias_add(conv, filter_bias)
                h = tf.nn.relu(a)
                pooled = tf.nn.max_pool(value=h, ksize=[1, self.n_window-filter_size+1, 1, 1], strides=[1,1,1,1], padding='VALID')
                pooled_out.append(pooled)

            n_total_filters = self.n_filters * len(self.region_sizes)
            h_pooled = tf.concat(concat_dim=3, values=pooled_out)
            h_pooled_flat = tf.reshape(h_pooled, shape=[-1, n_total_filters])

            h_dropout = tf.nn.dropout(h_pooled_flat, keep_prob)

            out_logits = tf.nn.xw_plus_b(h_dropout, self.w2, self.b2)

            l2 = tf.add_n([tf.nn.l2_loss(param) for param in self.regularizables])

            cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_logits, labels=labels))
            cost = cross_entropy + alpha_l2 * l2

            predictions = tf.to_int32(tf.argmax(out_logits, 1))
            n_errors = tf.reduce_sum(tf.to_int32(tf.not_equal(predictions, labels)))

            # optimizer = tf.train.AdamOptimizer(1e-3)
            optimizer_fine_tune = tf.train.AdagradOptimizer(learning_rate=learning_rate_tune)
            optimizer_train = tf.train.AdagradOptimizer(learning_rate=learning_rate_train)
            grads = tf.gradients(cost, self.fine_tuning_params + self.training_params)
            fine_tuning_grads = grads[:len(self.fine_tuning_params)]
            training_grads = grads[-len(self.training_params):]
            fine_tune_op = optimizer_fine_tune.apply_gradients(zip(fine_tuning_grads, self.fine_tuning_params))
            optimizer = fine_tune_op
            # train_op = optimizer_train.apply_gradients(zip(training_grads, self.training_params))
            # optimizer = tf.group(fine_tune_op, train_op)

        with tf.Session(graph=self.graph) as session:
            session.run(tf.initialize_all_variables())
            n_batches = np.int(np.ceil(self.x_train.shape[0] / minibatch_size))
            for epoch_ix in range(max_epochs):
                start = time.time()
                train_cost = 0
                train_xentropy = 0
                train_errors = 0
                for batch_ix in range(n_batches):
                    feed_dict = {
                        w2v_idxs: self.x_train[batch_ix*minibatch_size: (batch_ix+1)*minibatch_size],
                        labels: self.y_train[batch_ix*minibatch_size: (batch_ix+1)*minibatch_size],
                        keep_prob: 0.5,
                    }
                    _, cost_val, xentropy, train_errors = session.run([optimizer, cost, cross_entropy, n_errors], feed_dict=feed_dict)
                    train_cost += cost_val
                    train_xentropy += xentropy
                    train_errors += train_errors

                feed_dict = {
                    w2v_idxs: self.x_valid,
                    labels: self.y_valid,
                    keep_prob: 1.,
                }
                valid_cost, valid_xentropy, pred, valid_errors = session.run([cost, cross_entropy, predictions, n_errors], feed_dict=feed_dict)
                precision, recall, f1_score = self.compute_scores(self.y_valid, pred)

                print 'epoch: %d train_cost: %f train_errors: %d valid_cost: %f valid_errors: %d F1: %f took: %f' \
                      % (
                          epoch_ix, train_cost, train_errors, valid_cost, valid_errors, f1_score, time.time() - start)

    def compute_scores(self, true_values, predictions):
        results = Metrics.compute_all_metrics(y_true=true_values, y_pred=predictions, average='macro')
        f1_score = results['f1_score']
        precision = results['precision']
        recall = results['recall']

        return precision, recall, f1_score

    def predict(self, on_training_set=False, on_validation_set=False, on_testing_set=False, **kwargs):
        pass

    def to_string(self):
        print '[Tensorflow] Convolutional neural network'