__author__ = 'lqrz'

from SOTA.neural_network.A_neural_network import A_neural_network
from utils.metrics import Metrics

import time
import tensorflow as tf
import numpy as np
import cPickle

np.random.seed(1234)

class Multi_feat_Neural_Net(A_neural_network):

    def __init__(self, pos_embeddings, ner_embeddings, sent_nr_embeddings, tense_embeddings, cnn_features,
                 train_pos_feats, valid_pos_feats, test_pos_feats,
                 train_ner_feats, valid_ner_feats, test_ner_feats,
                 train_sent_nr_feats, valid_sent_nr_feats, test_sent_nr_feats,
                 train_tense_feats, valid_tense_feats, test_tense_feats,
                 training_param_names, tuning_param_names,
                 n_hidden,
                 **kwargs):

        super(Multi_feat_Neural_Net, self).__init__(**kwargs)

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1234)

        # parameters
        self.w1_w2v = None
        self.w1_pos = None
        self.w1_ner = None
        self.w1_sent_nr = None
        self.w1_tense = None
        self.w2 = None
        self.b2 = None
        self.w3 = None
        self.b3 = None

        # datasets
        self.train_pos_feats = train_pos_feats
        self.valid_pos_feats = valid_pos_feats
        self.test_pos_feats = test_pos_feats
        self.train_ner_feats = train_ner_feats
        self.valid_ner_feats = valid_ner_feats
        self.test_ner_feats = test_ner_feats
        self.train_sent_nr_feats = train_sent_nr_feats
        self.valid_sent_nr_feats = valid_sent_nr_feats
        self.test_sent_nr_feats = test_sent_nr_feats
        self.train_tense_feats = train_tense_feats
        self.valid_tense_feats = valid_tense_feats
        self.test_tense_feats = test_tense_feats

        # embeddings
        self.pos_embeddings = pos_embeddings
        self.n_pos_emb = pos_embeddings.shape[1] if pos_embeddings is not None else None

        self.ner_embeddings = ner_embeddings
        self.n_ner_emb = ner_embeddings.shape[1] if ner_embeddings is not None else None

        self.sent_nr_embeddings = sent_nr_embeddings
        self.n_sent_nr_emb = sent_nr_embeddings.shape[1] if sent_nr_embeddings is not None else None

        self.tense_embeddings = tense_embeddings
        self.n_tense_emb = tense_embeddings.shape[1] if tense_embeddings is not None else None

        # w2v filters
        self.w2v_filters_weights = []
        self.w2v_filters_bias = []
        # pos filters
        self.pos_filters_weights = []
        self.pos_filters_bias = []
        # ner filters
        self.ner_filters_weights = []
        self.ner_filters_bias = []
        # sent_nr filters
        self.sent_nr_filters_weights = []
        self.sent_nr_filters_bias = []
        # tense filters
        self.tense_filters_weights = []
        self.tense_filters_bias = []

        self.features_to_use = self.parse_features_to_use(cnn_features)

        self.n_hidden = n_hidden
        self.hidden_layer_size = self.determine_hidden_layer_size()

        # parameters to get L2
        self.regularizables = []

        self.training_param_names = training_param_names
        self.tuning_param_names = tuning_param_names
        self.fine_tuning_params = []
        self.training_params = []

        self.initialize_plotting_lists()

    def parse_features_to_use(self, cnn_features):
        features_to_use = cnn_features.keys()

        features_configuration = dict()

        for feat in features_to_use:
            settings = self.__class__.FEATURE_MAPPING[feat]
            settings['nr_filters'] = cnn_features[feat]['n_filters']
            settings['nr_region_sizes'] = cnn_features[feat]['region_sizes'].__len__()
            settings['region_sizes'] = cnn_features[feat]['region_sizes']

            features_configuration[feat] = settings

        return features_configuration

    @classmethod
    def get_features_crf_position(cls, features):
        positions = []
        for feat in features:
            pos = cls.FEATURE_MAPPING[feat]['crf_position']
            if pos:
                positions.append((feat,pos))

        return positions

    def determine_hidden_layer_size(self):
        size = 0

        for desc in self.features_to_use.values():

            feature = desc['name']
            c_window = desc['use_cw']
            feat_dim = self.get_features_dimensions()[feature]

            size += self.n_window * feat_dim

        return size

    def get_features_dimensions(self):
        d = dict()
        d['w2v'] = self.n_emb
        d['pos'] = self.n_pos_emb
        d['ner'] = self.n_ner_emb
        d['sent_nr'] = self.n_sent_nr_emb
        d['tense'] = self.n_tense_emb

        return d

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
        self.train_costs_list = []
        self.train_cross_entropy_list = []
        self.train_errors_list = []
        self.valid_costs_list = []
        self.valid_cross_entropy_list = []
        self.valid_errors_list = []
        self.precision_list = []
        self.recall_list = []
        self.f1_score_list = []

        # w2v no-convolution no-max-pooling
        self.epoch_l2_w2v_nc_nmp_weight_list = []

        # w2v convolution and max-pooling
        self.epoch_l2_w2v_c_mp_filters_list = []
        self.epoch_l2_w2v_c_mp_weight_list = []

        # w2v convolution no-max-pooling
        self.epoch_l2_w2v_c_nmp_filters_list = []
        self.epoch_l2_w2v_c_nmp_weight_list = []

        # pos convolution and max-pooling
        self.epoch_l2_pos_filters_list = []
        self.epoch_l2_pos_weight_list = []

        # ner convolution and max-pooling
        self.epoch_l2_ner_filters_list = []
        self.epoch_l2_ner_weight_list = []

        # sent_nr convolution and max-pooling
        self.epoch_l2_sent_nr_weight_list = []
        self.epoch_l2_sent_nr_filters_list = []

        # tense convolution and max-pooling
        self.epoch_l2_tense_weight_list = []
        self.epoch_l2_tense_filters_list = []

        # w2 dense layer
        self.epoch_l2_w2_list = []

        return

    def train(self, static, batch_size, **kwargs):
        self.initialize_parameters(static)

        if not batch_size:
            # train SGD
            minibatch_size = 1
        else:
            minibatch_size = batch_size

        if self.n_hidden > 0:
            self.logger.info('Using two hidden layer MLP architecture with window: %d hidden_layer: %d batch_size: %d learning_rate_train: %f learning_rate_tune: %f' % (
            self.n_window, self.n_hidden, minibatch_size, kwargs['learning_rate_train'], kwargs['learning_rate_tune']))
            self.forward_function = self.forward_two_hidden_layer
            self.hidden_activations = self.hidden_activations_two_hidden_layer
        else:
            self.logger.info('Using one hidden layer MLP architecture with window: %d batch_size: %d learning_rate_train: %f learning_rate_tune: %f' % (
                self.n_window, minibatch_size, kwargs['learning_rate_train'], kwargs['learning_rate_tune']))
            self.forward_function = self.forward_one_hidden_layer
            self.hidden_activations = self.hidden_activations_one_hidden_layer

        with self.graph.as_default():
            self.logger.info('Trainable parameters: ' + ', '.join([param.name for param in tf.trainable_variables()]))
            self.logger.info('Training-params: ' + ','.join([param.name for param in self.training_params]))
            self.logger.info('Tuning-params: ' + ','.join([param.name for param in self.fine_tuning_params]))
            self.logger.info('Regularizable parameters: ' + ', '.join([param.name for param in self.regularizables]))

        self.logger.info('Hidden layer size: %d' % self.hidden_layer_size)
        self._train_graph(minibatch_size, **kwargs)

    def initialize_parameters(self, static):
        with self.graph.as_default():

            if self.using_w2v_feature():
                self.w1_w2v = tf.Variable(initial_value=self.pretrained_embeddings, dtype=tf.float32,
                                          trainable=not static,
                                          name='w1_w2v')

                self.regularizables.append(self.w1_w2v)

                if 'w2v' in self.tuning_param_names:
                    self.fine_tuning_params.append(self.w1_w2v)
                elif 'w2v' in self.training_param_names:
                    self.training_params.append(self.w1_w2v)
                else:
                    raise Exception

            if self.using_pos_feature():
                self.w1_pos = tf.Variable(initial_value=self.pos_embeddings, dtype=tf.float32,
                                          trainable=not static, name='w1_pos')

                self.regularizables.append(self.w1_pos)

                if 'pos' in self.tuning_param_names:
                    self.fine_tuning_params.append(self.w1_pos)
                elif 'pos' in self.training_param_names:
                    self.training_params.append(self.w1_pos)
                else:
                    raise Exception

            if self.using_ner_feature():
                self.w1_ner = tf.Variable(initial_value=self.ner_embeddings, dtype=tf.float32,
                                          trainable=not static, name='w1_ner')

                self.regularizables.append(self.w1_ner)

                if 'ner' in self.tuning_param_names:
                    self.fine_tuning_params.append(self.w1_ner)
                elif 'ner' in self.training_param_names:
                    self.training_params.append(self.w1_ner)
                else:
                    raise Exception

            if self.using_sent_nr_feature():
                self.w1_sent_nr = tf.Variable(initial_value=self.sent_nr_embeddings, dtype=tf.float32,
                                              trainable=not static, name='w1_sent_nr')

                self.regularizables.append(self.w1_sent_nr)

                if 'sent_nr' in self.tuning_param_names:
                    self.fine_tuning_params.append(self.w1_sent_nr)
                elif 'sent_nr' in self.training_param_names:
                    self.training_params.append(self.w1_sent_nr)
                else:
                    raise Exception

            if self.using_tense_feature():
                self.w1_tense = tf.Variable(initial_value=self.tense_embeddings, dtype=tf.float32,
                                            trainable=not static, name='w1_tense')

                self.regularizables.append(self.w1_tense)
                if 'tense' in self.tuning_param_names:
                    self.fine_tuning_params.append(self.w1_tense)
                elif 'tense' in self.training_param_names:
                    self.training_params.append(self.w1_tense)
                else:
                    raise Exception

            if self.n_hidden > 0:
                # two hidden layers
                self.w2 = tf.Variable(tf.truncated_normal(shape=[self.hidden_layer_size, self.n_hidden], stddev=0.1, seed=tf.set_random_seed(1234)),
                                      name='w2')
                self.b2 = tf.Variable(tf.constant(0.1, shape=[self.n_hidden]), name='b2')

                self.w3 = tf.Variable(initial_value=tf.truncated_normal(shape=[self.n_hidden, self.n_out], stddev=0.1, seed=tf.set_random_seed(1234)),
                                      name='w3')
                self.b3 = tf.Variable(tf.constant(0.1, shape=[self.n_out]), name='b3')

                self.regularizables.append(self.w3)
                self.training_params.append(self.w3)
                self.training_params.append(self.b3)
            else:
                # one hidden layer
                self.w2 = tf.Variable(tf.truncated_normal(shape=[self.hidden_layer_size, self.n_out], stddev=0.1, seed=tf.set_random_seed(1234)),
                                      name='w2')
                self.b2 = tf.Variable(tf.constant(0.1, shape=[self.n_out]), name='b2')

            self.regularizables.append(self.w2)
            self.training_params.append(self.w2)
            self.training_params.append(self.b2)

        return

    def perform_embeddings_lookup(self, w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs):
        embeddings = []

        if self.using_w2v_feature():
            w1_w2v_x = tf.nn.embedding_lookup(self.w1_w2v, w2v_idxs)
            w1_w2v_x_r = tf.reshape(w1_w2v_x, shape=[-1, self.n_window * self.n_emb])
            embeddings.append(w1_w2v_x_r)

        if self.using_pos_feature():
            w1_pos_x = tf.nn.embedding_lookup(self.w1_pos, pos_idxs)
            w1_pos_x_r = tf.reshape(w1_pos_x, shape=[-1, self.n_window * self.n_pos_emb])
            embeddings.append(w1_pos_x_r)

        if self.using_ner_feature():
            w1_ner_x = tf.nn.embedding_lookup(self.w1_ner, ner_idxs)
            w1_ner_x_r = tf.reshape(w1_ner_x, shape=[-1, self.n_window * self.n_ner_emb])
            embeddings.append(w1_ner_x_r)

        if self.using_sent_nr_feature():
            w1_sent_nr_x = tf.nn.embedding_lookup(self.w1_sent_nr, sent_nr_idxs)
            w1_sent_nr_x_r = tf.reshape(w1_sent_nr_x, shape=[-1, self.n_window * self.n_sent_nr_emb])
            embeddings.append(w1_sent_nr_x_r)

        if self.using_tense_feature():
            w1_tense_x = tf.nn.embedding_lookup(self.w1_tense, tense_idxs)
            w1_tense_x_r = tf.reshape(w1_tense_x, shape=[-1, self.n_window * self.n_tense_emb])
            embeddings.append(w1_tense_x_r)

        embeddings_concat = tf.concat(concat_dim=1, values=embeddings)

        return embeddings_concat

    def hidden_activations_one_hidden_layer(self, w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob):
        # lookup and reshape
        embeddings_concat = self.perform_embeddings_lookup(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs)

        h = tf.tanh(embeddings_concat)

        h_dropout = tf.nn.dropout(h, keep_prob)

        return h_dropout

    def forward_one_hidden_layer(self, w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob):
        h = self.hidden_activations_one_hidden_layer(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob)

        return tf.nn.xw_plus_b(h, self.w2, self.b2)

    def hidden_activations_two_hidden_layer(self, w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob):
        # lookup and reshape
        embeddings_concat = self.perform_embeddings_lookup(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs)

        h1 = tf.tanh(embeddings_concat)

        h1_dropout = tf.nn.dropout(h1, keep_prob)

        h2 = tf.tanh(tf.nn.xw_plus_b(h1_dropout, self.w2, self.b2))

        h2_dropout = tf.nn.dropout(h2, keep_prob)

        return h2_dropout

    def forward_two_hidden_layer(self, w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob):
        h = self.hidden_activations_two_hidden_layer(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob)

        return tf.nn.xw_plus_b(h, self.w3, self.b3)

    def compute_predictions(self, out_logits):
        return tf.to_int32(tf.argmax(tf.nn.softmax(out_logits), 1))

    def _train_graph(self, minibatch_size, max_epochs,
                     learning_rate_train, learning_rate_tune, lr_decay,
                     plot,
                     **kwargs):

        self.learning_rate_train = learning_rate_train
        self.learning_rate_tune = learning_rate_tune
        self.minibatch_size = minibatch_size
        self.lr_decay = lr_decay

        with self.graph.as_default():

            # tf.trainable_variables()

            w2v_idxs = tf.placeholder(tf.int32, shape=[None, self.n_window], name='w2v_idxs')
            pos_idxs = tf.placeholder(tf.int32, shape=[None, self.n_window], name='pos_idxs')
            ner_idxs = tf.placeholder(tf.int32, shape=[None, self.n_window], name='ner_idxs')
            sent_nr_idxs = tf.placeholder(tf.int32, shape=[None, self.n_window], name='sent_nr_idxs')
            tense_idxs = tf.placeholder(tf.int32, shape=[None, self.n_window], name='tense_idxs')

            labels = tf.placeholder(tf.int32, name='labels')
            keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            out_logits = self.forward_function(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob)

            # l2 = tf.add_n([tf.nn.l2_loss(param) for param in self.regularizables])

            cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_logits, labels=labels))
            # cost = cross_entropy + alpha_l2 * l2
            cost = cross_entropy

            predictions = self.compute_predictions(out_logits)
            n_errors = tf.reduce_sum(tf.to_int32(tf.not_equal(predictions, labels)))

            optimizer = self.instantiate_optimizer(learning_rate_tune, learning_rate_train, cost)

        with tf.Session(graph=self.graph) as session:
            session.run(tf.initialize_all_variables())
            n_batches = np.int(np.ceil(self.x_train.shape[0] / float(minibatch_size)))

            early_stopping_cnt_since_last_update = 0
            early_stopping_min_validation_cost = np.inf
            early_stopping_min_iteration = None
            model_update = None

            for epoch_ix in range(max_epochs):
                start = time.time()

                if self.early_stopping_threshold is not None:
                    if early_stopping_cnt_since_last_update >= self.early_stopping_threshold:
                        assert early_stopping_min_iteration is not None
                        self.logger.info('Training early stopped at iteration %d' % early_stopping_min_iteration)
                        break

                train_cost = 0
                train_xentropy = 0
                train_errors = 0
                for batch_ix in range(n_batches):
                    feed_dict = self.get_feed_dict(batch_ix, minibatch_size,
                                                   w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs,
                                                   labels, keep_prob, keep_prob_value=.5, dataset='train')
                    _, cost_val, xentropy, errors = session.run([optimizer, cost, cross_entropy, n_errors], feed_dict=feed_dict)
                    train_cost += cost_val
                    train_xentropy += xentropy
                    train_errors += errors

                feed_dict = self.get_feed_dict(None, minibatch_size,
                                               w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs,
                                               labels, keep_prob, keep_prob_value=1., dataset='valid')
                valid_cost, valid_xentropy, pred, valid_errors = session.run([cost, cross_entropy, predictions, n_errors], feed_dict=feed_dict)

                precision, recall, f1_score = self.compute_scores(self.y_valid, pred)

                if plot:
                    l2_w1_w2v, l2_w1_pos, l2_w1_ner, l2_w1_sent_nr, l2_w1_tense, l2_w2, = \
                        self.compute_parameters_sum()

                    self.update_monitoring_lists(train_cost, train_xentropy, train_errors,
                                                 valid_cost, valid_xentropy, valid_errors,
                                                 l2_w1_w2v, l2_w1_pos, l2_w1_ner, l2_w1_sent_nr, l2_w1_tense, l2_w2,
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

                self.logger.info('epoch: %d train_cost: %f train_errors: %d valid_cost: %f valid_errors: %d F1: %f upd: %s took: %f' \
                      % (epoch_ix, train_cost, train_errors, valid_cost, valid_errors, f1_score, model_update, time.time() - start))

            self.saver = tf.train.Saver(self.training_params + self.fine_tuning_params)
            self.saver.save(session, self.get_output_path('params.model'))

        if plot:
            self.logger.info('Making plots')
            self.make_plots()

        if self.pickle_lists:
            self.logger.info('Pickling lists')
            self.perform_pickle_lists()

        return True

    def perform_pickle_lists(self):
        """
        This is to make some later plots.
        """
        output_filename = '_'.join(
            [str(self.n_window), str(self.minibatch_size), 'lrtrain', str(self.learning_rate_train), 'lrtune',
             str(self.learning_rate_tune), str(self.n_hidden), str(self.lr_decay), 'filters', 'None', 'regions', 'None'])

        cPickle.dump(self.valid_costs_list,
                     open(self.get_output_path('valid_cost_list-' + output_filename + '.p'), 'wb'))
        cPickle.dump(self.valid_cross_entropy_list,
                     open(self.get_output_path('valid_cross_entropy_list-' + output_filename + '.p'), 'wb'))
        cPickle.dump(self.train_costs_list,
                     open(self.get_output_path('train_cost_list-' + output_filename + '.p'), 'wb'))
        cPickle.dump(self.train_cross_entropy_list,
                     open(self.get_output_path('train_cross_entropy_list-' + output_filename + '.p'), 'wb'))
        cPickle.dump(self.f1_score_list,
                     open(self.get_output_path('valid_f1_score_list-' + output_filename + '.p'), 'wb'))

        return True

    def get_feed_dict(self, batch_ix, minibatch_size,
                      w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs,
                      labels, keep_prob, keep_prob_value,
                      dataset):

        feed_dict = dict()

        if dataset == 'train':
            x_w2v = self.x_train
            x_pos = self.train_pos_feats
            x_ner = self.train_ner_feats
            x_sent_nr = self.train_sent_nr_feats
            x_tense = self.train_tense_feats
            y = self.y_train
        elif dataset == 'valid':
            x_w2v = self.x_valid
            x_pos = self.valid_pos_feats
            x_ner = self.valid_ner_feats
            x_sent_nr = self.valid_sent_nr_feats
            x_tense = self.valid_tense_feats
            y = self.y_valid
        elif dataset == 'test':
            x_w2v = self.x_test
            x_pos = self.test_pos_feats
            x_ner = self.test_ner_feats
            x_sent_nr = self.test_sent_nr_feats
            x_tense = self.test_tense_feats
            y = self.y_test
        else:
            raise Exception('Invalid param for dataset')

        if self.using_w2v_feature():
            feed_dict[w2v_idxs] = x_w2v[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else x_w2v

        if self.using_pos_feature():
            feed_dict[pos_idxs] = x_pos[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else x_pos

        if self.using_ner_feature():
            feed_dict[ner_idxs] = x_ner[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else x_ner

        if self.using_sent_nr_feature():
            feed_dict[sent_nr_idxs] = x_sent_nr[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else x_sent_nr

        if self.using_tense_feature():
            feed_dict[tense_idxs] = x_tense[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else x_tense

        assert feed_dict.__len__() > 0

        if labels is not None:
            feed_dict[labels] = y[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else y

        feed_dict[keep_prob] = keep_prob_value

        return feed_dict

    def compute_parameters_sum(self):
        w1_w2v_sum = 0
        w1_pos_sum = 0
        w1_ner_sum = 0
        w1_sent_nr_sum = 0
        w1_tense_sum = 0

        if self.using_w2v_feature():
            w1_w2v_sum = tf.reduce_sum(tf.square(self.w1_w2v)).eval()

        if self.using_pos_feature():
            w1_pos_sum = tf.reduce_sum(tf.square(self.w1_pos)).eval()

        if self.using_ner_feature():
            w1_ner_sum = tf.reduce_sum(tf.square(self.w1_ner)).eval()

        if self.using_sent_nr_feature():
            w1_sent_nr_sum = tf.reduce_sum(tf.square(self.w1_sent_nr)).eval()

        if self.using_tense_feature():
            w1_tense_sum = tf.reduce_sum(tf.square(self.w1_tense)).eval()

        w2_sum = tf.reduce_sum(tf.square(self.w2)).eval()

        return w1_w2v_sum, w1_pos_sum, w1_ner_sum, w1_sent_nr_sum, w1_tense_sum, w2_sum

    def update_monitoring_lists(self, train_cost, train_xentropy, train_errors,
                                valid_cost, valid_xentropy, valid_errors,
                                l2_w1_w2v, l2_w1_pos, l2_w1_ner, l2_w1_sent_nr, l2_w1_tense, l2_w2,
                                precision, recall, f1_score):

        # train stats
        self.train_costs_list.append(train_cost)
        self.train_cross_entropy_list.append(train_xentropy)
        self.train_errors_list.append(train_errors)

        # valid stats
        self.valid_costs_list.append(valid_cost)
        self.valid_cross_entropy_list.append(valid_xentropy)
        self.valid_errors_list.append(valid_errors)

        # embeddings sum
        self.epoch_l2_w2v_c_mp_weight_list.append(l2_w1_w2v)
        self.epoch_l2_pos_weight_list.append(l2_w1_pos)
        self.epoch_l2_ner_weight_list.append(l2_w1_ner)
        self.epoch_l2_sent_nr_weight_list.append(l2_w1_sent_nr)
        self.epoch_l2_tense_weight_list.append(l2_w1_tense)
        self.epoch_l2_w2_list.append(l2_w2)

        # scores
        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.f1_score_list.append(f1_score)

        return

    def make_plots(self):
        actual_time = str(time.time())

        self.plot_training_cost_and_error(self.train_costs_list, self.train_errors_list, self.valid_costs_list,
                                          self.valid_errors_list, actual_time)

        self.plot_scores(self.precision_list, self.recall_list, self.f1_score_list, actual_time)

        plot_data_dict = {
            'w2v_emb': self.epoch_l2_w2v_c_mp_weight_list,
            'pos_emb': self.epoch_l2_pos_weight_list,
            'ner_emb': self.epoch_l2_ner_weight_list,
            'sent_nr_emb': self.epoch_l2_sent_nr_weight_list,
            'tense_emb': self.epoch_l2_tense_weight_list,
            'w2': self.epoch_l2_w2_list
        }
        self.plot_penalties_general(plot_data_dict, actual_time=actual_time)

        self.plot_cross_entropies(self.train_cross_entropy_list, self.valid_cross_entropy_list, actual_time)

        return

    def instantiate_optimizer(self, learning_rate_tune, learning_rate_train, cost):
        # optimizer = tf.train.AdamOptimizer(1e-3)
        optimizer_fine_tune = tf.train.AdagradOptimizer(learning_rate=learning_rate_tune)
        optimizer_train = tf.train.AdagradOptimizer(learning_rate=learning_rate_train)
        grads = tf.gradients(cost, self.fine_tuning_params + self.training_params)
        fine_tuning_grads = grads[:len(self.fine_tuning_params)]
        training_grads = grads[-len(self.training_params):]
        fine_tune_op = optimizer_fine_tune.apply_gradients(zip(fine_tuning_grads, self.fine_tuning_params))
        train_op = optimizer_train.apply_gradients(zip(training_grads, self.training_params))
        optimizer = tf.group(fine_tune_op, train_op)

        return optimizer

    def compute_scores(self, true_values, predictions):
        results = Metrics.compute_all_metrics(y_true=true_values, y_pred=predictions, average='macro')
        f1_score = results['f1_score']
        precision = results['precision']
        recall = results['recall']

        return precision, recall, f1_score

    def predict(self, on_training_set=False, on_validation_set=False, on_testing_set=False, **kwargs):

        results = dict()

        if on_training_set:
            dataset = 'train'
            y_test = self.y_train
        elif on_validation_set:
            dataset = 'valid'
            y_test = self.y_valid
        elif on_testing_set:
            dataset = 'test'
            y_test = self.y_test
        else:
            raise Exception

        with self.graph.as_default():

            # Input data
            # idxs = tf.placeholder(tf.int32)
            # out_logits = self.compute_output_layer_logits(idxs)

            w2v_idxs = self.graph.get_tensor_by_name(name='w2v_idxs:0')
            pos_idxs = self.graph.get_tensor_by_name(name='pos_idxs:0')
            ner_idxs = self.graph.get_tensor_by_name(name='ner_idxs:0')
            sent_nr_idxs = self.graph.get_tensor_by_name(name='sent_nr_idxs:0')
            tense_idxs = self.graph.get_tensor_by_name(name='tense_idxs:0')
            keep_prob = self.graph.get_tensor_by_name(name='dropout_keep_prob:0')
            # labels = self.graph.get_tensor_by_name(name='labels:0')

            out_logits = self.forward_function(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob)

            predictions = self.compute_predictions(out_logits)

        with tf.Session(graph=self.graph) as session:
            # init.run()

            self.saver.restore(session, self.get_output_path('params.model'))

            feed_dict = self.get_feed_dict(None, None,
                                           w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs,
                                           None, keep_prob, keep_prob_value=1., dataset=dataset)
            pred = session.run(predictions, feed_dict=feed_dict)

        results['flat_trues'] = y_test
        results['flat_predictions'] = pred

        return results

    def using_w2v_feature(self):
        return self.using_feature(feature='w2v', convolve=None, max_pool=None)

    def using_pos_feature(self):
        return self.using_feature(feature='pos', convolve=None, max_pool=None)

    def using_ner_feature(self):
        return self.using_feature(feature='ner', convolve=None, max_pool=None)

    def using_sent_nr_feature(self):
        return self.using_feature(feature='sent_nr', convolve=None, max_pool=None)

    def using_tense_feature(self):
        return self.using_feature(feature='tense', convolve=None, max_pool=None)

    def using_feature(self, feature, convolve, max_pool):
        result = None

        if convolve is None and max_pool is None:
            result = any([feat['name'] == feature for feat in self.features_to_use.values()])

        assert result is not None

        return result

    def to_string(self):
        print '[Tensorflow] Multi-features feed-forward neural network'

    def get_hidden_activations(self, on_training_set, on_validation_set, on_testing_set, **kwargs):

        hidden_activations = None

        if on_training_set:
            dataset = 'train'
        elif on_validation_set:
            dataset = 'valid'
        elif on_testing_set:
            dataset = 'test'
        else:
            raise Exception

        with self.graph.as_default():
            w2v_idxs = self.graph.get_tensor_by_name(name='w2v_idxs:0')
            pos_idxs = self.graph.get_tensor_by_name(name='pos_idxs:0')
            ner_idxs = self.graph.get_tensor_by_name(name='ner_idxs:0')
            sent_nr_idxs = self.graph.get_tensor_by_name(name='sent_nr_idxs:0')
            tense_idxs = self.graph.get_tensor_by_name(name='tense_idxs:0')
            keep_prob = self.graph.get_tensor_by_name(name='dropout_keep_prob:0')

            h = self.hidden_activations(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob)

        with tf.Session(graph=self.graph) as session:
            # init.run()

            self.saver.restore(session, self.get_output_path('params.model'))

            feed_dict = self.get_feed_dict(None, None,
                                           w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs,
                                           None, keep_prob, keep_prob_value=1., dataset=dataset)
            hidden_activations = session.run(h, feed_dict=feed_dict)

        return hidden_activations


    def get_output_logits(self, on_training_set, on_validation_set, on_testing_set, **kwargs):
        return []