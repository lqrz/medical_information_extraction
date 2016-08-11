__author__ = 'lqrz'

from SOTA.neural_network.A_neural_network import A_neural_network
from utils.metrics import Metrics

import time
import tensorflow as tf
import numpy as np

class Convolutional_Neural_Net(A_neural_network):

    def __init__(self, pos_embeddings, ner_embeddings, sent_nr_embeddings, tense_embeddings, cnn_features,
                 train_pos_feats, valid_pos_feats, test_pos_feats,
                 train_ner_feats, valid_ner_feats, test_ner_feats,
                 train_sent_nr_feats, valid_sent_nr_feats, test_sent_nr_feats,
                 train_tense_feats, valid_tense_feats, test_tense_feats,
                 training_param_names, tuning_param_names,
                 **kwargs):

        super(Convolutional_Neural_Net, self).__init__(**kwargs)

        self.graph = tf.Graph()

        # parameters
        self.w1_w2v = None
        self.w1_pos = None
        self.w1_ner = None
        self.w1_sent_nr = None
        self.w1_tense = None
        self.w2 = None
        self.b2 = None

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

        self.n_hidden = self.determine_hidden_layer_size()

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
            convolve = desc['convolve']
            max_pool = desc['max_pool']
            c_window = desc['use_cw']
            region_sizes = desc['region_sizes']
            n_regions = desc['nr_region_sizes']
            nr_filters = desc['nr_filters']
            feat_dim = self.get_features_dimensions()[feature]

            feat_extended = '_'.join([feature, 'c' if convolve else 'nc', 'm' if max_pool else 'nm'])

            if convolve:
                window = 1
                n_filters = nr_filters
                # n_regions = self.determine_nr_region_sizes(feat_extended)
                filter_windows = np.sum([self.n_window - rs + 1 for rs in region_sizes])
                filter_window_width = self.determine_resulting_width(feat_extended, feat_dim)
            else:
                window = self.n_window
                n_regions = 1
                n_filters = 1
                filter_window_width = feat_dim
                filter_windows = 1

            if max_pool:
                feat_dimension = 1
                filter_windows = n_regions
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

        with self.graph.as_default():
            print 'Trainable parameters: ' + ', '.join([param.name for param in tf.trainable_variables()])
            print 'Regularizable parameters: ' + ', '.join([param.name for param in self.regularizables])

        print 'Hidden layer size: %d' % self.n_hidden
        self._train_graph(minibatch_size, **kwargs)

    def initialize_parameters(self, static):
        with self.graph.as_default():

            if self.using_w2v_convolution_maxpool_feature():
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

            if self.using_pos_convolution_maxpool_feature():
                self.w1_pos = tf.Variable(initial_value=self.pos_embeddings, dtype=tf.float32,
                                          trainable=not static, name='w1_pos')

                self.regularizables.append(self.w1_pos)
                if 'pos' in self.tuning_param_names:
                    self.fine_tuning_params.append(self.w1_pos)
                elif 'pos' in self.training_param_names:
                    self.training_params.append(self.w1_pos)
                else:
                    raise Exception

            if self.using_ner_convolution_maxpool_feature():
                self.w1_ner = tf.Variable(initial_value=self.ner_embeddings, dtype=tf.float32,
                                          trainable=not static, name='w1_ner')

                self.regularizables.append(self.w1_ner)
                if 'ner' in self.tuning_param_names:
                    self.fine_tuning_params.append(self.w1_ner)
                elif 'ner' in self.training_param_names:
                    self.training_params.append(self.w1_ner)
                else:
                    raise Exception

            if self.using_sent_nr_convolution_maxpool_feature():
                self.w1_sent_nr = tf.Variable(initial_value=self.sent_nr_embeddings, dtype=tf.float32,
                                              trainable=not static, name='w1_sent_nr')

                self.regularizables.append(self.w1_sent_nr)
                if 'sent_nr' in self.tuning_param_names:
                    self.fine_tuning_params.append(self.w1_sent_nr)
                elif 'sent_nr' in self.training_param_names:
                    self.training_params.append(self.w1_sent_nr)
                else:
                    raise Exception

            if self.using_tense_convolution_maxpool_feature():
                self.w1_tense = tf.Variable(initial_value=self.tense_embeddings, dtype=tf.float32,
                                            trainable=not static, name='w1_tense')

                self.regularizables.append(self.w1_tense)
                if 'tense' in self.tuning_param_names:
                    self.fine_tuning_params.append(self.w1_tense)
                elif 'tense' in self.training_param_names:
                    self.training_params.append(self.w1_tense)
                else:
                    raise Exception


            self.w2 = tf.Variable(tf.truncated_normal(shape=[self.n_hidden, self.n_out], stddev=0.1), name='w2')
            self.b2 = tf.Variable(tf.constant(0.1, shape=[self.n_out]), name='b2')

            self.regularizables.append(self.w2)
            self.training_params.append(self.w2)
            self.training_params.append(self.b2)

            if self.using_w2v_convolution_maxpool_feature():
                self.initialise_filters(region_sizes=self.features_to_use['w2v_c_m']['region_sizes'],
                                        n_filters=self.features_to_use['w2v_c_m']['nr_filters'],
                                        embedding_size=self.n_emb,
                                        name='w2v_c_m',
                                        filters_list=self.w2v_filters_weights,
                                        bias_list=self.w2v_filters_bias)

            if self.using_pos_convolution_maxpool_feature():
                self.initialise_filters(region_sizes=self.features_to_use['pos_c_m']['region_sizes'],
                                        n_filters=self.features_to_use['pos_c_m']['nr_filters'],
                                        embedding_size=self.n_pos_emb,
                                        name='pos_c_m',
                                        filters_list=self.pos_filters_weights,
                                        bias_list=self.pos_filters_bias)

            if self.using_ner_convolution_maxpool_feature():
                self.initialise_filters(region_sizes=self.features_to_use['ner_c_m']['region_sizes'],
                                        n_filters=self.features_to_use['ner_c_m']['nr_filters'],
                                        embedding_size=self.n_ner_emb,
                                        name='ner_c_m',
                                        filters_list=self.ner_filters_weights,
                                        bias_list=self.ner_filters_bias)

            if self.using_sent_nr_convolution_maxpool_feature():
                self.initialise_filters(region_sizes=self.features_to_use['sent_nr_c_m']['region_sizes'],
                                        n_filters=self.features_to_use['sent_nr_c_m']['nr_filters'],
                                        embedding_size=self.n_sent_nr_emb,
                                        name='sent_nr_c_m',
                                        filters_list=self.sent_nr_filters_weights,
                                        bias_list=self.sent_nr_filters_bias)

            if self.using_tense_convolution_maxpool_feature():
                self.initialise_filters(region_sizes=self.features_to_use['tense_c_m']['region_sizes'],
                                        n_filters=self.features_to_use['tense_c_m']['nr_filters'],
                                        embedding_size=self.n_tense_emb,
                                        name='tense_c_m',
                                        filters_list=self.tense_filters_weights,
                                        bias_list=self.tense_filters_bias)

        return

    def initialise_filters(self, region_sizes, n_filters, embedding_size, name, filters_list, bias_list):
        # this is for the w2v convolution
        for i, rs in enumerate(region_sizes):
            filter_shape = [rs, embedding_size, 1, n_filters]
            w_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name=name+'_w_filter_shape_' + str(rs))
            b_filter = tf.Variable(tf.constant(0.1, shape=[n_filters]), name=name+'_b_filter_shape_' + str(rs))
            filters_list.append(w_filter)
            bias_list.append(b_filter)

            self.training_params.append(w_filter)
            self.training_params.append(b_filter)

        return

    def convolve(self, input, filter_weights, filter_bias, region_sizes):

        pooled_out = []

        for filter_w, filter_b, filter_size in zip(filter_weights, filter_bias, region_sizes):
            # filter_weight = self.w2v_filters_weights[0]
            # filter_bias = self.w2v_filters_bias[0]
            # filter_size = self.region_sizes[0]
            # "VALID" padding means that we slide the filter over our sentence without padding the edges
            conv = tf.nn.conv2d(input=input, filter=filter_w, strides=[1, 1, 1, 1], padding='VALID')
            a = tf.nn.bias_add(conv, filter_b)
            h = tf.nn.relu(a)
            pooled = tf.nn.max_pool(value=h, ksize=[1, self.n_window - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID')
            pooled_out.append(pooled)

        return pooled_out

    def perform_convolutions(self, w2v, pos, ner, sent_nr, tense):

        pooled_out = []
        n_total_filters = 0

        if self.using_w2v_convolution_maxpool_feature():
            pooled_out.extend(
                self.convolve(w2v,
                              self.w2v_filters_weights,
                              self.w2v_filters_bias,
                              self.features_to_use['w2v_c_m']['region_sizes']
                              )
            )

            n_total_filters += self.features_to_use['w2v_c_m']['nr_filters'] * \
                               self.features_to_use['w2v_c_m']['nr_region_sizes']

        if self.using_pos_convolution_maxpool_feature():
            pooled_out.extend(
                self.convolve(pos,
                              self.pos_filters_weights,
                              self.pos_filters_bias,
                              self.features_to_use['pos_c_m']['region_sizes']
                              )
            )

            n_total_filters += self.features_to_use['pos_c_m']['nr_filters'] * \
                               self.features_to_use['pos_c_m']['nr_region_sizes']

        if self.using_ner_convolution_maxpool_feature():
            pooled_out.extend(
                self.convolve(ner,
                              self.ner_filters_weights,
                              self.ner_filters_bias,
                              self.features_to_use['ner_c_m']['region_sizes']
                              )
            )

            n_total_filters += self.features_to_use['ner_c_m']['nr_filters'] * \
                               self.features_to_use['ner_c_m']['nr_region_sizes']

        if self.using_sent_nr_convolution_maxpool_feature():
            pooled_out.extend(
                self.convolve(sent_nr,
                              self.sent_nr_filters_weights,
                              self.sent_nr_filters_bias,
                              self.features_to_use['sent_nr_c_m']['region_sizes']
                              )
            )

            n_total_filters += self.features_to_use['sent_nr_c_m']['nr_filters'] * \
                               self.features_to_use['sent_nr_c_m']['nr_region_sizes']

        if self.using_tense_convolution_maxpool_feature():
            pooled_out.extend(
                self.convolve(tense,
                              self.tense_filters_weights,
                              self.tense_filters_bias,
                              self.features_to_use['tense_c_m']['region_sizes']
                              )
            )

            n_total_filters += self.features_to_use['tense_c_m']['nr_filters'] * \
                               self.features_to_use['tense_c_m']['nr_region_sizes']

        assert n_total_filters > 0 and pooled_out.__len__() > 0

        h_pooled = tf.concat(concat_dim=3, values=pooled_out)
        h_pooled_flat = tf.reshape(h_pooled, shape=[-1, n_total_filters])

        return h_pooled_flat

    def perform_embeddings_lookup(self, w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs):
        w1_x_expanded = None
        w1_pos_expanded = None
        w1_ner_expanded = None
        w1_sent_nr_expanded = None
        w1_tense_expanded = None

        if self.using_w2v_convolution_maxpool_feature():
            w1_w2v_x = tf.nn.embedding_lookup(self.w1_w2v, w2v_idxs)
            w1_x_expanded = tf.expand_dims(w1_w2v_x, -1)

        if self.using_pos_convolution_maxpool_feature():
            w1_pos_x = tf.nn.embedding_lookup(self.w1_pos, pos_idxs)
            w1_pos_expanded = tf.expand_dims(w1_pos_x, -1)

        if self.using_ner_convolution_maxpool_feature():
            w1_ner_x = tf.nn.embedding_lookup(self.w1_ner, ner_idxs)
            w1_ner_expanded = tf.expand_dims(w1_ner_x, -1)

        if self.using_sent_nr_convolution_maxpool_feature():
            w1_sent_nr_x = tf.nn.embedding_lookup(self.w1_sent_nr, sent_nr_idxs)
            w1_sent_nr_expanded = tf.expand_dims(w1_sent_nr_x, -1)

        if self.using_tense_convolution_maxpool_feature():
            w1_tense_x = tf.nn.embedding_lookup(self.w1_tense, tense_idxs)
            w1_tense_expanded = tf.expand_dims(w1_tense_x, -1)

        return w1_x_expanded, w1_pos_expanded, w1_ner_expanded, w1_sent_nr_expanded, w1_tense_expanded

    def hidden_activations(self, w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs):

        w1_x_expanded, w1_pos_expanded, w1_ner_expanded, w1_sent_nr_expanded, w1_tense_expanded = \
            self.perform_embeddings_lookup(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs)

        return self.perform_convolutions(w1_x_expanded, w1_pos_expanded, w1_ner_expanded,
                                                  w1_sent_nr_expanded, w1_tense_expanded)

    def forward_pass(self, w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob):

        h_pooled_flat = self.hidden_activations(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs)

        h_dropout = tf.nn.dropout(h_pooled_flat, keep_prob)

        out_logits = tf.nn.xw_plus_b(h_dropout, self.w2, self.b2)

        return out_logits

    def compute_predictions(self, out_logits):
        return tf.to_int32(tf.argmax(out_logits, 1))

    def _train_graph(self, minibatch_size, max_epochs,
                     learning_rate_train, learning_rate_tune, lr_decay,
                     plot, alpha_l2=0.001,
                     **kwargs):

        with self.graph.as_default():
            tf.set_random_seed(1234)

            # tf.trainable_variables()

            w2v_idxs = tf.placeholder(tf.int32, shape=[None, self.n_window], name='w2v_idxs')
            pos_idxs = tf.placeholder(tf.int32, shape=[None, self.n_window], name='pos_idxs')
            ner_idxs = tf.placeholder(tf.int32, shape=[None, self.n_window], name='ner_idxs')
            sent_nr_idxs = tf.placeholder(tf.int32, shape=[None, self.n_window], name='sent_nr_idxs')
            tense_idxs = tf.placeholder(tf.int32, shape=[None, self.n_window], name='tense_idxs')

            labels = tf.placeholder(tf.int32, name='labels')
            keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            out_logits = self.forward_pass(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob)

            l2 = tf.add_n([tf.nn.l2_loss(param) for param in self.regularizables])

            cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_logits, labels=labels))
            cost = cross_entropy + alpha_l2 * l2

            predictions = self.compute_predictions(out_logits)
            n_errors = tf.reduce_sum(tf.to_int32(tf.not_equal(predictions, labels)))

            optimizer = self.instantiate_optimizer(learning_rate_tune, learning_rate_train, cost)

        with tf.Session(graph=self.graph) as session:
            session.run(tf.initialize_all_variables())
            n_batches = np.int(np.ceil(self.x_train.shape[0] / minibatch_size))
            for epoch_ix in range(max_epochs):
                start = time.time()
                train_cost = 0
                train_xentropy = 0
                train_errors = 0
                for batch_ix in range(n_batches):
                    feed_dict = self.get_feed_dict(batch_ix, minibatch_size,
                                                   w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs,
                                                   labels, keep_prob, keep_prob_val=.5, dataset='train')
                    _, cost_val, xentropy, errors = session.run([optimizer, cost, cross_entropy, n_errors], feed_dict=feed_dict)
                    train_cost += cost_val
                    train_xentropy += xentropy
                    train_errors += errors

                feed_dict = self.get_feed_dict(None, minibatch_size,
                                               w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs,
                                               labels, keep_prob, keep_prob_val=1., dataset='valid')
                valid_cost, valid_xentropy, pred, valid_errors = session.run([cost, cross_entropy, predictions, n_errors], feed_dict=feed_dict)

                precision, recall, f1_score = self.compute_scores(self.y_valid, pred)

                if plot:
                    l2_w1_w2v, l2_w1_pos, l2_w1_ner, l2_w1_sent_nr, l2_w1_tense, l2_w2, \
                    w2v_filters_sum, pos_filters_sum, ner_filters_sum, sent_nr_filters_sum, tense_filters_sum = \
                        self.compute_parameters_sum()

                    self.update_monitoring_lists(train_cost, train_xentropy, train_errors,
                                                 valid_cost, valid_xentropy, valid_errors,
                                                 l2_w1_w2v, l2_w1_pos, l2_w1_ner, l2_w1_sent_nr, l2_w1_tense, l2_w2,
                                                 w2v_filters_sum, pos_filters_sum, ner_filters_sum, sent_nr_filters_sum,
                                                 tense_filters_sum,
                                                 precision, recall, f1_score)

                print 'epoch: %d train_cost: %f train_errors: %d valid_cost: %f valid_errors: %d F1: %f took: %f' \
                      % (
                          epoch_ix, train_cost, train_errors, valid_cost, valid_errors, f1_score, time.time() - start)

            self.saver = tf.train.Saver(self.training_params + self.fine_tuning_params)
            self.saver.save(session, self.get_output_path('params.model'))

        if plot:
            print 'Making plots'
            self.make_plots()

    def get_feed_dict(self, batch_ix, minibatch_size,
                      w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs,
                      labels,
                      keep_prob, keep_prob_val,
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

        if self.using_w2v_convolution_maxpool_feature():
            feed_dict[w2v_idxs] = x_w2v[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else x_w2v

        if self.using_pos_convolution_maxpool_feature():
            feed_dict[pos_idxs] = x_pos[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else x_pos

        if self.using_ner_convolution_maxpool_feature():
            feed_dict[ner_idxs] = x_ner[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else x_ner

        if self.using_sent_nr_convolution_maxpool_feature():
            feed_dict[sent_nr_idxs] = x_sent_nr[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else x_sent_nr

        if self.using_tense_convolution_maxpool_feature():
            feed_dict[tense_idxs] = x_tense[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else x_tense

        assert feed_dict.__len__() > 0

        if labels is not None:
            feed_dict[labels] = y[batch_ix * minibatch_size: (batch_ix + 1) * minibatch_size] \
                if batch_ix is not None else y

        feed_dict[keep_prob] = keep_prob_val

        return feed_dict

    def compute_parameters_sum(self):
        w1_w2v_sum = 0
        w2v_filters_sum = 0

        w1_pos_sum = 0
        pos_filters_sum = 0

        w1_ner_sum = 0
        ner_filters_sum = 0

        w1_sent_nr_sum = 0
        sent_nr_filters_sum = 0

        w1_tense_sum = 0
        tense_filters_sum = 0

        if self.using_w2v_convolution_maxpool_feature():
            w1_w2v_sum = tf.reduce_sum(tf.square(self.w1_w2v)).eval()
            w2v_filters_sum = tf.add_n(map(lambda x: tf.reduce_sum(tf.square(x)), self.w2v_filters_weights)).eval()

        if self.using_pos_convolution_maxpool_feature():
            w1_pos_sum = tf.reduce_sum(tf.square(self.w1_pos)).eval()
            pos_filters_sum = tf.add_n(map(lambda x: tf.reduce_sum(tf.square(x)), self.pos_filters_weights)).eval()

        if self.using_ner_convolution_maxpool_feature():
            w1_ner_sum = tf.reduce_sum(tf.square(self.w1_ner)).eval()
            ner_filters_sum = tf.add_n(map(lambda x: tf.reduce_sum(tf.square(x)), self.ner_filters_weights)).eval()

        if self.using_sent_nr_convolution_maxpool_feature():
            w1_sent_nr_sum = tf.reduce_sum(tf.square(self.w1_sent_nr)).eval()
            sent_nr_filters_sum = tf.add_n(map(lambda x: tf.reduce_sum(tf.square(x)), self.sent_nr_filters_weights)).eval()

        if self.using_tense_convolution_maxpool_feature():
            w1_tense_sum = tf.reduce_sum(tf.square(self.w1_tense)).eval()
            tense_filters_sum = tf.add_n(map(lambda x: tf.reduce_sum(tf.square(x)), self.tense_filters_weights)).eval()

        w2_sum = tf.reduce_sum(tf.square(self.w2)).eval()

        return w1_w2v_sum, w1_pos_sum, w1_ner_sum, w1_sent_nr_sum, w1_tense_sum, w2_sum, \
               w2v_filters_sum, pos_filters_sum, ner_filters_sum, sent_nr_filters_sum, tense_filters_sum

    def update_monitoring_lists(self, train_cost, train_xentropy, train_errors,
                                valid_cost, valid_xentropy, valid_errors,
                                l2_w1_w2v, l2_w1_pos, l2_w1_ner, l2_w1_sent_nr, l2_w1_tense, l2_w2,
                                w2v_filters_sum, pos_filters_sum, ner_filters_sum, sent_nr_filters_sum,
                                tense_filters_sum,
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

        # filters sum
        self.epoch_l2_w2v_c_mp_filters_list.append(w2v_filters_sum)
        self.epoch_l2_pos_filters_list.append(pos_filters_sum)
        self.epoch_l2_ner_filters_list.append(ner_filters_sum)
        self.epoch_l2_sent_nr_filters_list.append(sent_nr_filters_sum)
        self.epoch_l2_tense_filters_list.append(tense_filters_sum)

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
            'w2v_c_mp_filters': self.epoch_l2_w2v_c_mp_filters_list,
            # 'w2v_c_mp_emb': self.epoch_l2_,
            # 'w2v_c_nmp_filters': epoch_l2_w2v_c_nmp_filters_list,
            # 'w2v_c_nmp_emb': epoch_l2_w2v_c_nmp_weight_list,
            'pos_c_mp_filters': self.epoch_l2_pos_filters_list,
            'ner_c_mp_filters': self.epoch_l2_ner_filters_list,
            'sent_nr_c_mp_filters': self.epoch_l2_sent_nr_filters_list,
            'tense_c_mp_filters': self.epoch_l2_tense_filters_list,
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
            w2v_idxs = self.graph.get_tensor_by_name(name='w2v_idxs:0')
            pos_idxs = self.graph.get_tensor_by_name(name='pos_idxs:0')
            ner_idxs = self.graph.get_tensor_by_name(name='ner_idxs:0')
            sent_nr_idxs = self.graph.get_tensor_by_name(name='sent_nr_idxs:0')
            tense_idxs = self.graph.get_tensor_by_name(name='tense_idxs:0')
            keep_prob = self.graph.get_tensor_by_name(name='dropout_keep_prob:0')

            out_logits = self.forward_pass(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob)

            predictions = self.compute_predictions(out_logits)

        with tf.Session(graph=self.graph) as session:
            # init.run()

            self.saver.restore(session, self.get_output_path('params.model'))

            feed_dict = self.get_feed_dict(None, None,
                                           w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs,
                                           None, keep_prob, keep_prob_val=1., dataset=dataset)

            pred = session.run(predictions, feed_dict=feed_dict)

        results['flat_trues'] = y_test
        results['flat_predictions'] = pred

        return results

    def using_w2v_convolution_maxpool_feature(self):
        return self.using_feature(feature='w2v', convolve=True, max_pool=True)

    def using_pos_convolution_maxpool_feature(self):
        return self.using_feature(feature='pos', convolve=True, max_pool=True)

    def using_ner_convolution_maxpool_feature(self):
        return self.using_feature(feature='ner', convolve=True, max_pool=True)

    def using_sent_nr_convolution_maxpool_feature(self):
        return self.using_feature(feature='sent_nr', convolve=True, max_pool=True)

    def using_tense_convolution_maxpool_feature(self):
        return self.using_feature(feature='tense', convolve=True, max_pool=True)

    def using_feature(self, feature, convolve, max_pool):
        result = None

        if convolve is None and max_pool is None:
            result = any([feat['name'] == feature for feat in self.features_to_use])
        else:
            result = any([feat['name'] == feature and feat['convolve'] == convolve and feat['max_pool'] == max_pool
                          for feat in self.features_to_use.values()])

        assert result is not None

        return result

    def to_string(self):
        print '[Tensorflow] Convolutional neural network'

    def get_hidden_activations(self, on_training_set=False, on_validation_set=False, on_testing_set=False):

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

            h = self.hidden_activations(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs)

        with tf.Session(graph=self.graph) as session:
            # init.run()

            self.saver.restore(session, self.get_output_path('params.model'))

            feed_dict = self.get_feed_dict(None, None,
                                           w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs,
                                           None, keep_prob, keep_prob_val=1., dataset=dataset)

            hidden_activations = session.run(h, feed_dict=feed_dict)

        return hidden_activations

    def get_output_logits(self, on_training_set=False, on_validation_set=False, on_testing_set=False):

        output_logits = None

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

            out_logits = self.forward_pass(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs, keep_prob)

        with tf.Session(graph=self.graph) as session:
            # init.run()

            self.saver.restore(session, self.get_output_path('params.model'))

            feed_dict = self.get_feed_dict(None, None,
                                           w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs, tense_idxs,
                                           None, keep_prob, keep_prob_val=1., dataset=dataset)

            output_logits = session.run(out_logits, feed_dict=feed_dict)

        return output_logits
