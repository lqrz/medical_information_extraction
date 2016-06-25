__author__ = 'lqrz'

from SOTA.neural_network.A_neural_network import A_neural_network
import tensorflow as tf

class Convolutional_Neural_Net(A_neural_network):

    def __init__(self, pos_embeddings, n_filters, region_sizes, **kwargs):

        super(Convolutional_Neural_Net, self).__init__(**kwargs)

        self.graph = tf.Graph()

        # parameters
        self.w1_w2v = None
        self.w1_pos = None

        # embeddings
        self.pos_embeddings = pos_embeddings
        self.n_pos_emb = pos_embeddings.shape[1]

        # convolution filters params
        self.n_filters = n_filters
        self.region_sizes = region_sizes

        # w2v filters
        self.w2v_filters_weights = []
        self.w2v_filters_bias = []

        # parameters to get L2
        self.regularizables = []

        self.initialize_plotting_lists()

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

            self.w1_pos = tf.Variable(initial_value=self.pos_embeddings, dtype=tf.float32, trainable=not static, name='w1_pos')

            self.regularizables.append(self.w1_pos)

            for i, rs in enumerate(self.region_sizes):
                filter_shape = [rs, self.n_pos_emb, 1, self.n_filters]
                w_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w_filter')
                b_filter = tf.Variable(tf.constant(0.1, shape=[self.n_filters]), name='b_filter')
                self.w2v_filters_weights.append(w_filter)
                self.w2v_filters_bias.append(b_filter)

        return

    def _train_graph(self, minibatch_size, max_epochs, learning_rate, lr_decay, plot, alpha_l2=0.001,
                    **kwargs):

        with self.graph.as_default():

            w2v_idxs = tf.placeholder(tf.int32, name='w2v_idxs')
            pos_idxs = tf.placeholder(tf.int32, name='pos_idxs')
            labels = tf.placeholder(tf.int32, name='labels')

            w1_x = tf.nn.embedding_lookup(self.w1, w2v_idxs)
            w1_x_expanded = tf.expand_dims(w1_x, -1)
            # w1_x_r = tf.reshape(w1_x, shape=[-1, self.n_window * self.n_emb])


            for filter_weight, filter_bias in zip(self.w2v_filters_weights[0], self.w2v_filters_bias[0]):
                conv = tf.nn.conv2d(input=w1_x_expanded, filter=filter_weight, strides=[1,1,1,1], padding=None)
                a = tf.nn.bias_add(conv, filter_bias)
                h = tf.nn.relu(a)


        with tf.Session(graph=self.graph) as session:
            session.run(tf.initialize_all_variables())
            session.run([w1_x, w1_x_expanded, conv, a, h], {w2v_idxs: self.x_train[0]})




    def predict(self, on_training_set=False, on_validation_set=False, on_testing_set=False, **kwargs):
        pass