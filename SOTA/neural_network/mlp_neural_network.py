__author__ = 'root'

import gensim
from data import get_w2v_model
from data import get_w2v_training_data_vectors
import logging
from data.dataset import Dataset
import numpy as np
from collections import defaultdict
import theano
import theano.tensor as T
import os.path
import cPickle
from trained_models import get_cwnn_path
from collections import OrderedDict
from utils import utils
from A_neural_network import A_neural_network

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.warn_float64='ignore'
theano.config.floatX='float64'

def transform_crf_training_data(dataset):
    return [(token['word'],token['tag']) for archive in dataset.values() for token in archive.values()]

class MLP_neural_network_trainer(A_neural_network):

    def __init__(self, hidden_activation_f, out_activation_f, **kwargs):
        super(MLP_neural_network_trainer, self).__init__(**kwargs)
        #
        # self.x_train = x_train.astype(dtype=int)
        # self.y_train = y_train.astype(dtype=int)
        # self.x_valid = None
        # self.y_valid = None
        #
        # self.n_out = n_out
        self.hidden_activation_f = hidden_activation_f
        self.out_activation_f = out_activation_f
        # self.pretrained_embeddings = np.array(self.pretrained_embeddings, dtype=theano.config.floatX)

        self.params = OrderedDict()

    def forward_pass(self, weight_x, bias_1, weight_2, bias_2):
        h = self.hidden_activation_f(weight_x+bias_1)
        return self.out_activation_f(T.dot(h, weight_2)+bias_2)

    def train(self, learning_rate=0.01, batch_size=512, max_epochs=100,
              L1_reg=0.001, L2_reg=0.01, save_params=False, use_scan=False):

        # self.x_train = self.x_train[:10,:] #TODO: remove this. debug only.
        # self.y_train = self.y_train[:10] #TODO: remove this. debug only.

        # i need these theano vars for the minibatch (the givens in the train function). Otherwise i wouldnt.
        train_x = theano.shared(value=np.array(self.x_train, dtype='int64'), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype='int64'), name='train_y', borrow=True)

        y = T.vector(name='y', dtype='int64')

        idxs = T.matrix(name="idxs", dtype='int64') # columns: context window size/lines: tokens in the sentence
        n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # n_tokens = self.x_train.shape[0]    #tokens in sentence
        n_tokens = idxs.shape[0]    #tokens in sentence
        n_window = self.x_train.shape[1]    #context window size    #TODO: replace n_win with self.n_win

        lim = np.sqrt(6./(n_window*n_emb+self.n_out))

        w1 = theano.shared(value=self.pretrained_embeddings, name='w1', borrow=True)
        w2 = theano.shared(value=np.random.uniform(-lim,lim,(n_window*n_emb,self.n_out)).
                           astype(dtype=theano.config.floatX),
                           name='w2', borrow=True)
        b1 = theano.shared(value=np.zeros(n_window*n_emb), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out), name='b2', borrow=True)

        params = [w1,b1,w2,b2]
        param_names = ['w1','b1','w2','b2']

        self.params = OrderedDict(zip(param_names, params))

        w_x = w1[idxs].reshape((n_tokens, n_emb* n_window))

        minibatch_idx = T.lscalar()  # minibatch index

        # def forward_pass(weight_x, bias_1, weight_2, bias_2):
        #     h = self.hidden_activation_f(weight_x+bias_1)
        #     return self.out_activation_f(T.dot(h, weight_2)+bias_2)

        if use_scan:
            #TODO: DO I NEED THE SCAN AT ALL: NO! Im leaving it for reference only.
            # Unchanging variables are passed to scan as non_sequences.
            # Initialization occurs in outputs_info
            out, _ = theano.scan(fn=self.forward_pass,
                                    sequences=w_x,
                                    outputs_info=None,
                                    non_sequences=[b1,w2,b2])
            # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
            cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))
            y_predictions = T.argmax(out[:,-1,:], axis=1)
        else:
            out = self.forward_pass(w_x,b1,w2,b2)
            y_predictions = T.argmax(out, axis=1)
            cost = T.mean(T.nnet.categorical_crossentropy(out, y))

        # test_scan = theano.function(inputs=[idxs,y], outputs=[out,cost], updates=[])

        errors = T.sum(T.neq(y_predictions,y))

        # test_train = theano.function(inputs=[idxs,y], outputs=[out,cost], updates=[])
        # test_train_error = theano.function(inputs=[idxs,y], outputs=[cost], updates=[])

        #TODO: here im testing cost, probabilities, and error calculation. All ok.
        # test_predictions = theano.function(inputs=[idxs,y], outputs=[cost,out,errors], updates=[])
        # cost_out, probs_out, errors_out = test_predictions(self.x_train,self.y_train)

        # y_probabilities, error = test_train(self.x_train, self.y_train)
        # computed_error = test_train_error(self.x_train, self.y_train)

        # y_probabilities = test_scan(self.x_train)
        # y_predictions = np.argmax(y_probabilities[-1][:,0],axis=1)

        grads = [T.grad(cost, param) for param in params]

        # adagrad
        accumulated_grad = []
        for param in params:
            eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for param, grad, accum_grad in zip(params, grads, accumulated_grad):
            # accum = T.cast(accum_grad + T.sqr(grad), dtype=theano.config.floatX)
            accum = accum_grad + T.sqr(grad)
            updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
            updates.append((accum_grad, accum))

        train = theano.function(inputs=[minibatch_idx],
                                outputs=[cost,errors],
                                updates=updates,
                                givens={
                                    idxs: train_x[minibatch_idx*batch_size:(minibatch_idx+1)*batch_size],
                                    y: train_y[minibatch_idx*batch_size:(minibatch_idx+1)*batch_size]
                                })

        for epoch_index in range(max_epochs):
            epoch_cost = 0
            epoch_errors = 0
            for minibatch_index in range(self.x_train.shape[0]/batch_size):
                # error = train(self.x_train, self.y_train)
                cost_output, errors_output = train(minibatch_index)
                epoch_cost += cost_output
                epoch_errors += errors_output
            logger.info('Epoch %d Cost: %f Errors: %d' % (epoch_index+1, epoch_cost, epoch_errors))

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        return True

    def save_params(self):
        for param_name,param_obj in self.params.iteritems():
            cPickle.dump(param_obj, open(get_cwnn_path(param_name+'.p'),'wb'))

        return True

    def predict(self):

        self.x_valid = self.x_valid.astype(dtype=int)
        self.y_valid = self.y_valid.astype(dtype=int)

        y = T.vector(name='valid_y', dtype='int64')

        idxs = T.matrix(name="valid_idxs", dtype='int64') # columns: context window size/lines: tokens in the sentence
        n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # n_tokens = self.x_train.shape[0]    #tokens in sentence
        n_tokens = idxs.shape[0]    #tokens in sentence
        n_window = self.x_train.shape[1]    #context window size    #TODO: replace with self.n_win

        w_x = self.params['w1'][idxs].reshape((n_tokens, n_emb* n_window))

        out = self.forward_pass(w_x, self.params['b1'], self.params['w2'], self.params['b2'])
        y_predictions = T.argmax(out, axis=1)
        cost = T.mean(T.nnet.categorical_crossentropy(out, y))
        errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[idxs],
                                outputs=[y_predictions],
                                updates=[],
                                givens=[])

        predictions = perform_prediction(self.x_valid)

        return self.y_valid, predictions[-1]

    def to_string(self):
        return 'MLP NN with no tags.'


if __name__ == '__main__':
    crf_training_data_filename = 'handoverdata.zip'

    training_vectors_filename = get_w2v_training_data_vectors()

    n_window = 7 #TODO: make param.

    w2v_vectors = None
    w2v_model = None
    w2v_dims = None
    if os.path.exists(training_vectors_filename):
        logger.info('Loading W2V vectors from pickle file')
        w2v_vectors = cPickle.load(open(training_vectors_filename,'rb'))
        w2v_dims = len(w2v_vectors.values()[0])
    else:
        logger.info('Loading W2V model')
        W2V_PRETRAINED_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
        w2v_model = utils.Word2Vec.load_w2v(get_w2v_model(W2V_PRETRAINED_FILENAME))
        w2v_dims = w2v_model.syn0.shape[0]

    logger.info('Loading CRF training data')

    #TODO: if i tokenize myself, i get a diff split
    # training_sentences = Dataset.get_words_in_training_dataset(crf_training_data_filename)

    crf_training_dataset,_ = Dataset.get_crf_training_data(crf_training_data_filename)

    training_sentence_words = [word['word'] for archive in crf_training_dataset.values() for word in archive.values()]
    training_sentence_tags = [word['tag'] for archive in crf_training_dataset.values() for word in archive.values()]

    # annotated_data = transform_crf_training_data(crf_training_dataset)
    # unique_words = list(set([word for word,_ in annotated_data]))
    # unique_labels = list(set([tag for _,tag in annotated_data]))
    unique_words = list(set(training_sentence_words))
    unique_labels = list(set(training_sentence_tags))

    logger.info('Creating word-index dictionaries')
    index2word = defaultdict(None)
    word2index = defaultdict(None)
    word2index['<PAD>'] = len(unique_words)
    for i,word in enumerate(unique_words):
        index2word[i] = word
        word2index[word] = i

    logger.info('Creating label-index dictionaries')
    index2label = defaultdict(None)
    label2index = defaultdict(None)
    label2index['<PAD>'] = len(unique_labels)
    for i,label in enumerate(unique_labels):
        index2label[i] = label
        label2index[label] = i

    n_unique_words = len(unique_words)+1    # +1 for the <PAD>
    n_labels = len(unique_labels)

    lim = np.sqrt(6./(n_unique_words+w2v_dims))

    w = np.random.uniform(-lim,lim,(n_unique_words,w2v_dims))

    w = utils.NeuralNetwork.replace_with_word_embeddings(w, unique_words, w2v_vectors= w2v_vectors, w2v_model=w2v_model)

    #TODO: should i pad every end of sentence? TRY IT!
    x_train = np.matrix([map(lambda x: word2index[x], sentence) for sentence in
                         utils.NeuralNetwork.context_window(training_sentence_words,n_window)])
    # y_train = np.matrix([map(lambda x: label2index[x], sentence) for sentence in
    #                      context_window(training_sentence_tags,n_window)])

    y_train = np.array([label2index[tag] for tag in training_sentence_tags])

    logger.info('Instantiating Neural network')
    nn_trainer = MLP_neural_network_trainer(x_train, y_train,
                                        n_out=n_labels,
                                        # hidden_activation_f=T.nnet.sigmoid,
                                        hidden_activation_f=utils.NeuralNetwork.tanh_activation_function,
                                        # hidden_activation_f=Activation_function.linear,
                                        out_activation_f=utils.NeuralNetwork.softmax_activation_function,
                                        pretrained_embeddings=w)

    logger.info('Training Neural network')
    nn_trainer.train(learning_rate=.01, batch_size=512, max_epochs=50, save_params=False)

    logger.info('End')