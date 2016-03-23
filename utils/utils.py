__author__ = 'root'
import gensim
import theano.tensor as T
import numpy as np
from sklearn import metrics

class Word2Vec:

    def __init__(self):
        pass

    @staticmethod
    def load_w2v(model_filename):
        return gensim.models.Word2Vec.load_word2vec_format(model_filename, binary=True)


class NeuralNetwork:
    def __init__(self):
        pass

    @staticmethod
    def replace_with_word_embeddings(w, unique_words, w2v_vectors=None, w2v_model=None):
        for i,word in enumerate(unique_words):
            try:
                if w2v_vectors:
                    w[i,:] = w2v_vectors[word.lower()]
                else:
                    w[i,:] = w2v_model[word.lower()] #TODO: lower?
            except KeyError:
                continue

        return w

    @staticmethod
    def context_window(sentence, n_window):
        # make sure its uneven
        assert (n_window % 2) == 1, 'Window size must be uneven.'

        # add '<UNK>' tokens at begining and end of sentence
        l_padded = n_window //2 * ['<PAD>'] + sentence + n_window // 2 * ['<PAD>']

        # slide the window
        return [l_padded[i:(i+n_window)] for i in range(len(sentence))]

    @staticmethod
    def initialize_weights(n_in, n_out, function):
        lim = np.sqrt(6./(n_in+n_out))

        if function=='sigmoid':
            lim *= 4

        return np.random.uniform(-lim,lim,(n_in,n_out))

    @staticmethod
    def linear_activation_function(x):
        return x

    @staticmethod
    def tanh_activation_function(x):
        return T.tanh(x)

    @staticmethod
    def softmax_activation_function(x):
        return T.nnet.softmax(x)

class Metrics:

    @staticmethod
    def compute_accuracy_score(y_true, y_pred, **kwargs):
        return metrics.accuracy_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_f1_score(y_true, y_pred, **kwargs):
        return metrics.f1_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_classification_report(y_true, y_pred, labels, **kwargs):
        return metrics.classification_report(y_true, y_pred, labels, **kwargs)