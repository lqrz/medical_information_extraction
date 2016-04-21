__author__ = 'root'
import gensim
import theano.tensor as T
import numpy as np
from sklearn import metrics
import time
import pandas
from ggplot import *

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
    def context_window(sentence, n_window, pad_idx=None):
        # make sure its uneven
        assert (n_window % 2) == 1, 'Window size must be uneven.'

        # add '<PAD>' tokens at begining and end of sentence
        if pad_idx:
            l_padded = n_window //2 * [pad_idx] + sentence + n_window // 2 * [pad_idx]
        else:
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

    @staticmethod
    def plot(data, x_axis, x_label, y_label, title, output_filename=str(time.time())):
        df = pandas.DataFrame(data)

        p = ggplot(pandas.melt(df, id_vars=[x_axis]), aes(x=x_axis, y='value', color='variable')) + \
            geom_line() + \
            labs(x=x_label, y=y_label) + \
            ggtitle(title)

        ggsave(output_filename+'.png', p, dpi=100)

        return True

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

    @staticmethod
    def compute_precision_score(y_true, y_pred, **kwargs):
        return metrics.precision_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_recall_score(y_true, y_pred, **kwargs):
        return metrics.recall_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_all_metrics(y_true, y_pred, **kwargs):
        results = dict()

        results['accuracy'] = Metrics.compute_recall_score(y_true, y_pred, **kwargs)
        results['precision'] = Metrics.compute_precision_score(y_true, y_pred, **kwargs)
        results['recall'] = Metrics.compute_recall_score(y_true, y_pred, **kwargs)
        results['f1_score'] = Metrics.compute_f1_score(y_true, y_pred, **kwargs)

        return results