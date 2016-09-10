__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Forest():
    """
    Random forests and Gradient boosting decision trees for classification.
    """

    def __init__(self, classifier, pretrained_embeddings, pos_embeddings, ner_embeddings, sent_nr_embeddings,
                 tense_embeddings, n_window, **kwargs):
        self.clf = None
        self.pretrained_embeddings = pretrained_embeddings
        self.pos_embeddings = pos_embeddings
        self.ner_embeddings = ner_embeddings
        self.sent_nr_embeddings = sent_nr_embeddings
        self.tense_embeddings = tense_embeddings

        self.n_window = n_window

        if self.pretrained_embeddings is not None:
            self.n_w2v_emb = self.pretrained_embeddings.shape[1]
        if self.pos_embeddings is not None:
            self.n_pos_emb = self.pos_embeddings.shape[1]
        if self.ner_embeddings is not None:
            self.n_ner_emb = self.ner_embeddings.shape[1]
        if self.sent_nr_embeddings is not None:
            self.n_sent_nr_emb = self.sent_nr_embeddings.shape[1]
        if self.tense_embeddings is not None:
            self.n_tense_emb = self.tense_embeddings.shape[1]

        if classifier == 'gbdt':
            print '...Instantiating Gradient boosting classifier'
            # loss='deviance'
            self.clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=5, random_state=0,
                                             loss='deviance', verbose=False)
        elif classifier == 'rf':
            print '...Instantiating Random forest classifier'
            self.clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, n_jobs=-1,
                                         verbose=True)

    def train(self,
              x_train_w2v,
              x_train_pos,
              x_train_ner,
              x_train_sent_nr,
              x_train_tense,
              y_train
              ):

        print '...Fitting training data'

        x_train_reshaped = self.reshape_embeddings(x_train_w2v, x_train_pos, x_train_ner, x_train_sent_nr, x_train_tense, verbose=True)

        assert x_train_reshaped.shape[0] == np.array(y_train).shape[0]

        self.clf.fit(x_train_reshaped, y_train)

        return True

    def reshape_embeddings(self, x_w2v, x_pos, x_ner, x_sent_nr, x_tense, verbose=False):
        embeddings = []
        if self.pretrained_embeddings is not None:
            if verbose:
                print 'Using w2v'
            x_train_w2v_reshaped = self.pretrained_embeddings[x_w2v].reshape(-1, self.n_window * self.n_w2v_emb)
            embeddings.append(x_train_w2v_reshaped)
        if self.pos_embeddings is not None:
            if verbose:
                print 'Using pos'
            x_train_pos_reshaped = self.pos_embeddings[x_pos].reshape(-1, self.n_window * self.n_pos_emb)
            embeddings.append(x_train_pos_reshaped)
        if self.ner_embeddings is not None:
            if verbose:
                print 'Using ner'
            x_train_ner_reshaped = self.ner_embeddings[x_ner].reshape(-1, self.n_window * self.n_ner_emb)
            embeddings.append(x_train_ner_reshaped)
        if self.sent_nr_embeddings is not None:
            if verbose:
                print 'Using sent_nr'
            x_train_sent_nr_reshaped = self.sent_nr_embeddings[x_sent_nr].reshape(-1,
                                                                                        self.n_window * self.n_sent_nr_emb)
            embeddings.append(x_train_sent_nr_reshaped)
        if self.tense_embeddings is not None:
            if verbose:
                print 'Using tense'
            x_train_tense_reshaped = self.tense_embeddings[x_tense].reshape(-1, self.n_window * self.n_tense_emb)
            embeddings.append(x_train_tense_reshaped)

        assert np.unique(map(lambda x: x.shape[0], embeddings)).shape[0] == 1

        x_train = np.concatenate(embeddings, axis=1)

        return x_train

    def predict(self, x_test, x_test_pos, x_test_ner, x_test_sent_nr, x_test_tense):

        print '...Predicting testing data'

        x_test_reshaped = self.reshape_embeddings(x_test, x_test_pos, x_test_ner, x_test_sent_nr, x_test_tense, verbose=False)

        return self.clf.predict(x_test_reshaped)