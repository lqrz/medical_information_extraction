__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

class Forest():
    """
    Random forests and Gradient boosting decision trees for classification.
    """

    def __init__(self, classifier, pretrained_embeddings, n_window, **kwargs):
        self.clf = None
        self.pretrained_embeddings = pretrained_embeddings
        self.n_window = n_window
        self.n_emb = self.pretrained_embeddings.shape[1]

        if classifier == 'gbdt':
            print '...Instantiating Gradient boosting classifier'
            # loss='deviance'
            self.clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=5, random_state=0,
                                             loss='deviance', verbose=False)
        elif classifier == 'rf':
            print '...Instantiating Random forest classifier'
            self.clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, n_jobs=-1,
                                         verbose=True)

    def train(self, x_train, y_train):

        print '...Fitting training data'
        x_train_reshaped = self.pretrained_embeddings[x_train].reshape(-1, self.n_window * self.n_emb)
        self.clf.fit(x_train_reshaped, y_train)

        return True

    def predict(self, x_test):
        print '...Predicting testing data'
        x_test_reshaped = self.pretrained_embeddings[x_test].reshape(-1, self.n_window * self.n_emb)

        return self.clf.predict(x_test_reshaped)