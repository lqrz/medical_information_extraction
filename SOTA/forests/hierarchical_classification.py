__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from SOTA.neural_network.A_neural_network import A_neural_network
from SOTA.neural_network.train_layered_prediction_neural_network import get_original_labels
from SOTA.neural_network.train_layered_prediction_neural_network import filter_tags_to_predict
from SOTA.neural_network.train_layered_prediction_neural_network import load_w2v_model_and_vectors_cache
from utils.metrics import Metrics
from trained_models import get_random_forest_path
from trained_models import get_gbdt_path
from data import get_classification_report_labels
from data import get_hierarchical_mapping

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


def instantiate_classifier(classifier):
    if classifier == 'gbdt':
        # print '...Instantiating Gradient boosting classifier'
        # loss='deviance'
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=5, random_state=0,
                                         loss='deviance', verbose=False)
    elif classifier == 'rf':
        # print '...Instantiating Random forest classifier'
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, n_jobs=-1,
                                     verbose=False)

    return clf


def get_embeddings(unique_words, args):
    print '...Loading w2v embeddings'
    w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache(args)
    pretrained_embeddings = \
        A_neural_network.initialize_w(w2v_dims, unique_words, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

    return pretrained_embeddings


def get_data(add_words, add_tags, add_feats, n_window):

    print('...Loading CRF training and validation data')

    x_train, y_train, x_train_feats, \
    x_valid, y_valid, x_valid_feats, \
    x_test, y_test, x_test_feats, \
    word2index, index2word, \
    label2index, index2label, \
    features_indexes = A_neural_network.get_data(clef_training=True, clef_validation=True, add_words=add_words,
                              add_tags=add_tags, n_window=n_window, feat_positions=[1,2], add_feats=add_feats)

    train_sent_nr, valid_sent_nr, test_sent_nr = \
        A_neural_network.get_word_sentence_number_features(clef_training=True, clef_validation=True,
                                                           clef_testing=False)

    return x_train, y_train, x_train_feats, x_valid, y_valid, x_valid_feats, train_sent_nr, valid_sent_nr, \
           word2index, index2word, label2index, index2label


def train_classifier(clf, pretrained_embeddings, x_train, n_window, n_emb):

    # print '...Fitting training data'
    x_train_reshaped = pretrained_embeddings[x_train].reshape(-1, n_window * n_emb)
    clf.fit(x_train_reshaped, y_train)

    return True


def predict(clf, pretrained_embeddings, x_test, n_window, n_emb):

    # print '...Predicting testing data'
    x_test_reshaped = pretrained_embeddings[x_test].reshape(-1, n_window * n_emb)
    predictions = clf.predict(x_test_reshaped)

    return predictions


if __name__ == '__main__':

    n_window = 3
    n_window = 7
    n_window = 1
    n_window = 5
    add_words = ['<PAD>']
    add_tags = []
    add_feats = ['<PAD>']

    args = dict()
    args['w2v_vectors_cache'] = 'googlenews_representations_train_True_valid_True_test_False.p'
    args['w2v_model_name'] = None

    classifier = 'rf'   #random forest
    classifier = 'gbdt' #gradient boosting decision tree

    print 'Using classifier: %s and window_size: %d' % (classifier, n_window)

    x_train, y_train, x_train_feats, x_valid, y_valid, x_valid_feats, train_sent_nr, valid_sent_nr, \
    word2index, index2word, label2index, index2label,  = get_data(add_words, add_tags, add_feats, n_window)

    unique_words = word2index.keys()

    pretrained_embeddings = get_embeddings(unique_words, args)

    results = dict()

    classification_report_tags = get_classification_report_labels()

    tag_mapping = get_hierarchical_mapping()
    mappedlabel2index = dict(zip(set(tag_mapping.values()), range(set(tag_mapping.values()).__len__())))
    index2mappedlabel = dict(zip(range(set(tag_mapping.values()).__len__()), set(tag_mapping.values())))
    y_train_original_labels = map(lambda x: index2label[x], y_train)
    y_train_mapped_labels = map(lambda x: tag_mapping[x], y_train_original_labels)
    y_valid_original_labels = map(lambda x: index2label[x], y_valid)
    y_valid_mapped_labels = map(lambda x: tag_mapping[x], y_valid_original_labels)

    x_train_reshaped = pretrained_embeddings[np.array(x_train)].reshape(-1, n_window * pretrained_embeddings.shape[1])
    x_train = np.concatenate([np.array(x_train), np.array(x_train_feats[1]), np.array(x_train_feats[2]), np.array(train_sent_nr)[:, np.newaxis]], axis=1)
    x_train = np.concatenate([x_train_reshaped, np.array(x_train_feats[1]), np.array(x_train_feats[2]), np.array(train_sent_nr)[:, np.newaxis]], axis=1)

    clf = instantiate_classifier(classifier)
    clf.fit(x_train, y_train_mapped_labels)

    x_valid_reshaped = pretrained_embeddings[np.array(x_valid)].reshape(-1, n_window * pretrained_embeddings.shape[1])
    x_valid = np.concatenate([np.array(x_valid), np.array(x_valid_feats[1]), np.array(x_valid_feats[2]), np.array(valid_sent_nr)[:, np.newaxis]], axis=1)
    x_valid = np.concatenate([x_valid_reshaped, np.array(x_valid_feats[1]), np.array(x_valid_feats[2]), np.array(valid_sent_nr)[:, np.newaxis]], axis=1)

    predictions = clf.predict(x_valid)

    scores = Metrics.compute_all_metrics(y_true=y_valid_mapped_labels, y_pred=predictions, average=None,
                                         labels=list(set(tag_mapping.values())))

    df = pd.DataFrame.from_dict(scores)
    df['tag'] = list(set(tag_mapping.values()))
    df = df.set_index('tag')
    df[['accuracy', 'precision', 'recall', 'f1_score']].to_excel('scores.xls')

    print '...End'