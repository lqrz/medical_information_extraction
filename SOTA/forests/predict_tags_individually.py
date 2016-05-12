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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


def instantiate_classifier(classifier):
    if classifier == 'gbdt':
        # print '...Instantiating Gradient boosting classifier'
        # loss='deviance'
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=5, random_state=0,
                                         loss='deviance', max_features=300, verbose=False)
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


def get_data(crf_training_data_filename, test_data_filename, add_words, add_tags, n_window):

    print('...Loading CRF training data')
    x_train, y_train_all, x_test, y_test_all, word2index, index2word, label2index, index2label = \
        A_neural_network.get_data(crf_training_data_filename, test_data_filename, add_words, add_tags,
                                  x_idx=None, n_window=n_window)

    return x_train, y_train_all, x_test, y_test_all, word2index, index2word, label2index, index2label


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

    crf_training_data_filename = 'handoverdata.zip'
    test_data_filename = 'handover-set2.zip'

    n_window = 3
    n_window = 5
    n_window = 7
    n_window = 1
    add_words = ['<PAD>']
    add_tags = []
    tags = None
    tags = 'diagnosis.p'

    args = dict()
    args['w2v_vectors_cache'] = 'googlenews_representations_train_True_valid_True_test_False.p'
    args['w2v_model_name'] = None

    classifier = 'rf'   #random forest
    classifier = 'gbdt' #gradient boosting decision tree

    print 'Using classifier: %s and window_size: %d' % (classifier, n_window)

    x_train, y_train_all, x_test, y_test_all, word2index, index2word, label2index_all, index2label_all = \
        get_data(crf_training_data_filename, test_data_filename, add_words, add_tags, n_window)

    unique_words = word2index.keys()

    pretrained_embeddings = get_embeddings(unique_words, args)

    results = dict()

    classification_report_tags = get_classification_report_labels()

    for i, tag in enumerate(classification_report_tags):
        print 'Tag: %s' % tag
        tags = [tag]

        if tags:
            y_train_labels, y_test_labels = get_original_labels(y_test_all, y_train_all, index2label_all)
            # tags = get_param(tags)
            # tags_2nd_step = set(label2index_all.keys()) - set(tags)
            y_train, y_test, label2index, index2label = filter_tags_to_predict(y_train_labels, y_test_labels, tags)
        else:
            y_train = y_train_all
            y_test = y_test_all
            label2index = label2index_all

        if set(y_train).__len__()==1 or set(y_test).__len__()==1:
            print 'No training data.'
            results[i] = {
                        '1_tag': tag,
                        '2_precision': .0,
                        '3_recall': .0,
                        '4_f1_score': .0
                        }
            continue

        n_out = len(label2index.keys())

        clf = instantiate_classifier(classifier)

        n_emb = pretrained_embeddings.shape[1]

        train_classifier(clf, pretrained_embeddings, x_train, n_window, n_emb)

        predictions = predict(clf, pretrained_embeddings, x_test, n_window, n_emb)

        y_test = map(lambda x: index2label[x], y_test)
        predictions_labels = map(lambda x: index2label[x], predictions)

        # print '...Computing scores for tag %s' % tag
        scores = \
            Metrics.compute_all_metrics(y_test, predictions_labels,
                                        average=None,
                                        labels=[t[0] for t in sorted(label2index.iteritems(), key=lambda x: x[1])])

        results[i] = {
                        '1_tag': tag,
                        '2_precision': scores['precision'][label2index[tag]],
                        '3_recall': scores['recall'][label2index[tag]],
                        '4_f1_score': scores['f1_score'][label2index[tag]]
                        }

    df = pd.DataFrame.from_dict(results, orient='index')
    get_output = get_random_forest_path if classifier == 'rf' else get_gbdt_path
    np.savetxt(get_output('classification_report.txt'), df.values, fmt=['%-52s','%-f','%-f','%-f'])

    print 'End'