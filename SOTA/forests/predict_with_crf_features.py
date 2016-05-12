__author__ = 'lqrz'

import argparse
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from itertools import chain

from data.dataset import Dataset
from SOTA.crf.crf_sklearn_crfsuite import CRF

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

def train_classifier(clf, x_train, y_train):

    # print '...Fitting training data'
    clf.fit(x_train, y_train)

    return True


def predict(clf, x_test):

    # print '...Predicting testing data'
    # x_test_reshaped = pretrained_embeddings[x_test].reshape(-1, n_window * n_emb)
    predictions = clf.predict(x_test)

    return predictions

def parse_arguments():
    parser = argparse.ArgumentParser(description='CRF-MLP Ensemble')
    parser.add_argument('--w2vsimwords', action='store_true', default=False)
    parser.add_argument('--w2vvectors', action='store_true', default=False)
    parser.add_argument('--w2vmodel', action='store', type=str, default=None)
    parser.add_argument('--w2vvectorscache', action='store', type=str, default=None)
    parser.add_argument('--kmeans', action='store_true', default=False)
    parser.add_argument('--kmeansmodel', action='store', type=str, default=None)
    parser.add_argument('--lda', action='store_true', default=False)
    parser.add_argument('--unkscore', action='store_true', default=False)
    parser.add_argument('--metamap', action='store_true', default=False)
    parser.add_argument('--zipfeatures', action='store_true', default=False)
    parser.add_argument('--crfiters', action='store', type=int, default=50)

    arguments = parser.parse_args()

    args = dict()

    args['w2v_similar_words'] = arguments.w2vsimwords
    args['w2v_vector_features'] = arguments.w2vvectors
    args['w2v_model_file'] = arguments.w2vmodel
    args['w2v_vectors_cache'] = arguments.w2vvectorscache
    args['kmeans_features'] = arguments.kmeans
    args['kmeans_model_name'] = arguments.kmeansmodel
    args['lda_features'] = arguments.lda
    args['incl_unk_score'] = arguments.unkscore
    args['incl_metamap'] = arguments.metamap
    args['zip_features'] = arguments.zipfeatures
    args['crf_iters'] = arguments.crfiters

    return args

def determine_and_assign_feature_function(crf_model, args):
    #TODO: im using "custom_word_features_for_nnet"

    feature_function = crf_model.get_custom_word_features_for_nnet

    return feature_function

def use_testing_dataset(crf_model,
                        feature_function,
                        classifier,
                        **kwargs):


    logger.info('Using CLEF testing data')

    # train on 101 training documents, and predict on 100 testing documents.
    testing_data, testing_texts, _, _ = \
        Dataset.get_crf_training_data_by_sentence(file_name=test_data_filename,
                                                  path=Dataset.TESTING_FEATURES_PATH+'test',
                                                  extension=Dataset.TESTING_FEATURES_EXTENSION)

    # set the testing_data attribute of the model
    crf_model.testing_data = testing_data

    # get training features
    x_train = crf_model.get_features_from_crf_training_data(crf_model.training_data, feature_function)
    y_train = crf_model.get_labels_from_crf_training_data(crf_model.training_data)

    # get testing features
    x_test = crf_model.get_features_from_crf_training_data(crf_model.testing_data, feature_function)
    y_test = crf_model.get_labels_from_crf_training_data(crf_model.testing_data)

    x_train = [d.values() for d in list(chain(*chain(*x_train.values())))]
    y_train = list(chain(*chain(*y_train.values())))

    x_test = [d.values() for d in list(chain(*chain(*x_test.values())))]
    y_test = list(chain(*chain(*y_test.values())))

    clf = instantiate_classifier(classifier)
    train_classifier(clf, x_train, y_train)
    predictions = predict(clf, x_test)

if __name__ == '__main__':
    args = parse_arguments()
    crf_training_data_filename = 'handoverdata.zip'
    test_data_filename = 'handover-set2.zip'

    # get the CRF training data
    training_data, _, _, _ = Dataset.get_crf_training_data_by_sentence(crf_training_data_filename)

    # instantiate the CRF
    crf_model = CRF(training_data=training_data,
                    testing_data=None,
                    output_model_filename=None,
                    **args)

    feature_function = determine_and_assign_feature_function(crf_model, args)

    logger.info('Extracting features with: ' + feature_function.__str__())

    logger.info('Using w2v_similar_words:%s kmeans:%s lda:%s zip:%s' %
                (args['w2v_similar_words'], args['kmeans_features'], args['lda_features'], args['zip_features']))

    logger.info('Using w2v_model: %s and vector_dictionary: %s' % (args['w2v_model_file'], args['w2v_vectors_cache']))

    classifier = 'rf'
    results = use_testing_dataset(crf_model,
                                  feature_function,
                                  classifier,
                                  **args)

    print 'End'