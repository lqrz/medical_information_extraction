__author__='lqrz'

from itertools import chain
from collections import Counter
import numpy as np
import argparse

from data.dataset import Dataset
from metrics import Metrics
from empirical_distribution import Empirical_distribution

def construct_word_tags_counts(training_words):
    d = dict()
    for word in set(training_words):
        idxs = np.where(np.array(training_words) == word)[0]
        word_tags = np.array(training_tags)[idxs]
        d[word] = Counter(word_tags)

    return d


def predict(validation_words, d, default_tag=None):
    predictions = []
    for word in validation_words:
        try:
            prediction = d[word].most_common(n=1)[0][0]
        except KeyError:
            if default_tag:
                prediction = default_tag
            else:
                #sample from empirical distribution
                prediction = Empirical_distribution.Instance().sample_from_training_empirical_distribution()

        predictions.append(prediction)

    return predictions

def get_training_data():
    training_data, _, training_document_sentence_words, training_document_sentence_tags = Dataset.get_clef_training_dataset()

    training_words = list(chain(*chain(*training_document_sentence_words.values())))
    training_tags = list(chain(*chain(*training_document_sentence_tags.values())))

    return training_words, training_tags

def get_validation_data():
    _, _, validation_document_sentence_words, validation_document_sentence_tags = Dataset.get_clef_validation_dataset()
    validation_words = list(chain(*chain(*validation_document_sentence_words.values())))
    validation_tags = list(chain(*chain(*validation_document_sentence_tags.values())))

    return validation_words, validation_tags

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural net trainer')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sampling", action="store_true",
                       help="If word not found in training data, sample a tag from training empirical distribution.")
    group.add_argument("--nosampling", action="store_true",
                       help="If word not found in training data, use an unseen token as tag.")

    #parse arguments
    arguments = parser.parse_args()

    args = dict()

    args['with_sampling'] = arguments.sampling

    return args

if __name__ == '__main__':

    args = parse_arguments()

    print '...Getting data'
    training_words, training_tags = get_training_data()
    validation_words, validation_tags = get_validation_data()

    print '...Constructing word-tag-count dictionary'
    d = construct_word_tags_counts(training_words)

    if args['with_sampling']:
        print '...Predicting (sampling from empirical dist)'
        default_tag = None
    else:
        print '...Predicting (NOT sampling from empirical dist)'
        default_tag = '#IDK#'   #if i dont have it in the training data, then: i dont know
    predictions = predict(validation_words, d, default_tag)

    assert validation_tags.__len__() == predictions.__len__()

    print '...Computing metrics'
    results_micro = Metrics.compute_all_metrics(y_true=validation_tags, y_pred=predictions, average='micro')
    results_macro = Metrics.compute_all_metrics(y_true=validation_tags, y_pred=predictions, average='macro')

    correct = np.where(map(lambda (x,y): x==y, zip(validation_tags, predictions)))[0].__len__()
    errors = validation_tags.__len__() - correct

    print '##ERRORS'
    print errors

    print '##MICRO average'
    print results_micro

    print '##MACRO average'
    print results_macro

    print '...End'