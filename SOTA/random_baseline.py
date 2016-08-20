__author__ = 'root'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from itertools import chain
from nltk import FreqDist
import numpy as np

from data.dataset import Dataset
from utils.metrics import Metrics
from trained_models import get_random_baseline_path

def generate_predictions(dataset_tags, tag_index, probs):
    random_predictions = []

    for _ in dataset_tags:
        p = np.random.random()
        random_predictions.append(tag_index[np.where(p < np.cumsum(probs))[0][0]])

    assert random_predictions.__len__() == dataset_tags.__len__()

    return random_predictions

if __name__ == '__main__':
    _, _, _, train_document_sentence_tags = Dataset.get_clef_training_dataset(lowercase=True)
    _, _, _, valid_document_sentence_tags = Dataset.get_clef_validation_dataset(lowercase=True)
    _, _, _, test_document_sentence_tags = Dataset.get_clef_testing_dataset(lowercase=True)

    train_tags = list(chain(*chain(*train_document_sentence_tags.values())))

    fd = FreqDist(train_tags)

    tag_index = fd.keys()

    total_count = np.sum(fd.values())
    probs = [count/float(total_count) for count in fd.values()]

    valid_tags = list(chain(*chain(*valid_document_sentence_tags.values())))
    test_tags = list(chain(*chain(*test_document_sentence_tags.values())))

    train_predictions = generate_predictions(train_tags, tag_index, probs)
    valid_predictions = generate_predictions(valid_tags, tag_index, probs)
    test_predictions = generate_predictions(test_tags, tag_index, probs)

    Metrics.print_metric_results(train_y_true=train_tags, train_y_pred=train_predictions,
                                 valid_y_true=valid_tags, valid_y_pred=valid_predictions,
                                 test_y_true=test_tags, test_y_pred=test_predictions,
                                 metatags=False,
                                 get_output_path=get_random_baseline_path,
                                 additional_labels=[])

    print 'End'