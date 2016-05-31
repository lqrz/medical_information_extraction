__author__='lqrz'

from nltk import FreqDist
from itertools import chain
import numpy as np
from ggplot import *
import pandas as pd
from collections import OrderedDict

from data.dataset import Dataset
from data import get_classification_report_labels
from singleton import Singleton

@Singleton
class Empirical_distribution():

    def __init__(self):
        self.training_distribution = None
        self.validation_distribution = None

    def sample_from_training_empirical_distribution(self):
        training_dist = self._get_training_empirical_distribution()

        return self.sample_from_empirical_distribution(training_dist)

    def sample_from_empirical_distribution(self, distribution):
        tags = [t[0] for t in distribution]
        probs = [t[1] for t in distribution]
        acc_dist = np.cumsum(probs, dtype=float)
        sample_prob = np.random.random()
        sample_idxs = np.min(np.where(acc_dist >= sample_prob)[0])

        return tags[sample_idxs]

    def get_empirical_distribution(self, document_sentence_tags):
        tags = list(chain(*(chain(*document_sentence_tags.values()))))
        fd = FreqDist(tags)
        normalized_probs = fd.values() / np.sum(fd.values(), dtype=float)

        dist = list(zip(fd.keys(), normalized_probs))

        return dist

    def _get_training_empirical_distribution(self):
        if not self.training_distribution:
            _, _, _, document_sentence_tags = Dataset.get_clef_training_dataset()
            self.training_distribution = self.get_empirical_distribution(document_sentence_tags)

        return self.training_distribution

    def _get_validation_empirical_distribution(self):
        if not self.validation_distribution:
            _, _, _, document_sentence_tags = Dataset.get_clef_validation_dataset()
            self.validation_distribution = self.get_empirical_distribution(document_sentence_tags)

        return self.validation_distribution

def plot_empirical_distribution(distribution, label2index):
    labels = map(lambda x: label2index[x], [tag for tag,_ in distribution])
    probs = [prob for _,prob in distribution]

    data = {
        'labels': labels,
        'probs': probs
    }

    df = pd.DataFrame(data)
    p = ggplot(df, aes(x='labels', weight='probs')) + \
        geom_bar(stat='bin')

    ggsave('empirical_distribution', p, dpi=100, bbox_inches='tight')

if __name__ == '__main__':
    ed = Empirical_distribution.Instance()
    training_dist = ed._get_training_empirical_distribution()
    validation_dist = ed._get_validation_empirical_distribution()
    sample = [ed.sample_from_training_empirical_distribution() for _ in range(10)]

    data = dict(training_dist)
    for tag in get_classification_report_labels():
        try:
            prob = data[tag]
            print tag, "{:10.4f}".format(prob)
        except:
            continue

    _, _, _, document_sentence_tags = Dataset.get_clef_validation_dataset()
    tags = list(chain(*(chain(*document_sentence_tags.values()))))

    label2index = OrderedDict()
    label2index.update(list(zip(set(tags), range(set(tags).__len__()))))

    plot_empirical_distribution(validation_dist, label2index)

    print '...End'