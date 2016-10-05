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

    def get_frequency_distribution(self, document_sentence_tags):
        tags = list(chain(*(chain(*document_sentence_tags.values()))))
        fd = FreqDist(tags)

        return fd

    def get_empirical_distribution(self, document_sentence_tags):

        fd = self.get_frequency_distribution(document_sentence_tags)

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

def plot_empirical_distribution(distribution, token_counts):
    import rpy2.robjects as robj
    import rpy2.robjects.pandas2ri  # for dataframe conversion
    from rpy2.robjects.packages import importr
    from trained_models import get_analysis_folder_path
    from data import get_training_classification_report_labels

    # labels = map(lambda x: label2index[x], [tag for tag,_ in distribution])
    labels = get_training_classification_report_labels()
    probs = []
    counts = []
    for lab in labels:
        probs.extend([p for t,p in distribution if t==lab])
        counts.append(token_counts[lab])

    data = {
        'labels': labels,
        'probs': probs,
        'counts': counts
    }

    df = pd.DataFrame(data)
    plotFunc = robj.r("""
        library(ggplot2)

        function(df, output_filename){
            str(df)
            # the following instructions are for the plot to take the order given in the dataframe,
            # otherwise, ggplot will reorder it alphabetically.
            df$labels <- as.character(df$labels)
            df$labels <- factor(df$labels, levels=unique(df$labels))
            str(df)
            p <- ggplot(df, aes(x=labels, y=probs)) +
            geom_bar(stat="identity") +
            labs(x='Label', y='Probability', title='Empirical distribution') +
            ylim(0,1) +
            theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust=0.5)) +
            geom_text(aes(label=counts, angle=90), size=3., hjust=-.3, alpha=.7)

            print(p)

            ggsave(output_filename, plot=p)

            }
        """)

    gr = importr('grDevices')
    robj.pandas2ri.activate()
    conv_df = robj.conversion.py2ri(df)

    plotFunc(conv_df, get_analysis_folder_path('empirical_distribution.png'))

    gr.dev_off()

    # p = ggplot(df, aes(x='labels', weight='probs')) + \
    #     geom_bar(stat='bin')

    # ggsave('empirical_distribution', p, dpi=100, bbox_inches='tight')

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
        except KeyError:
            continue

    _, _, _, document_sentence_tags = Dataset.get_clef_training_dataset(lowercase=False)
    tags = list(chain(*(chain(*document_sentence_tags.values()))))

    # label2index = OrderedDict()
    # label2index.update(list(zip(set(tags), range(set(tags).__len__()))))

    training_counts = ed.get_frequency_distribution(document_sentence_tags)

    plot_empirical_distribution(training_dist, training_counts)

    print '...End'