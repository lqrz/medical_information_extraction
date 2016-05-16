__author__='lqrz'

from nltk import FreqDist
from itertools import chain
import numpy as np

from data.dataset import Dataset

class Singleton():
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Other than that, there are
    no restrictions that apply to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    Limitations: The decorated class cannot be inherited from.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)

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

if __name__ == '__main__':
    ed = Empirical_distribution.Instance()
    training_dist = ed._get_training_empirical_distribution()
    validation_dist = ed._get_validation_empirical_distribution()
    sample = [ed.sample_from_training_empirical_distribution() for _ in range(10)]

    print '...End'