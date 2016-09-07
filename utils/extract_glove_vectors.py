__author__ = 'lqrz'

from itertools import chain
import cPickle

from data.dataset import Dataset
from data import get_glove_pretrained_vectors_filepath
from data import get_glove_path

def load_glove_vectors_cache():
    glove_file = open(get_glove_pretrained_vectors_filepath(), 'rb')

    _, _, training_tokens, _ = Dataset.get_clef_training_dataset(lowercase=True)
    _, _, validation_tokens, _ = Dataset.get_clef_validation_dataset(lowercase=True)
    _, _, testing_tokens, _ = Dataset.get_clef_testing_dataset(lowercase=True)

    unique_tokens = set(
        list(chain(*chain(*training_tokens.values()))) + list(chain(*chain(*validation_tokens.values()))) +
        list(chain(*chain(*testing_tokens.values()))))

    glove_representations = dict()

    for line in glove_file:
        word = line.strip().split(' ')[0]
        representation = line.strip().split(' ')[1:]

        assert representation.__len__() == 300

        if word in unique_tokens:
            glove_representations[word] = representation

    return glove_representations

def dump_representations(glove_representations):
    cPickle.dump(glove_representations, open(get_glove_path('glove_representations.p'), 'wb'))

    return True

if __name__ == '__main__':

    glove_representations = load_glove_vectors_cache()

    dump_representations(glove_representations)