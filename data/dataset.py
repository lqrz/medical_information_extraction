__author__ = 'root'

import data
import zipfile
from collections import defaultdict
import io
# from nltk import sent_tokenize

class Dataset:

    CRF_FEATURES_PATH = '101informationextraction/documents/'
    CRF_FEATURES_EXTENSION = '.txt'

    TRAINING_SENTENCES_PATH = '101writtenfreetextreports/'
    TRAINING_SENTENCES_EXTENSION = '.txt'

    def __init__(self):
        pass

    def get_training_file_sentences(self, file_name):
        f = self.get_filename(file_name, self.TRAINING_SENTENCES_PATH, self.TRAINING_SENTENCES_EXTENSION)
        # sentences = []
        training_sentences = defaultdict(list)

        for i, text in enumerate(f):
            training_sentences[i].append(text.strip().replace(u'\ufeff',''))

        return training_sentences

    @staticmethod
    def get_filename(file_name, filename_match, extension_match):
        resource = data.get_resource(file_name)

        if '.zip' in file_name:
            zip_file = zipfile.ZipFile(resource, mode='r')
            for fname in zip_file.namelist():
                if filename_match in fname and extension_match in fname:
                    # f = zip_file.read(fname)
                    zip_item = zip_file.open(fname)
                    zip_item = io.TextIOWrapper(zip_item, encoding='utf-8', newline='\n')
                    yield zip_item.read()

    @staticmethod
    def get_crf_training_data(file_name):
        f = Dataset.get_filename(file_name, Dataset.CRF_FEATURES_PATH, Dataset.CRF_FEATURES_EXTENSION)
        training_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        sentences = defaultdict(list)
        # yield zip_file.read(f)
        # for line in f.split('\n'):
        for i, text in enumerate(f):
            j = 0 # do not use enumerate instead
            for line in text.split('\n'):
                if len(line) < 2:
                    continue
                line = line.strip().split('\t')
                training_data[i][j]['word'] = line[0]
                training_data[i][j]['features'] = line[1:-2]
                training_data[i][j]['tag'] = line[-2] # -2 is the true label, -1 is their predicted tag
                j += 1
                sentences[i].append(line[0])

        return training_data, sentences


if __name__ == '__main__':
    training_data_filename = 'handoverdata.zip'

    dataset = Dataset()

    dataset.get_training_file_sentences(training_data_filename)

    dataset.get_crf_training_data(training_data_filename)