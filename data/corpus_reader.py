__author__ = 'root'
from dataset import Dataset

class CorpusReader(object):
    TRAINING_FILES = [('handoverdata.zip',
                       Dataset.TRAINING_SENTENCES_PATH,
                       Dataset.TRAINING_SENTENCES_EXTENSION),
                      ('patients_separated.zip',
                       Dataset.I2B2_PATH,
                       Dataset.I2B2_EXTENSION)]
    # TRAINING_FILES = [('handoverdata.zip',
    #                    Dataset.TRAINING_SENTENCES_PATH,
    #                    Dataset.TRAINING_SENTENCES_EXTENSION)]

    def __init__(self):
        pass

    def __iter__(self):
        for (file_name,path,extension) in CorpusReader.TRAINING_FILES:
            f = Dataset.get_filename(file_name, path, extension)
            for i, text in enumerate(f):
                for sent in text.strip().replace(u'\ufeff','').split('\n'):
                    yield Dataset.word_tokenize(sent)