__author__ = 'root'
from dataset import Dataset

class CorpusReader(object):
    # TRAINING_FILES = [('handoverdata.zip',
    #                    Dataset.TRAINING_SENTENCES_PATH,
    #                    Dataset.TRAINING_SENTENCES_EXTENSION)]

    def __init__(self, clef=False, i2b2=False):

        self.training_files = []

        if clef:
            self.training_files.append(
                ('handoverdata.zip', Dataset.TRAINING_SENTENCES_PATH, Dataset.TRAINING_SENTENCES_EXTENSION)
            )
        if i2b2:
            self.training_files.append(
                ('patients_separated.zip', Dataset.I2B2_PATH+'i2b2_patient_', Dataset.I2B2_EXTENSION)
            )

    def __iter__(self):
        for (file_name,path,extension) in self.training_files:
            f = Dataset.get_filename(file_name, path, extension)
            for i, (doc_nr, text) in enumerate(f):
                sentences = text.strip().replace(u'\ufeff', '').split('\n')
                if sentences.__len__() == 1:
                    sentences = sentences[0].split('.')
                for sent in sentences:
                    yield Dataset.word_tokenize(sent)