__author__ = 'root'

import data
import zipfile
from collections import defaultdict
import io
from nltk import sent_tokenize
import re

class Dataset:

    CRF_FEATURES_PATH = 'handoverdata/101informationextraction/documents/'
    CRF_FEATURES_EXTENSION = '.txt'

    TESTING_FEATURES_PATH = 'handover-set2/100informationextraction/documents/100informationextraction-output-C=1/documents/'
    TESTING_FEATURES_EXTENSION = '.xml.data'

    TRAINING_SENTENCES_PATH = 'handoverdata/101writtenfreetextreports/'
    TRAINING_SENTENCES_EXTENSION = '.txt'

    I2B2_PATH = 'patients_separated/'
    I2B2_EXTENSION = '.txt'

    # List of contractions adapted from Robert MacIntyre's tokenizer.
    CONTRACTIONS2 = [re.compile(r"(?i)\b(can)(not)\b"),
                     re.compile(r"(?i)\b(d)('ye)\b"),
                     re.compile(r"(?i)\b(gim)(me)\b"),
                     re.compile(r"(?i)\b(gon)(na)\b"),
                     re.compile(r"(?i)\b(got)(ta)\b"),
                     re.compile(r"(?i)\b(lem)(me)\b"),
                     re.compile(r"(?i)\b(mor)('n)\b"),
                     re.compile(r"(?i)\b(wan)(na) ")]
    CONTRACTIONS3 = [re.compile(r"(?i) ('t)(is)\b"),
                     re.compile(r"(?i) ('t)(was)\b")]

    def __init__(self):
        pass

    @staticmethod
    def get_training_file_tokenized_sentences(file_name):
        sentences_words = defaultdict(list)
        sentences_tags = defaultdict(list)

        data = Dataset.get_crf_training_data(file_name)
        annotations = data[0]
        texts = data[1].values()
        for i,text in enumerate(texts):
            sentence_words = []
            sentence_tags = []
            for j,word in enumerate(text):
                sentence_words.append(word)
                sentence_tags.append(annotations[i][j]['tag'])
                if word == u'.':
                    sentences_words[i].append(sentence_words)
                    sentences_tags[i].append(sentence_tags)
                    sentence_words = []
                    sentence_tags = []

        return sentences_words, sentences_tags

    @staticmethod
    def get_training_file_text(file_name, path, extension):
        f = Dataset.get_filename(file_name, path, extension)
        # sentences = []
        training_text = defaultdict()
        for i, (doc_nr, text) in enumerate(f):
            training_text[doc_nr] = text.strip().replace(u'\ufeff','')

        return training_text

    @staticmethod
    def get_words_in_training_dataset(file_name):
        documents_text = Dataset.get_training_file_text(file_name,
                                                   Dataset.TRAINING_SENTENCES_PATH,
                                                   Dataset.TRAINING_SENTENCES_EXTENSION)

        #TODO: remove later. for comparison only.
        # training_data, sentences_from_annotations = Dataset.get_crf_training_data(file_name)
        # training_data = [word['word'] for archive in training_data.values() for word in archive.values()]

        text = []
        for sentence in documents_text.values():
            words = Dataset.word_tokenize(sentence)
            text.extend(words)

        return text

    @staticmethod
    def word_tokenize(text):
        # return [token for sent in sent_tokenize(text) for token in self._tokenize(sent)]
        tokens = []
        for sent in sent_tokenize(text):
            # if sent.find('glaucoma.almost') <> -1:
            #     print 'encontrado'
            for token in Dataset._tokenize(sent):
                tokens.append(token)

        return tokens

    @staticmethod
    def _tokenize(text):
        #starting quotes
        text = re.sub(r'^\"', r'``', text)
        text = re.sub(r'(``)', r' \1 ', text)
        text = re.sub(r'([ (\[{<])"', r'\1 `` ', text)

        #punctuation
        text = re.sub(r'([:,])([^\d])', r' \1 \2', text)
        text = re.sub(r'(,)(\d+)', r' \1 \2 ', text) # added by me. It wouldnt split 'Abbot,93'
        text = re.sub(r'\.\.\.', r' ... ', text)
        text = re.sub(r'[;@#$%&]', r' \g<0> ', text)
        text = re.sub(r'([^\.])(\.)([\]\)}>"\']*)\s*$', r'\1 \2\3 ', text)
        text = re.sub(r'([^\.])(\.)([\w])', r'\1 \2 \3', text) # added by me. It wouldnt split 'glaucoma.almost'
        text = re.sub(r'[?!]', r' \g<0> ', text)

        text = re.sub(r"([^'])' ", r"\1 ' ", text)

        #parens, brackets, etc.
        text = re.sub(r'[\]\[\(\)\{\}\<\>]', r' \g<0> ', text)
        text = re.sub(r'--', r' -- ', text)

        #add extra space to make things easier
        text = " " + text + " "

        #ending quotes
        text = re.sub(r'"', " '' ", text)
        text = re.sub(r'(\S)(\'\')', r'\1 \2 ', text)

        text = re.sub(r"([^' ])('[sS]|'[mM]|'[dD]|') ", r"\1 \2 ", text)
        text = re.sub(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ", r"\1 \2 ",
                      text)

        for regexp in Dataset.CONTRACTIONS2:
            text = regexp.sub(r' \1 \2 ', text)
        for regexp in Dataset.CONTRACTIONS3:
            text = regexp.sub(r' \1 \2 ', text)

        # We are not using CONTRACTIONS4 since
        # they are also commented out in the SED scripts
        # for regexp in self.CONTRACTIONS4:
        #     text = regexp.sub(r' \1 \2 \3 ', text)

        return text.split()

    @staticmethod
    def get_filename(file_name, filename_match, extension_match):
        """
        opens the zip file, and reads all the documents that satisfy the matching expression.
        bewatch, it reads the files in this order: output1, output10, output 100, output11, ..., output2.txt, ...

        :param file_name:
        :param filename_match:
        :param extension_match:
        :return:
        """
        resource = data.get_resource(file_name)

        if '.zip' in file_name:
            zip_file = zipfile.ZipFile(resource, mode='r')
            for fname in zip_file.namelist():
                if filename_match in fname and extension_match in fname:
                    doc_nr = int(fname.replace(filename_match,'').replace(extension_match,''))
                    # f = zip_file.read(fname)
                    zip_item = zip_file.open(fname)
                    zip_item = io.TextIOWrapper(zip_item, encoding='utf-8', newline='\n')
                    yield (doc_nr, zip_item.read())

    @staticmethod
    def get_crf_training_data(file_name):
        f = Dataset.get_filename(file_name, Dataset.CRF_FEATURES_PATH, Dataset.CRF_FEATURES_EXTENSION)

        training_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        sentences = defaultdict(list)

        for i, (doc_nr, text) in enumerate(f):
            j = 0 # do not use enumerate instead
            for line in text.split('\n'):
                if len(line) < 2:
                    continue
                line = line.strip().split('\t')
                training_data[doc_nr][j]['word'] = line[0]
                training_data[doc_nr][j]['features'] = line[1:-2]
                training_data[doc_nr][j]['tag'] = line[-2] # -2 is the true label, -1 is their predicted tag
                j += 1
                sentences[doc_nr].append(line[0])

        return training_data, sentences

    @staticmethod
    def get_crf_training_data_by_sentence(file_name, path=CRF_FEATURES_PATH+'output', extension=CRF_FEATURES_EXTENSION):
        """
        returns a dictionary indexed in the following way:
            -doc_nr
                -list of sentences
                    -sentence
                        -list of word_dict
                            -word_dict: ['word'], ['features'], ['tag']
        :param file_name:
        :return:
        """
        f = Dataset.get_filename(file_name, path, extension)

        # training_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        sentences = defaultdict(list)

        dict_sentence = []

        training_data = defaultdict(list)

        word_sentence = []
        tag_sentence = []
        document_sentence_words = defaultdict(list)
        document_sentence_tags = defaultdict(list)

        for i, (doc_nr, text) in enumerate(f):
            for line in text.split('\n'):
                if len(line) < 2:
                    if dict_sentence:
                        training_data[doc_nr].append(dict_sentence)
                        document_sentence_words[doc_nr].append(word_sentence)
                        document_sentence_tags[doc_nr].append(tag_sentence)
                        dict_sentence = []
                        word_sentence = []
                        tag_sentence = []
                    continue
                dict_word = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
                line = line.strip().split('\t')
                dict_word['word'] = line[0]
                dict_word['features'] = line[1:-2]
                dict_word['tag'] = line[-2] # -2 is the true label, -1 is their predicted tag
                dict_sentence.append(dict_word)
                word_sentence.append(dict_word['word'])
                tag_sentence.append(dict_word['tag'])
                sentences[doc_nr].append(line[0])

            if dict_sentence:
                training_data[doc_nr].append(dict_sentence)

        return training_data, sentences, document_sentence_words, document_sentence_tags


if __name__ == '__main__':
    training_data_filename = 'handoverdata.zip'
    testing_data_filename = 'handover-set2.zip'

    dataset = Dataset()

    testing_data, sentences, document_sentence_words, document_sentence_tags = \
        Dataset.get_crf_training_data_by_sentence(testing_data_filename,
                                                  Dataset.TESTING_FEATURES_PATH+'test',
                                                  Dataset.TESTING_FEATURES_EXTENSION)

    # files_text = Dataset.get_training_file_tokenized_sentences(training_data_filename)
    #
    # Dataset.get_words_in_training_dataset(training_data_filename)
    #
    # train_doc_texts = Dataset.get_training_file_text(training_data_filename,
    #                                                       Dataset.TRAINING_SENTENCES_PATH,
    #                                                       Dataset.TRAINING_SENTENCES_EXTENSION)

    # training_data, sentences = dataset.get_crf_training_data(training_data_filename)

    print 'End'
