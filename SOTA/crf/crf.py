__author__ = 'root'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import pycrfsuite
import logging
import codecs
from data.dataset import Dataset
from nltk.tokenize import word_tokenize
from sklearn.cross_validation import LeaveOneOut
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CRF:

    def __init__(self, training_data, training_texts, test_data, output_model_filename):
        self.training_data = training_data
        self.file_texts = training_texts
        # self.file_texts = dataset.get_training_file_sentences(training_data_filename)

        if test_data_filename:
            self.test_data = Dataset.get_crf_training_data(test_data_filename)

        self.output_model_filename = output_model_filename

    def get_sentence_labels(self, sentence, file_idx):
        # return [self.training_data[file_idx][j]['tag'] for j, sentence in enumerate(sentence.split(' '))]
        tags = []
        # for j, word in enumerate(word_tokenize(sentence)):
        for j, word in enumerate(sentence):
            if self.training_data[file_idx][j]['word'] != word:
                print 'mismatch'
            else:
                tags.append(self.training_data[file_idx][j]['tag'])

        return tags

    def get_labels(self):
        return [self.get_sentence_labels(sentence, file_idx)
                for file_idx, sentence in enumerate(self.file_texts)]

    def get_labels_from_crf_training_data(self):
        return [self.get_sentence_labels(sentence, file_idx)
                for file_idx, sentence in self.file_texts.iteritems()]

    def get_word_features(self, sentence, file_idx, word_idx):
        features = []

        word = sentence[word_idx]
        word_lemma = self.training_data[file_idx][word_idx]['features'][0]
        word_ner = self.training_data[file_idx][word_idx]['features'][1]
        word_pos = self.training_data[file_idx][word_idx]['features'][2]
        word_parse_tree = self.training_data[file_idx][word_idx]['features'][3]
        word_basic_dependents = self.training_data[file_idx][word_idx]['features'][4]
        word_basic_governors = self.training_data[file_idx][word_idx]['features'][5]
        word_unk_score = self.training_data[file_idx][word_idx]['features'][6]
        word_phrase = self.training_data[file_idx][word_idx]['features'][7]
        word_top_candidate_1 = self.training_data[file_idx][word_idx]['features'][8]
        word_top_candidate_2 = self.training_data[file_idx][word_idx]['features'][9]
        word_top_candidate_3 = self.training_data[file_idx][word_idx]['features'][10]
        word_top_candidate_4 = self.training_data[file_idx][word_idx]['features'][11]
        word_top_candidate_5 = self.training_data[file_idx][word_idx]['features'][12]
        word_top_mapping = self.training_data[file_idx][word_idx]['features'][13]
        word_medication_score = self.training_data[file_idx][word_idx]['features'][14]
        word_location = self.training_data[file_idx][word_idx]['features'][15]

        word_tag = self.training_data[file_idx][word_idx]['tag']

        # Unigram
        # U01:%x[0,0]
        features.append(word)
        # U06:%x[0,1]
        features.append(word_lemma)
        # U11:%x[0,2]
        features.append(word_ner)
        # U16:%x[0,3]
        features.append(word_pos)
        # U21:%x[0,4]
        features.append(word_parse_tree)
        # U26:%x[0,5]
        features.append(word_basic_dependents)
        # U31:%x[0,6]
        features.append(word_basic_governors)
        # U36:%x[0,8]
        features.append(word_phrase)
        # U41:%x[0,9]
        features.append(word_top_candidate_1)
        # U46:%x[0,10]
        features.append(word_top_candidate_2)
        # U51:%x[0,11]
        features.append(word_top_candidate_3)
        # U56:%x[0,12]
        features.append(word_top_candidate_4)
        # U61:%x[0,13]
        features.append(word_top_candidate_5)
        # U66:%x[0,14]
        features.append(word_top_mapping)
        # U71:%x[0,15]
        features.append(word_medication_score)
        # U76:%x[0,16]
        features.append(word_location)
        # U80:%x[0,1]/%x[0,2]/%x[0,3]/%x[0,5]/%x[0,6]/%x[0,7]/%x[0,8]/%x[0,9]/%x[0,10]/%x[0,11]/%x[0,12]/%x[0,13]/%x[0,14]/%x[0,15]/%x[0,16]
        features.append(
            word_lemma + '/'+
            word_ner + '/'+
            word_pos + '/'+
            word_parse_tree + '/'+
            word_basic_dependents + '/'+
            word_basic_governors + '/'+
            word_unk_score + '/'+
            word_phrase + '/'+
            word_top_candidate_1 + '/'+
            word_top_candidate_2 + '/'+
            word_top_candidate_3 + '/'+
            word_top_candidate_4 + '/'+
            word_top_candidate_5 + '/'+
            word_top_mapping + '/'+
            word_medication_score + '/'+
            word_location
        )

        if word_idx > 0:
            # U00:%x[-1,0]
            previous_word = sentence[word_idx-1]
            features.append(previous_word)
            # U03:%x[-1,0]/%x[0,0]
            features.append(previous_word +'/'+ word)
            # U05:%x[-1,1]
            features.append(self.training_data[file_idx][word_idx-1]['features'][0])
            # U10:%x[-1,2]
            features.append(self.training_data[file_idx][word_idx-1]['features'][1])
            # U15:%x[-1,3]
            features.append(self.training_data[file_idx][word_idx-1]['features'][2])
            # U20:%x[-1,4]
            features.append(self.training_data[file_idx][word_idx-1]['features'][3])
            # U25:%x[-1,5]
            features.append(self.training_data[file_idx][word_idx-1]['features'][4])
            # U30:%x[-1,6]
            features.append(self.training_data[file_idx][word_idx-1]['features'][5])
            # U35:%x[-1,8]
            features.append(self.training_data[file_idx][word_idx-1]['features'][7])
            # U40:%x[-1,9]
            features.append(self.training_data[file_idx][word_idx-1]['features'][8])
            # U45:%x[-1,10]
            features.append(self.training_data[file_idx][word_idx-1]['features'][9])
            # U50:%x[-1,11]
            features.append(self.training_data[file_idx][word_idx-1]['features'][10])
            # U55:%x[-1,12]
            features.append(self.training_data[file_idx][word_idx-1]['features'][11])
            # U60:%x[-1,13]
            features.append(self.training_data[file_idx][word_idx-1]['features'][12])
            # U65:%x[-1,14]
            features.append(self.training_data[file_idx][word_idx-1]['features'][13])
            # U70:%x[-1,15]
            features.append(self.training_data[file_idx][word_idx-1]['features'][14])
            # U75:%x[-1,16]
            features.append(self.training_data[file_idx][word_idx-1]['features'][15])
            # Bigram
            # B
            previous_tag = self.training_data[file_idx][word_idx-1]['tag']
            features.append(previous_tag+'/'+word_tag)
            # U08:%x[-1,1]/%x[0,1]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][0] +'/'+
                word_lemma
            )
            # U13:%x[-1,2]/%x[0,2]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][1] +'/'+
                word_ner
            )
            # U18:%x[-1,3]/%x[0,3]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][2] +'/'+
                word_pos
            )
            # U23:%x[-1,4]/%x[0,4]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][3] +'/'+
                word_parse_tree
            )
            # U28:%x[-1,5]/%x[0,5]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][4] +'/'+
                word_basic_dependents
            )
            # U33:%x[-1,6]/%x[0,6]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][5] +'/'+
                word_basic_governors
            )
            # U38:%x[-1,8]/%x[0,8]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][7] +'/'+
                word_phrase
            )
            # U43:%x[-1,9]/%x[0,9]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][8] +'/'+
                word_top_candidate_1
            )
            # U48:%x[-1,10]/%x[0,10]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][9] +'/'+
                word_top_candidate_2
            )
            # U53:%x[-1,11]/%x[0,11]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][10] +'/'+
                word_top_candidate_3
            )
            # U58:%x[-1,12]/%x[0,12]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][11] +'/'+
                word_top_candidate_4
            )
            # U63:%x[-1,13]/%x[0,13]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][12] +'/'+
                word_top_candidate_5
            )
            # U68:%x[-1,14]/%x[0,14]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][13] +'/'+
                word_top_mapping
            )
            # U73:%x[-1,15]/%x[0,15]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][14] +'/'+
                word_medication_score
            )
            # U78:%x[-1,16]/%x[0,16]
            features.append(
                self.training_data[file_idx][word_idx-1]['features'][15] +'/'+
                word_location
            )
        else:
            # features['BOS'] = True
            features.append(True)

        if word_idx < len(sentence)-1:
            # U02:%x[1,0]
            next_word = sentence[word_idx+1]
            features.append(next_word)
            # U04:%x[0,0]/%x[1,0]
            features.append(word +'/'+ next_word)
            # U07:%x[1,1]
            features.append(self.training_data[file_idx][word_idx+1]['features'][0])
            # U12:%x[1,2]
            features.append(self.training_data[file_idx][word_idx+1]['features'][1])
            # U17:%x[1,3]
            features.append(self.training_data[file_idx][word_idx+1]['features'][2])
            # U22:%x[1,4]
            features.append(self.training_data[file_idx][word_idx+1]['features'][3])
            # U27:%x[1,5]
            features.append(self.training_data[file_idx][word_idx+1]['features'][4])
            # U32:%x[1,6]
            features.append(self.training_data[file_idx][word_idx+1]['features'][5])
            # U37:%x[1,8]
            features.append(self.training_data[file_idx][word_idx+1]['features'][7])
            # U42:%x[1,9]
            features.append(self.training_data[file_idx][word_idx+1]['features'][8])
            # U47:%x[1,10]
            features.append(self.training_data[file_idx][word_idx+1]['features'][9])
            # U52:%x[1,11]
            features.append(self.training_data[file_idx][word_idx+1]['features'][10])
            # U57:%x[1,12]
            features.append(self.training_data[file_idx][word_idx+1]['features'][11])
            # U62:%x[1,13]
            features.append(self.training_data[file_idx][word_idx+1]['features'][12])
            # U67:%x[1,14]
            features.append(self.training_data[file_idx][word_idx+1]['features'][13])
            # U72:%x[1,15]
            features.append(self.training_data[file_idx][word_idx+1]['features'][14])
            # U77:%x[1,16]
            features.append(self.training_data[file_idx][word_idx+1]['features'][15])
            # U09:%x[0,1]/%x[1,1]
            features.append(
                word_lemma +'/'+
                self.training_data[file_idx][word_idx+1]['features'][0]
            )
            # U14:%x[0,2]/%x[1,2]
            features.append(
                word_ner +'/'+
                self.training_data[file_idx][word_idx+1]['features'][1]
            )
            # U19:%x[0,3]/%x[1,3]
            features.append(
                word_pos +'/'+
                self.training_data[file_idx][word_idx+1]['features'][2]
            )
            # U24:%x[0,4]/%x[1,4]
            features.append(
                word_parse_tree +'/'+
                self.training_data[file_idx][word_idx+1]['features'][3]
            )
            # U29:%x[0,5]/%x[1,5]
            features.append(
                word_basic_dependents +'/'+
                self.training_data[file_idx][word_idx+1]['features'][4]
            )
            # U34:%x[0,6]/%x[1,6]
            features.append(
                word_basic_governors +'/'+
                self.training_data[file_idx][word_idx+1]['features'][5]
            )
            # U39:%x[0,8]/%x[1,8]
            features.append(
                word_phrase +'/'+
                self.training_data[file_idx][word_idx+1]['features'][7]
            )
            # U44:%x[0,9]/%x[1,9]
            features.append(
                word_top_candidate_1 +'/'+
                self.training_data[file_idx][word_idx+1]['features'][8]
            )
            # U49:%x[0,10]/%x[1,10]
            features.append(
                word_top_candidate_2 +'/'+
                self.training_data[file_idx][word_idx+1]['features'][9]
            )
            # U54:%x[0,11]/%x[1,11]
            features.append(
                word_top_candidate_3 +'/'+
                self.training_data[file_idx][word_idx+1]['features'][10]
            )
            # U59:%x[0,12]/%x[1,12]
            features.append(
                word_top_candidate_4 +'/'+
                self.training_data[file_idx][word_idx+1]['features'][11]
            )
            # U64:%x[0,13]/%x[1,13]
            features.append(
                word_top_candidate_5 +'/'+
                self.training_data[file_idx][word_idx+1]['features'][12]
            )
            # U69:%x[0,14]/%x[1,14]
            features.append(
                word_top_mapping +'/'+
                self.training_data[file_idx][word_idx+1]['features'][13]
            )
            # U74:%x[0,15]/%x[1,15]
            features.append(
                word_medication_score +'/'+
                self.training_data[file_idx][word_idx+1]['features'][14]
            )
            # U79:%x[0,16]/%x[1,16]
            features.append(
                word_location +'/'+
                self.training_data[file_idx][word_idx+1]['features'][15]
            )
        else:
            # features['EOS'] = True
            features.append(True)

        return features

    def get_sentence_features(self, sentence, file_idx):
        # return [self.training_data[file_idx][j]['features'] for j, word in enumerate(word_tokenize(sentence))]
        features = []
        # for j, word in enumerate(word_tokenize(sentence)):
        for j, word in enumerate(sentence):
            if self.training_data[file_idx][j]['word'] != word:
                print 'mismatch'
            else:
                features.append(self.get_word_features(sentence, file_idx, j))

        return features

    def get_features(self):
        return [self.get_sentence_features(sentence, file_idx)
                for file_idx, file_text in self.file_texts.iteritems()
                for sentence in file_text[0].split('\n')]

    def get_features_from_crf_training_data(self):
        return [self.get_sentence_features(sentence, file_idx)
                for file_idx, sentence in self.file_texts.iteritems()]

    def train(self, x_idxs, verbose=False):
        # x_train = self.get_features()
        x_train = self.get_features_from_crf_training_data()
        y_train = self.get_labels_from_crf_training_data()

        x_train = np.array(x_train)[x_idxs]
        y_train = np.array(y_train)[x_idxs]

        crf_trainer = pycrfsuite.Trainer(verbose=verbose)

        for xseq, yseq in zip(x_train, y_train):
            crf_trainer.append(xseq, yseq)

        crf_trainer.set_params({
            'c1': 1.0,
            'c2': 1e-3,
            'max_iterations': 50,
            'feature.possible_transitions': True
        })

        crf_trainer.train(self.output_model_filename)

        return

    def predict(self, y_idx):

        accuracy = 0

        tagger = pycrfsuite.Tagger()
        tagger.open(self.output_model_filename)

        x_train = self.get_features_from_crf_training_data()
        y_train = self.get_labels_from_crf_training_data()

        x_train = np.array(x_train)[y_idx]
        y_train = np.array(y_train)[y_idx]

        for sent in x_train:
            # print tagger.tag(sent)
            predictions = tagger.tag(sent)
            accuracy = sum([pred==y_train.__getitem__(0)[i] for i, pred in enumerate(predictions)])

        return float(accuracy)/len(predictions)


if __name__ == '__main__':
    training_data_filename = 'handoverdata.zip'
    test_data_filename = None
    output_model_filename = 'crf_trained.model'

    training_data, training_texts = Dataset.get_crf_training_data(training_data_filename)

    results = []

    loo = LeaveOneOut(training_data.__len__())
    for x_idx, y_idx in loo:
        # print x_idx, y_idx
        crf_model = CRF(training_data, training_texts, test_data_filename, output_model_filename)
        crf_model.train(x_idx,verbose=False)
        results.append(crf_model.predict(y_idx))

    print results
    print 'Mean accuracy: ', np.mean(results)
