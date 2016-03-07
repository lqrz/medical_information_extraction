__author__ = 'root'

from itertools import chain
import nltk
import pycrfsuite
import sklearn
from nltk.corpus import conll2002
import logging
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_filename = 'crfsuite_test.model'

def get_features(sent, i):
    word = sent[i][0]
    pos_tag = sent[i][1]
    label = sent[i][2]

    features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'postag=' + pos_tag,
            'postag[:2]=' + pos_tag[:2],
        ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features

def sent2features(sent):
        return [get_features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
        return [label for _,_,label in sent]

def sent2tokens(sent):
        return [token for token,_,_ in sent]

def train():
        #retrieve corpus (spanish sentences)
    logger.info('Retrieving corpus')
    logger.info('Corpus len: %d', len(conll2002.iob_sents('esp.train')))
    corpus = conll2002.iob_sents('esp.train')

    #create features
    logger.info('Getting features')
    # features = get_features(corpus[0], 0)
    x_train = [sent2features(sent) for sent in corpus]
    y_train = [sent2labels(sent) for sent in corpus]

    # print x_train
    logger.info('Instantiating CRF')
    crf_trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(x_train, y_train):
        crf_trainer.append(xseq, yseq)

    crf_trainer.set_params({
        'c1': 1.0,
        'c2': 1e-3,
        'max_iterations': 50,
        'feature.possible_transitions': True
    })

    logger.info('Training CRF')
    crf_trainer.train(model_filename)

    return

def predict():

    corpus = conll2002.iob_sents('esp.testb')

    tagger = pycrfsuite.Tagger()
    tagger.open(model_filename)

    sent = corpus[0]
    prediction = tagger.tag(sent2features(sent))
    true = sent2labels(sent)

    print 'Original sentence: ', sent2tokens(sent)
    print 'Label prediction: ', prediction
    print 'True labels: ', true

    info = tagger.info()

    len(info.state_features)

    print Counter(info.state_features).most_common(20)


if __name__=='__main__':

    # train()

    predict()

    logger.info('End')