__author__ = 'root'

from sklearn.cluster import KMeans
import logging
from joblib import dump
from data import get_w2v_model
from data.dataset import Dataset
from nltk import word_tokenize
import numpy as np
import utils

def filter_w2v_model(w2v_model):
    training_data_filename = 'handoverdata.zip'
    sentences = Dataset.get_training_file_text(training_data_filename,
                                                    Dataset.TRAINING_SENTENCES_PATH,
                                                    Dataset.TRAINING_SENTENCES_EXTENSION)
    sentences = sentences.values()
    words = set()
    for sentence in sentences:
        words = words.union([w.lower() for w in word_tokenize(sentence[0]) if w.isalpha()])

    vectors = dict()
    for word in words:
        try:
            vectors[word] = w2v_model[word]
        except:
            pass

    matrix = np.zeros((len(vectors.keys()), w2v_model.syn0.shape[1]))

    print matrix.shape

    for i,vec in enumerate(vectors.values()):
        matrix[i] = vec

    return matrix


if __name__ == '__main__':

    model_output_filename = 'kmeans.model'

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info('Loading W2V model')
    W2V_PRETRAINED_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    w2v_model = utils.Word2Vec.load_w2v(get_w2v_model(W2V_PRETRAINED_FILENAME))

    logger.info('Filtering w2v model')
    filtered_w2v_model = filter_w2v_model(w2v_model)
    logger.info('Remaining words: '+str(filtered_w2v_model.shape[0]))

    logger.info('Initializing KMeans model')
    kmeans_model = KMeans(n_clusters=100, init='k-means++', n_jobs=1)

    logger.info('Training KMeans model')
    kmeans_model.fit(filtered_w2v_model)

    logger.info('Persisting KMeans model')
    dump(kmeans_model, model_output_filename)