__author__ = 'root'

from sklearn.cluster import KMeans
import logging
import gensim
from joblib import load, dump
from data import get_w2v_model


def load_w2v(model_filename):
    return gensim.models.Word2Vec.load_word2vec_format(model_filename, binary=True)

if __name__ == '__main__':

    model_output_filename = 'kmeans.model'

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info('Loading W2V model')
    W2V_PRETRAINED_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    w2v_model = load_w2v(get_w2v_model(W2V_PRETRAINED_FILENAME))

    logger.info('Initializing KMeans model')
    kmeans_model = KMeans(n_clusters=500, init='k-means++', n_jobs=4)

    logger.info('Training KMeans model')
    kmeans_model.fit(w2v_model.syn0)

    logger.info('Persisting KMeans model')
    dump(kmeans_model, model_output_filename)