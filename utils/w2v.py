__author__ = 'root'
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from data import get_w2v_model
from data import get_w2v_directory
import logging
from data.corpus_reader import CorpusReader
import gensim
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train word2vec model')
    parser.add_argument('--existingmodel', action='store', type=str, default=None)
    parser.add_argument('--mincount', action='store', type=int, required=True)
    parser.add_argument('--dim', action='store', type=int, required=True)
    parser.add_argument('--clef', action='store_true', default=False)
    parser.add_argument('--i2b2', action='store_true', default=False)
    arguments = parser.parse_args()

    #check at least one dataset was provided
    assert arguments.clef or arguments.i2b2

    existingmodel = arguments.existingmodel

    if existingmodel:
        # w2v_initial_model_filename = 'GoogleNews-vectors-negative300.bin.gz'
        w2v_initial_model = utils.Word2Vec.load_w2v(get_w2v_model(existingmodel))

        sentences = ['hola', 'que', 'tal']
        w2v_initial_model.train(sentences)
    else:
        min_count = arguments.mincount
        vector_dim = arguments.dim
        dataset_params = {
            'clef': arguments.clef,
            'i2b2': arguments.i2b2
        }

        output_model_file = '-'.join(['w2v_clinical','clef',str(dataset_params['clef']),
                                      'i2b2',str(dataset_params['i2b2']),str(vector_dim),
                                      str(min_count),'.model.bin'])

        cr = CorpusReader(**dataset_params)

        model = gensim.models.Word2Vec(cr, min_count=min_count, size=vector_dim, workers=4)
        output_path = get_w2v_directory(output_model_file)
        model.save_word2vec_format(output_path, binary=True)
        logger.info('Saved model at: %s' % output_path)

    print 'end'