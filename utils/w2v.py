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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__=='__main__':
    using_preexisting_model = False

    if using_preexisting_model:
        w2v_initial_model_filename = 'GoogleNews-vectors-negative300.bin.gz'
        w2v_initial_model = utils.Word2Vec.load_w2v(get_w2v_model(w2v_initial_model_filename))

        sentences = ['hola', 'que', 'tal']
        w2v_initial_model.train(sentences)
    else:
        min_count = 1
        vector_dim = 100
        output_model_file = 'w2v_clinical_clef-'+str(vector_dim)+'-'+str(min_count)+'.model'
        cr = CorpusReader()

        model = gensim.models.Word2Vec(cr, min_count=min_count, size=vector_dim, workers=4)
        output_path = get_w2v_directory(output_model_file)
        model.save(output_path)
        logger.info('Saved model at: %s' % output_path)

    print 'end'