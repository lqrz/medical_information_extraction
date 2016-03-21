__author__ = 'root'

import logging
import gensim
import bz2

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import get_wikipedia_file
from joblib import dump


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':

    n_topics = 100
    model_output_filename = 'wikipedia_lda.model'

    wordids_filepath = get_wikipedia_file('en_wiki_wordids.txt.bz2')
    tfidf_filepath = get_wikipedia_file('en_wiki_tfidf.mm')

    # load id->word mapping (the dictionary), one of the results of step 2 above
    id2word = gensim.corpora.Dictionary.load_from_text(bz2.BZ2File(wordids_filepath))
    # load corpus iterator
    mm = gensim.corpora.MmCorpus(tfidf_filepath)
    # mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

    lda_model = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=n_topics,
                                                update_every=1, chunksize=10000, passes=1)

    dump(lda_model, model_output_filename)