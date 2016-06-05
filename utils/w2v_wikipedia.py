#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp, outp1, outp2 = sys.argv[1:5]
    space = " "
    i = 0

    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")

    inp = outp

    model = Word2Vec(LineSentence(inp), size=100, window=5, min_count=5, negative=10,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.save_word2vec_format(outp2, binary=False)