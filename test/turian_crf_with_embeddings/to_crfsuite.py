#!/usr/bin/env python
"""
Code originally from CRFsuite, modified by Joseph Turian to include representations.
You can enter combinations of brown and/or embedding representations
to add to the feature set.
"""

import sys, string
from file import myopen

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-b", "--brown", dest="brown", action="append", help="brown clusters to use")
parser.add_option("-e", "--embedding", dest="embedding", action="append", help="embedding to use")
parser.add_option("--brown-prefixes", dest="prefixes", default="4,6,10,20", help="brown prefixes")
parser.add_option("--embedding-scale", dest="embeddingscale", type="float", action="append", help="scaling factor for embeddings (if a list, different scaling for each embedding)")
parser.add_option("--no-pos-features", dest="no_pos_features", default=False, action="store_true", help="don't use any POS features")
parser.add_option("--compound-representation-features", dest="compound_representation_features", default=False, action="store_true", help="compound the word representation features")
parser.add_option("--compound-embedding-threshold", dest="compound_embedding_threshold", default="0", help="min threshold for compound embedding features")

(options, args) = parser.parse_args()
assert len(args) == 0

prefixes = [int(s) for s in string.split(options.prefixes, sep=",")]

if options.brown is None: options.brown = []
word_to_cluster = []
for i, brownfile in enumerate(options.brown):
    print >> sys.stderr, "Reading Brown file: %s" % brownfile
    word_to_cluster.append({})
    assert len(word_to_cluster) == i+1
    for l in myopen(brownfile):
        cluster, word, cnt = string.split(l)
        word_to_cluster[i][word] = cluster

if options.embedding is None: options.embedding = []
word_to_embedding = []
for i, embeddingfile in enumerate(options.embedding):
    print >> sys.stderr, "Reading Embedding file: %s" % embeddingfile
    word_to_embedding.append({})
    assert len(word_to_embedding) == i+1
    if len(options.embeddingscale) == 1:
        scale = options.embeddingscale[0]
    else:
        assert len(options.embeddingscale) == len(options.embedding)
        scale = options.embeddingscale[i]
    for l in myopen(embeddingfile):
        sp = string.split(l)
        word_to_embedding[i][sp[0]] = [float(v)*scale for v in sp[1:]]

    embedding_length = word_to_embedding[i].values()[0].__len__()
    word_to_embedding[i]['*UNKNOWN*'] = [0.]*embedding_length

def output_features(fo, seq):
    for i in range(2, len(seq)-2):
        fs = []

        fs.append('U00=%s' % seq[i-2][0])
        fs.append('U01=%s' % seq[i-1][0])
        fs.append('U02=%s' % seq[i][0])
        fs.append('U03=%s' % seq[i+1][0])
        fs.append('U04=%s' % seq[i+2][0])
        fs.append('U05=%s/%s' % (seq[i-1][0], seq[i][0]))
        fs.append('U06=%s/%s' % (seq[i][0], seq[i+1][0]))

        for j, cluster in enumerate(word_to_cluster):
            for name, pos in zip(["U00", "U01", "U02", "U03", "U04"], [i-2,i-1,i,i+1,i+2]):
                if seq[pos][0] not in cluster: continue
                for p in prefixes:
                    fs.append("%sbp%d-%d=%s" % (name, j, p, cluster[seq[pos][0]][:p]))
            if options.compound_representation_features:
                for name, poss in zip(["U15", "U16", "U17", "U18", "U20", "U21", "U22"], [(i-2,i-1), (i-1,i), (i, i+1), (i+1, i+2), (i-2, i-1, i), (i-1, i, i+1), (i, i+1, i+2)]):
                    cs = []
                    for pos in poss:
                        if seq[pos][0] not in cluster: continue
                        cs.append(cluster[seq[pos][0]])
                    if len(cs) < len(poss): continue
                    assert len(cs) == len(poss)
                    for p in prefixes:
                        fs.append("%sbp%d-%d=%s" % (name, j, p, string.join([c[:p] for c in cs], sep="/")))

        for j, embedding in enumerate(word_to_embedding):
            for name, pos in zip(["U00", "U01", "U02", "U03", "U04"], [i-2,i-1,i,i+1,i+2]):
                w = seq[pos][0]
                if w not in embedding: w = "*UNKNOWN*"
                for d in range(len(embedding[w])):
                    fs.append("%se%d-%d=1:%g" % (name, j, d, embedding[w][d]))
            if options.compound_representation_features:
#                for name, poss in zip(["U15", "U16", "U17", "U18", "U20", "U21", "U22"], [(i-2,i-1), (i-1,i), (i, i+1), (i+1, i+2), (i-2, i-1, i), (i-1, i, i+1), (i, i+1, i+2)]):
                for name, poss in zip(["U15", "U16", "U17", "U18", "U20", "U21", "U22"], [(i-2,i-1), (i-1,i), (i, i+1), (i+1, i+2), (i-2, i-1, i), (i-1, i, i+1), (i, i+1, i+2)]):
                    es = []
                    for pos in poss:
                        w = seq[pos][0]
                        if w not in embedding: w = "*UNKNOWN*"
                        es.append(embedding[w])
                    assert len(es) == len(poss)

                    value = [1.]
                    #print "Orig:", value
                    import numpy
                    for e in es:
                        #print "value <-", e, "*", value
                        value = numpy.outer(value, e)
                        value = numpy.reshape(value, value.size)
                        #print value
                    # Transform the value back into the correct scale
                    # e.g. sqrt for product of two vectors
                    # We preserve the sign of the value, though
                    value = (numpy.abs(value) ** (1./len(es))) * ((value > 0)*2-1)

                    # Threshold values
                    value = value * (numpy.abs(value) >= float(options.compound_embedding_threshold))
                    for d in range(len(value)):
                        if value[d] == 0: continue
                        fs.append("%se%d-%d=1:%g" % (name, j, d, value[d]))
#                    print value.shape
#                    print value

        if not options.no_pos_features:
            fs.append('U10=%s' % seq[i-2][1])
            fs.append('U11=%s' % seq[i-1][1])
            fs.append('U12=%s' % seq[i][1])
            fs.append('U13=%s' % seq[i+1][1])
            fs.append('U14=%s' % seq[i+2][1])
            fs.append('U15=%s/%s' % (seq[i-2][1], seq[i-1][1]))
            fs.append('U16=%s/%s' % (seq[i-1][1], seq[i][1]))
            fs.append('U17=%s/%s' % (seq[i][1], seq[i+1][1]))
            fs.append('U18=%s/%s' % (seq[i+1][1], seq[i+2][1]))

            fs.append('U20=%s/%s/%s' % (seq[i-2][1], seq[i-1][1], seq[i][1]))
            fs.append('U21=%s/%s/%s' % (seq[i-1][1], seq[i][1], seq[i+1][1]))
            fs.append('U22=%s/%s/%s' % (seq[i][1], seq[i+1][1], seq[i+2][1]))

        fo.write('%s\t%s\n' % (seq[i][2], '\t'.join(fs)))
    fo.write('\n')

def encode(x):
    x = x.replace('\\', '\\\\')
    x = x.replace(':', '\\:')
    return x

fi = open('train_lqrz.txt', 'rb')
fo = open('converted_train.txt', 'wb')

d = ('', '', '')

seq = [d, d]
for line in fi:
    line = line.strip('\n')
    if not line:
        seq.append(d)
        seq.append(d)
        output_features(fo, seq)
        seq = [d, d]
    else:
        fields = line.strip('\n').split(' ')
        seq.append((encode(fields[0]), encode(fields[1]), encode(fields[2])))
