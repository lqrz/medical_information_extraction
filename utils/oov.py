___author__ = 'lqrz'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from itertools import chain
import subprocess
import cPickle
from collections import Counter, defaultdict

from data.dataset import Dataset
from data import get_mythes_checkfile
from data import get_mythes_lookup_path
from data import get_mythes_english_thesaurus_index_path
from data import get_mythes_english_thesaurus_data_path
from data import get_mythes_oov_replacements_path


def oov_replacement():
    _, _, training_tokens, _ = Dataset.get_clef_training_dataset()
    _, _, validation_tokens, _ = Dataset.get_clef_validation_dataset()

    training_tokens = [w.lower() for w in list(chain(*chain(*training_tokens.values())))]
    validation_tokens = [w.lower() for w in list(chain(*chain(*validation_tokens.values())))]

    oovs = set(validation_tokens).difference(set(training_tokens))

    checklist_filename = 'oovs.txt'
    checklist_path = get_mythes_checkfile(checklist_filename)
    fout = open(checklist_path, 'wb')
    for oov in oovs:
        fout.write(oov + '\n')
    fout.close()

    lookup_pgm_path = get_mythes_lookup_path()
    lookup_index_path = get_mythes_english_thesaurus_index_path()
    lookup_data_path = get_mythes_english_thesaurus_data_path()
    params = ' '.join([lookup_pgm_path, lookup_index_path, lookup_data_path, checklist_path])
    p = subprocess.Popen([params],
                         stdout=subprocess.PIPE,
                         shell=True)
    # p = subprocess.Popen(['ls', '-l'],
    #                      stdout=subprocess.PIPE,
    #                      shell=True)
    std_out, std_error = p.communicate()

    if std_error is not None:
        print 'Couldnt launch MyThes lookup script'
        exit(0)

    replacements = dict()
    synonym_count = 0
    chosen_count = 0
    for line in std_out.split('\n'):
        synonym_count += 1
        words = line.split('\t')
        oov = words[0]
        synonyms = [w.lower() for w in set(words[1:]).difference('')]
        chosen = None
        for syn in synonyms:
            if syn in training_tokens:
                chosen = syn
                chosen_count += 1
                break

        if chosen is not None:
            replacements[oov] = chosen

    out_path = open(get_mythes_oov_replacements_path(), 'wb')
    cPickle.dump(replacements, out_path)

    print 'Out of the %d OOVs, %d had synonyms in the Thesaurus, and %d had synonyms in the training data' % \
          (oovs.__len__(), synonym_count, chosen_count)

    return

def data_augmentation():
    _, _, training_tokens, training_tags = Dataset.get_clef_training_dataset()
    _, _, validation_tokens, validation_tags = Dataset.get_clef_validation_dataset()

    training_tokens = [w.lower() for w in list(chain(*(chain(*training_tokens.values()))))]
    training_tags = list(chain(*(chain(*training_tags.values()))))

    validation_tokens = [w.lower() for w in list(chain(*(chain(*validation_tokens.values()))))]
    validation_tags = list(chain(*(chain(*validation_tags.values()))))

    training_tags_tokens = defaultdict(list)
    training_tags_tokens_positions = defaultdict(list)
    for i, (tag, token) in enumerate(zip(training_tags, training_tokens)):
        training_tags_tokens[tag].append(token)
        training_tags_tokens_positions[(tag,token)].append(i)

    tag_count = Counter(training_tags)

    na_count = tag_count.most_common(n=1)[0][1]

    tmp_path = get_mythes_checkfile('check.tmp')
    fout = open(tmp_path, 'wb')

    lookup_pgm_path = get_mythes_lookup_path()
    lookup_index_path = get_mythes_english_thesaurus_index_path()
    lookup_data_path = get_mythes_english_thesaurus_data_path()
    params = ' '.join([lookup_pgm_path, lookup_index_path, lookup_data_path, tmp_path])
    p = subprocess.Popen([params],
                         stdout=subprocess.PIPE,
                         shell=True)

    for tag, cnt in tag_count.iteritems():
        to_add = na_count - cnt
        tokens = list(set(training_tags_tokens[tag]))
        fout.write('\n'.join(tokens))
        fout.close()
        std_out, std_err = p.communicate()

        if std_err is not None:
            print 'Error while running the lookup script'
            exit(0)

        for i, line in enumerate(std_out.split('\n')):
            token = line.split('\t')[0]
            synonyms = [w for w in line.split('\t')[1:] if w.split(' ').__len__() == 1 and w != '']


if __name__ == '__main__':
    # oov_replacement()

    data_augmentation()