__author__ = 'root'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from data.dataset import Dataset
from crf_sklearn_crfsuite import CRF
from sklearn.cross_validation import LeaveOneOut
from trained_models import get_crpp_path
import subprocess
import utils
from utils.metrics import Metrics

def save_to_file(features, labels, doc_nrs, file_name):
    f_out = open(file_name, 'w')

    feats_to_save = ['word', 'word_lemma', 'word_ner', 'word_pos', 'word_parse_tree', 'word_dependents',
                     'word_governors', 'word_unk_score', 'word_phrase', 'word_candidate_1', 'word_candidate_2', 'word_candidate_3',
                     'word_candidate_4', 'word_candidate_5', 'word_mapping', 'word_medication_score',
                     'word_location']

    for doc_nr,sentences_feats in features.iteritems():
        if doc_nr not in doc_nrs:
            continue

        for sent_nr, sentence_feats in enumerate(sentences_feats):
            for word_nr, word_feats in enumerate(sentence_feats):
                feat_list = [word_feats[feat] for feat in feats_to_save]
                if labels:
                    feat_list.append(labels[doc_nr][sent_nr][word_nr])
                f_out.write('\t'.join(map(str,feat_list))+'\n')
            f_out.write('\n')   #separate sentences

    f_out.close()

    return


if __name__=='__main__':
    """
    This is for the original paper features.
    """

    training_data_filename = 'handoverdata.zip'

    training_data, training_texts, _, _ = Dataset.get_crf_training_data_by_sentence(training_data_filename)

    crf_model = CRF(training_data, training_texts,
                    test_data=None,
                    output_model_filename=None,
                    w2v_vector_features=False,
                    w2v_similar_words=False,
                    kmeans_features=False,
                    lda_features=False,
                    zip_features=False,
                    original_inc_unk_score=True,
                    original_include_metamap=True,
                    w2v_model=None,
                    w2v_vectors_dict=None)

    print 'Getting features'
    x = crf_model.get_features_from_crf_training_data(crf_model.get_original_paper_word_features)
    print 'Getting labels'
    y = crf_model.get_labels_from_crf_training_data()

    crfpp_model_filename = get_crpp_path('output.model')
    crfpp_train_filename = get_crpp_path('train_features.tmp.txt')
    crfpp_test_filename = get_crpp_path('test_features.tmp.txt')
    crfpp_template_filename = get_crpp_path('template.bigram')
    crfpp_template_filename = get_crpp_path('template.original')
    crfpp_test_output_filename = get_crpp_path('test_out.tmp.txt')
    conlleval_filename = utils.get_filename('conlleval.pl')

    f_train = True
    f_predict = True
    f_conlleval = False

    max_iters = 50

    y_predictions = []
    y_true = []

    accuracy_results = []
    f1_score_results = []
    precision_results = []
    recall_results = []

    loo = LeaveOneOut(training_data.__len__())
    for i, (x_idx, y_idx) in enumerate(loo):

        # if i > 0:
        #     break

        print 'Cross-validation: %d' % i

        save_to_file(x, labels=y, doc_nrs=x_idx, file_name=crfpp_train_filename)
        save_to_file(x, labels=y, doc_nrs=y_idx, file_name=crfpp_test_filename)

        if f_train:
            print 'Running CRF++ train'
            return_code = subprocess.call(['crf_learn', '-m', str(max_iters), '--algorithm=CRF', '-e 0.000010','-p', '2', crfpp_template_filename, crfpp_train_filename, crfpp_model_filename])
        if f_predict:
            print 'Running CRF++ predict'
            p = subprocess.Popen(['crf_test', '-v 0', '-m', crfpp_model_filename, crfpp_test_filename],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
            out,err = p.communicate()

            f_out = open(crfpp_test_output_filename,'w')
            for line in out.split('\n'):
                if line:
                    y_predictions.append(line.split('\t')[-1])
                    y_true.append(line.split('\t')[-2])
                    f_out.write(line+'\n')
                else:
                    # f_out.write('\n')
                    continue
            f_out.close()

        if f_conlleval:
            print 'Running Conlleval script'
            p = subprocess.Popen(['perl',conlleval_filename,' -r -d "\t" < '+crfpp_test_output_filename])
            out,err = p.communicate()

    assert y_predictions.__len__() == y_true.__len__()

    print '## MICRO results'
    print Metrics.compute_all_metrics(y_true, y_predictions, average='micro')
    print '## MACRO results'
    print Metrics.compute_all_metrics(y_true, y_predictions, average='macro')

    print 'End'