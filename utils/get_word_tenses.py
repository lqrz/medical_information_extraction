__author__ = 'lqrz'

from data.dataset import Dataset

pos_tense_mapping = {
    'VB': 'present',
    'VBD': 'past',
    'VBG': 'present',
    'VBN': 'past',
    'VBP': 'present',
    'VBZ': 'present',
    '#NA': 'NA',
    '#FUTURE': 'future'
}

def determine_pos(previous_pos, word_dict, governor_word_dict, sent_dicts, i):
    global pos_tense_mapping

    pos_tag = word_dict['features'][2]
    if pos_tag in pos_tense_mapping.keys():
        if previous_pos == 'VBZ':
            # then this refers to present perfect
            return 'VBZ'
        else:
            return pos_tag
    elif pos_tag == 'MD' and word_dict['word'] == 'will':
        return '#FUTURE'
    else:
        if governor_word_dict:
            # continue with the governor
            governor_governor_surface = governor_word_dict['features'][5]
        else:
            # punctuations have no governors
            return '#NA'

        governor_ix = [j for j,dic in enumerate(sent_dicts) if dic['word'] == governor_word_dict['word']][0]
        previous_pos = sent_dicts[governor_ix-1]['features'][2]
        try:
            governor_governor_dict = [dic for dic in sent_dicts if dic['word'] == governor_governor_surface][0]
        except IndexError:
            governor_governor_dict = []

        return determine_pos(previous_pos, governor_word_dict, governor_governor_dict, sent_dicts, i+1)


def get_dataset_sentence_tenses(dataset_dicts):
    global pos_tense_mapping

    sent_tense = dict()
    dataset_sentences = []

    for j, sent_dicts in enumerate(dataset_dicts):
        dataset_sentences.append([w['word'] for w in sent_dicts])
        verb_pos_in_sentence = [word['features'][2] for word in sent_dicts if
                                word['features'][2] in pos_tense_mapping.keys()]

        if not verb_pos_in_sentence:
            # i didnt find a verb, everything is present by default.
            sent_tense[j] = ['present'] * sent_dicts.__len__()
        elif verb_pos_in_sentence.__len__() == 1:
            sent_tense[j] = [pos_tense_mapping[verb_pos_in_sentence[0]]] * sent_dicts.__len__()
            if verb_pos_in_sentence[0] == 'MD':
                print 'debug'
        else:
            # if there is more than one, determine each assignment.
            previous_pos = None
            word_tense = []
            for i, word_dict in enumerate(sent_dicts):
                governor_surface = word_dict['features'][5]
                try:
                    governor_dict = [dic for dic in sent_dicts if dic['word'] == governor_surface][0]
                except IndexError:
                    governor_dict = []

                if i > 0:
                    previous_pos = sent_dicts[i - 1]['features'][2]

                pos_tag = determine_pos(previous_pos, word_dict, governor_dict, sent_dicts, i)
                tense = pos_tense_mapping[pos_tag]
                if pos_tag == 'MD':
                    print 'debug'
                word_tense.append(tense)
                previous_pos = tense
            sent_tense[j] = word_tense

    return sent_tense

def get_training_set_tenses():
    training_data, _, _, _ = Dataset.get_clef_training_dataset()
    training_dicts = [sent for doc in training_data.values() for sent in doc]
    return get_dataset_sentence_tenses(training_dicts)

def get_validation_set_tenses():
    validation_data, _, _, _ = Dataset.get_clef_validation_dataset()
    validation_dicts = [sent for doc in validation_data.values() for sent in doc]
    return get_dataset_sentence_tenses(validation_dicts)

def get_testing_set_tenses():
    testing_data, _, _, _ = Dataset.get_clef_testing_dataset()
    testing_dicts = [sent for doc in testing_data.values() for sent in doc]
    return get_dataset_sentence_tenses(testing_dicts)

if __name__ == '__main__':
    get_training_set_tenses()

    # [(word['word'], word['features'][5], word['tag']) for word in sent_dicts]
    # [(f['word'], f['features'][2], f['features'][4], f['features'][5]) for f in training_data[0][7]]

    print 'End'

