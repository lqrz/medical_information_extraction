__author__ = 'lqrz'

from itertools import chain
from nltk.corpus import stopwords

from data.dataset import Dataset

# these punctuation symbols were taken from string.punctuation
PUNCTUATION = ['!','"',"'",'(',')',',','-','.','/',':',';','<','?','[',']','\\','`','{','|','}']

LEFT = ['_', '__', '^', '@','=','>','*','+','&','$','%','#']

if __name__ == '__main__':
    training_data, sentences, train_document_sentence_words, train_document_sentence_tags = Dataset.get_clef_training_dataset()
    training_data, sentences, valid_document_sentence_words, valid_document_sentence_tags = Dataset.get_clef_validation_dataset()
    training_data, sentences, test_document_sentence_words, _ = Dataset.get_clef_testing_dataset()

    train_tokens = [w.lower() for w in list(chain(*chain(*train_document_sentence_words.values())))
                    if w not in PUNCTUATION]
    train_tokens_out = [w.lower() for w in list(chain(*chain(*train_document_sentence_words.values())))
                        if w in PUNCTUATION]
    train_unique_tokens = set(train_tokens)
    train_documents = train_document_sentence_words.__len__()

    valid_tokens = [w.lower() for w in list(chain(*chain(*valid_document_sentence_words.values())))
                    if w not in PUNCTUATION]
    valid_tokens_out = [w.lower() for w in list(chain(*chain(*valid_document_sentence_words.values())))
                        if w in PUNCTUATION]
    valid_unique_tokens = set(valid_tokens)
    valid_documents = valid_document_sentence_words.__len__()

    test_tokens = [w.lower() for w in list(chain(*chain(*test_document_sentence_words.values())))
                   if w not in PUNCTUATION]
    test_tokens_out = [w.lower() for w in list(chain(*chain(*test_document_sentence_words.values())))
                       if w in PUNCTUATION]
    test_unique_tokens = set(test_tokens)
    test_documents = test_document_sentence_words.__len__()

    valid_train_overlap = valid_unique_tokens.intersection(train_unique_tokens)
    test_train_overlap = test_unique_tokens.intersection(train_unique_tokens)

    stop_words = stopwords.words('english')

    tokens_out = set(train_tokens_out+valid_tokens_out+test_tokens_out)

    valid_without_sw = [w for w in valid_train_overlap if w not in stop_words]
    test_without_sw = [w for w in test_train_overlap if w not in stop_words]

    train_tags = set(chain(*chain(*train_document_sentence_tags.values())))
    valid_tags = set(chain(*chain(*valid_document_sentence_tags.values())))

    print '...End'