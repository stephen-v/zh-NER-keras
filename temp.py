from __future__ import print_function
import numpy
from keras.utils.data_utils import get_file
from zipfile import ZipFile
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import cifar10


def load_data(path='conll2000.zip', min_freq=2):
    path = get_file(path, origin='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/conll2000.zip')
    print(path)
    archive = ZipFile(path, 'r')
    train = _parse_data(archive.open('conll2000/train.txt'))
    test = _parse_data(archive.open('conll2000/test.txt'))
    archive.close()

    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = ['<pad>', '<unk>'] + [w for w, f in iter(word_counts.items()) if f >= min_freq]
    pos_tags = sorted(list(set(row[1] for sample in train + test for row in sample)))  # in alphabetic order
    chunk_tags = sorted(list(set(row[2] for sample in train + test for row in sample)))  # in alphabetic order

    train = _process_data(train, vocab, pos_tags, chunk_tags)
    test = _process_data(test, vocab, pos_tags, chunk_tags)
    return train, test, (vocab, pos_tags, chunk_tags)


def _parse_data(fh):
    string = fh.read()
    data = [[row.split() for row in sample.split('\n')] for sample in string.decode().strip().split('\n\n')]
    fh.close()
    return data


def _process_data(data, vocab, pos_tags, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_pos = [[pos_tags.index(w[1]) for w in s] for s in data]
    y_chunk = [[chunk_tags.index(w[2]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    y_pos = pad_sequences(y_pos, maxlen, value=-1)  # lef padded with -1. Indeed, any integer works as it will be masked
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_pos = numpy.eye(len(pos_tags), dtype='float32')[y]
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y]
    else:
        y_pos = numpy.expand_dims(y_pos, 2)
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_pos, y_chunk



import numpy
from collections import Counter

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF

EPOCHS = 10
EMBED_DIM = 200
BiRNN_UNITS = 200


def classification_report(y_true, y_pred, labels):
    '''Similar to the one in sklearn.metrics, reports per classs recall, precision and F1 score'''
    y_true = numpy.asarray(y_true).ravel()
    y_pred = numpy.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('', 'recall', 'precision', 'f1-score', 'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report])
    N = len(y_true)
    print(formatter('avg / total', sum(report2[0]) / N, sum(report2[1]) / N, sum(report2[2]) / N, N) + '\n')


(train_x, _, train_y), (test_x, _, test_y), (vocab, _, class_labels) = load_data()


model = Sequential()
model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
crf = CRF(len(class_labels), sparse_target=True)
model.add(crf)
model.summary()

model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])

test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
test_y_true = test_y[test_x > 0]

print('\n---- Result of BiLSTM-CRF ----\n')
classification_report(test_y_true, test_y_pred, class_labels)