import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle


def load_data():
    train = _parse_data(open('data/train_data.data', 'rb'))
    test = _parse_data(open('data/test_data.data', 'rb'))

    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]

    # save initial config data
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags)


def _parse_data(fh):
    string = fh.read().decode('utf-8')
    data = [[row.split() for row in sample.split('\r\n')] for
            sample in
            string.strip().split('\r\n\r\n')]
    fh.close()
    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    # y_pos = [[pos_tags.index(w[1]) for w in s] for s in data]
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    # y_pos = pad_sequences(y_pos, maxlen, value=-1)  # lef padded with -1. Indeed, any integer works as it will be masked
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        # y_pos = numpy.eye(len(pos_tags), dtype='float32')[y_pos]
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        # y_pos = numpy.expand_dims(y_pos, 2)
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length
