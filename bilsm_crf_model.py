
import numpy
from collections import Counter

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
import process_data

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


(train_x, train_y), (test_x, test_y), (vocab, class_labels) = process_data.load_data()


model = Sequential()
model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
crf = CRF(len(class_labels), sparse_target=True)
model.add(crf)
model.summary()

model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])
model.save('model/crf.h5')


test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
test_y_true = test_y[test_x > 0]

print('\n---- Result of BiLSTM-CRF ----\n')
classification_report(test_y_true, test_y_pred, class_labels)