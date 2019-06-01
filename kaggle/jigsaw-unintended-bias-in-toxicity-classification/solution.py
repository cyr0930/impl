# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import Callback, EarlyStopping
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

input_dir = '../../data/jigsaw-unintended-bias-in-toxicity-classification/'
output_dir = '../../data/dummy/'
glove_dir = '../../data/glove/'


class AucCallback(Callback):
    def __init__(self, x_train, x_val, y_train, y_val):
        super().__init__()
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        if output_dir:
            # y_pred = self.model.predict_proba(self.x_train)
            # roc = roc_auc_score(self.y_train >= 0.5, y_pred)
            # print('AUC_train: ' + str(round(roc, 4)))
            y_pred_val = self.model.predict_proba(self.x_val)
            roc_val = roc_auc_score(self.y_val >= 0.5, y_pred_val)
            print('AUC_val: ' + str(round(roc_val, 4)))
        return


nrows = None if output_dir else 100
data = pd.read_csv(input_dir + 'train.csv', nrows=nrows)
training_ratio = 0.8 if output_dir else 1
ntrains = 2 ** math.floor(math.log2(data.shape[0] * 0.8))
train_data = data[:ntrains]
validation_data = data[ntrains:]
test_data = pd.read_csv(input_dir + 'test.csv')
# train_data.sample(frac=1)
X_train, X_val, X_test = train_data['comment_text'], validation_data['comment_text'], test_data['comment_text']
Y_train, Y_val = train_data['target'], validation_data['target']

vocab_size = 100
embedding_dim = 16
batch_size = 512
epoch = 3
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

maxlen = max(max(map(lambda x: len(x), X_train)), 200)
X_train = pad_sequences(X_train, maxlen=maxlen)
X_val = pad_sequences(X_val, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

num_words = min(vocab_size, len(tokenizer.word_index)) + 1

l = []
l.append(layers.Embedding(num_words, embedding_dim, input_length=maxlen,))
l.append(layers.GRU(16))
l.append(layers.Dense(1, activation='sigmoid'))
model = keras.Sequential(l)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size, verbose=2,
          callbacks=[AucCallback(X_train, X_val, Y_train, Y_val), EarlyStopping(monitor='val_loss', patience=3)])

pred = model.predict(X_test)
with open(output_dir + 'submission.csv', 'w') as f:
    f.write('id,prediction\n')
    for i in range(len(pred)):
        f.write(str(test_data['id'].loc[i]) + ',' + str(pred[i][0]) + '\n')
