import time
import math
import pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

input_dir = '../../data/jigsaw-unintended-bias-in-toxicity-classification/'
output_dir = '../../data/dummy/'
is_submit = not output_dir

nrows = None if is_submit else 100000
data = pd.read_csv(input_dir + 'train.csv', nrows=nrows)
training_ratio = 1 if is_submit else 0.8
ntrains = 2 ** math.floor(math.log2(data.shape[0] * training_ratio))
train_data = data[:ntrains]
validation_data = data[ntrains:]
test_data = pd.read_csv(input_dir + 'test.csv')
# train_data.sample(frac=1)
X_train, X_val, X_test = train_data['comment_text'], validation_data['comment_text'], test_data['comment_text']
Y_train, Y_val = train_data['target'], validation_data['target'] >= 0.5

vocab_size = 30000
embedding_dim = 16
batch_size = 512
epoch = 3
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

X_val = pad_sequences(X_val, maxlen=max(map(lambda x: len(x), X_val)))
X_test = pad_sequences(X_test, maxlen=max(map(lambda x: len(x), X_test)))

num_words = min(vocab_size, len(tokenizer.word_index)) + 1

l = []
l.append(layers.Embedding(num_words, embedding_dim))
l.append(layers.GRU(16, return_sequences=True))
l.append(layers.GRU(4))
l.append(layers.Dense(1, activation='sigmoid'))
model = keras.Sequential(l)
model.compile(optimizer='adam', loss='binary_crossentropy')

for i in range(epoch):
    t = time.time()
    agg_loss = 0
    print('epoch: %d' % i)
    for j in range(0, ntrains, batch_size):
        X_batch = X_train[j:j+batch_size]
        Y_batch = Y_train[j:j+batch_size]
        X_batch = pad_sequences(X_batch, maxlen=max(map(lambda x: len(x), X_batch)))
        agg_loss += model.train_on_batch(X_batch, Y_batch, )
    if is_submit:
        print('Loss: %.4f, Time: %.4f' % (agg_loss, time.time()-t))
    else:
        y_pred_val = model.predict_proba(X_val)
        roc_val = roc_auc_score(Y_val, y_pred_val)
        print('AUC_val: %.4f, Loss: %.4f, Time: %.4f' % (roc_val, agg_loss, time.time()-t))

pred = model.predict(X_test)
with open(output_dir + 'submission.csv', 'w') as f:
    f.write('id,prediction\n')
    for i in range(len(pred)):
        f.write(str(test_data['id'].loc[i]) + ',' + str(pred[i][0]) + '\n')
