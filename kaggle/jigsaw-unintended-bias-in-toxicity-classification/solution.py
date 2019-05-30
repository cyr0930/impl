from tensorflow import keras
from tensorflow.python.keras import layers
import pandas as pd

root_dir = '../../data/jigsaw-unintended-bias-in-toxicity-classification'

train_data = pd.read_csv(root_dir + '/train.csv', nrows=100000)
test_data = pd.read_csv(root_dir + '/test.csv')
# train_data.sample(frac=1)
X_train, X_test = train_data['comment_text'], test_data['comment_text']
Y_train = train_data['target']

vocab_size = 10000
embedding_dim = 16
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<UNK>')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

maxlen = max(map(lambda x: len(x), X_train))
X_train = keras.preprocessing.sequence.pad_sequences(X_train, value=0, padding='post', maxlen=maxlen)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, value=0, padding='post', maxlen=maxlen)

model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mae'])
model.fit(X_train, Y_train, epochs=30, batch_size=512)

pred = model.predict(X_test)
with open('submission.csv', 'w') as f:
    f.write('id,prediction\n')
    for i in range(len(pred)):
        f.write(str(test_data['id'].loc[i]) + ',' + str(pred[i][0]) + '\n')
