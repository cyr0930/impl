import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

input_dir = '../../data/jigsaw-unintended-bias-in-toxicity-classification/'
output_dir = '../../data/dummy/'
glove_dir = '../../data/glove/'
is_submit = not output_dir
# tf.enable_eager_execution()

identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return metrics.roc_auc_score(subgroup_examples[label], subgroup_examples[model_name])
def compute_bpsn_auc(df, subgroup, label, model_name):
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return metrics.roc_auc_score(examples[label], examples[model_name])
def compute_bnsp_auc(df, subgroup, label, model_name):
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return metrics.roc_auc_score(examples[label], examples[model_name])
def compute_bias_metrics_for_model(dataset, subgroups, model, label_col):
    records = []
    for subgroup in subgroups:
        record = {'subgroup': subgroup, 'subgroup_size': len(dataset[dataset[subgroup]]),
                  'subgroup_auc': compute_subgroup_auc(dataset, subgroup, label_col, model),
                  'bpsn_auc': compute_bpsn_auc(dataset, subgroup, label_col, model),
                  'bnsp_auc': compute_bnsp_auc(dataset, subgroup, label_col, model)}
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
def calculate_overall_auc(df, model_name):
    true_labels = df['target_b']
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)
def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)
def get_final_metric(bias_df, overall_auc):
    bias_score = np.average([power_mean(bias_df['subgroup_auc'], -5), power_mean(bias_df['bpsn_auc'], -5),
                             power_mean(bias_df['bnsp_auc'], -5)])
    return 0.25 * overall_auc + 0.75 * bias_score

def construct_glove_matrix(word_idx, d, n, v):     # d: embedding_dim, n: num_words, v: vocab_size
    embeddings_index = {}
    with open(glove_dir + 'glove.6B.' + str(d) + 'd.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((n, d))
    for word, i in word_idx.items():
        if i > v:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
def add_start_end_token(X, start, end):
    X = [[start] + x for x in X]
    for x in X:
        x.append(end)
    return X

class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(units)
    def call(self, inputs, **kwargs):
        values, query = inputs
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


pd.set_option('mode.chained_assignment', None)
nrows = None if is_submit else 100000
data = pd.read_csv(input_dir + 'train.csv', nrows=nrows)
data.sample(frac=1)
for col in identity_columns:
    data.loc[:, col] = data[col] >= 0.5
training_ratio = 1 if is_submit else 0.8
ntrains = 2 ** math.floor(math.log2(data.shape[0] * training_ratio))
train_data = data[:ntrains]
validation_data = data[ntrains:]
test_data = pd.read_csv(input_dir + 'test.csv')
X_train, X_val, X_test = train_data['comment_text'], validation_data['comment_text'], test_data['comment_text']
Y_train = train_data['target']

train_data.loc[:, 'target_b'] = train_data['target'] >= 0.5
validation_data.loc[:, 'target_b'] = validation_data['target'] >= 0.5

vocab_size = 100000
batch_size = 512
epoch = 10
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>')
tokenizer.fit_on_texts(X_train)

num_words = min(vocab_size, len(tokenizer.word_index))
start_token, end_token = num_words, num_words+1
X_train = add_start_end_token(tokenizer.texts_to_sequences(X_train), start_token, end_token)
X_val = add_start_end_token(tokenizer.texts_to_sequences(X_val), start_token, end_token)
X_test = add_start_end_token(tokenizer.texts_to_sequences(X_test), start_token, end_token)

X_val = pad_sequences(X_val, maxlen=max(map(lambda x: len(x), X_val)))
X_test = pad_sequences(X_test, maxlen=max(map(lambda x: len(x), X_test)))
num_words += 3  # start token, end token, padding

embedding_dim = 300
glove_matrix = construct_glove_matrix(tokenizer.word_index, embedding_dim, num_words, vocab_size)

input_layer = keras.Input(shape=(None,))
output_layer = layers.Embedding(num_words, embedding_dim, embeddings_initializer=Constant(glove_matrix),
                                trainable=False)(input_layer)
output_layer = layers.GRU(128, return_sequences=True, return_state=True)(output_layer)
output_layer = BahdanauAttention(128)(output_layer)
output_layer = layers.Dense(32, activation='relu')(output_layer)
output_layer = layers.Dense(1, activation='sigmoid')(output_layer)
model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy')

for i in range(epoch):
    t = time.time()
    agg_loss = 0
    print('epoch: %d' % i)
    for j in range(0, ntrains, batch_size):
        X_batch = X_train[j:j+batch_size]
        Y_batch = Y_train[j:j+batch_size]
        X_batch = pad_sequences(X_batch, maxlen=max(map(lambda x: len(x), X_batch)))
        agg_loss += model.train_on_batch(X_batch, Y_batch.values)
    print('Loss: %.4f, Time: %.4f' % (agg_loss, time.time() - t))
    if not is_submit:
        t = time.time()
        if nrows is not None:
            X_train_eval = pad_sequences(X_train, maxlen=max(map(lambda x: len(x), X_train)))
            train_data.loc[:, 'my_model'] = model.predict(X_train_eval)
            bias_metrics_df = compute_bias_metrics_for_model(train_data, identity_columns, 'my_model', 'target_b')
            # print(bias_metrics_df)
            overall_auc = calculate_overall_auc(train_data, 'my_model')
            score = get_final_metric(bias_metrics_df, overall_auc)
            print('Score: %.4f, Overall AUC: %.4f (Training)' % (score, overall_auc))
        if nrows is not None or (nrows is None and i == epoch-1):
            validation_data.loc[:, 'my_model'] = model.predict(X_val)
            bias_metrics_df_val = compute_bias_metrics_for_model(validation_data, identity_columns, 'my_model', 'target_b')
            # print(bias_metrics_df_val)
            overall_auc_val = calculate_overall_auc(validation_data, 'my_model')
            score_val = get_final_metric(bias_metrics_df_val, overall_auc_val)
            print('Score: %.4f, Overall AUC: %.4f (Validation) Time: %.4f' % (score_val, overall_auc_val, time.time()-t))

if is_submit:
    pred = model.predict(X_test)
    with open(output_dir + 'submission.csv', 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(pred)):
            f.write(str(test_data['id'].loc[i]) + ',' + str(pred[i][0]) + '\n')
