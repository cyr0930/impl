import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import reduce
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

def draw_graph(fig, overall, l, idx):
    ax = fig.add_subplot(1, 3, idx)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(overall[0], overall[1], label='overall')
    for t in l:
        ax.plot(t[0], t[1], label=t[2])
    ax.legend(loc='lower left')
def draw_roc_curve(df, model_name, label):
    l_sub, l_bpsn, l_bnsp = [], [], []
    for subgroup in identity_columns:
        subgroup_examples = df[df[subgroup]]
        subgroup_negative_examples = df[df[subgroup] & ~df[label]]
        non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
        bpsn_examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        subgroup_positive_examples = df[df[subgroup] & df[label]]
        non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
        bnsp_examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        sub_fp, sub_tp, _ = metrics.roc_curve(subgroup_examples[label], subgroup_examples[model_name])
        bpsn_fp, bpsn_tp, _ = metrics.roc_curve(bpsn_examples[label], bpsn_examples[model_name])
        bnsp_fp, bnsp_tp, _ = metrics.roc_curve(bnsp_examples[label], bnsp_examples[model_name])
        l_sub.append((sub_fp, sub_tp, subgroup[:4]))
        l_bpsn.append((bpsn_fp, bpsn_tp, 'bpsn_' + subgroup[:4]))
        l_bnsp.append((bnsp_fp, bnsp_tp, 'bnsp_' + subgroup[:4]))
    overall = metrics.roc_curve(df[label], df[model_name])
    fig = plt.figure(figsize=(18, 5))
    draw_graph(fig, overall, l_sub, 1)
    draw_graph(fig, overall, l_bpsn, 2)
    draw_graph(fig, overall, l_bnsp, 3)
    fig.savefig(output_dir + 'roc' + str(time.time()) + '.png')
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
        record = {'subgroup': subgroup[:4], 'size': len(dataset[dataset[subgroup]]),
                  'subgroup_auc': compute_subgroup_auc(dataset, subgroup, label_col, model),
                  'bpsn_auc': compute_bpsn_auc(dataset, subgroup, label_col, model),
                  'bnsp_auc': compute_bnsp_auc(dataset, subgroup, label_col, model)}
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc')
def calculate_overall_auc(df, model_name, label_col):
    true_labels = df[label_col]
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


pd.set_option('mode.chained_assignment', None)
nrows = None if is_submit else 100000
data = pd.read_csv(input_dir + 'train.csv', nrows=nrows)
data = data.sample(frac=1)
for col in identity_columns:
    data.loc[:, col] = data[col] >= 0.5
training_ratio = 1 if is_submit else 0.8
ntrains = 2 ** math.floor(math.log2(data.shape[0] * training_ratio))
train_data = data[:ntrains]

for i in range(1):
    buf = train_data[reduce(lambda x, y: x & y, [train_data[i] == 0 for i in identity_columns])]
    train_data = train_data.append(buf[buf['target'] >= 0.5])

val_data = data[ntrains:]
test_data = pd.read_csv(input_dir + 'test.csv')

target_name = 'target_b'
train_data.loc[:, target_name] = train_data['target'] >= 0.5
val_data.loc[:, target_name] = val_data['target'] >= 0.5

vocab_size = 100000
batch_size = 512
epoch = 10
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>')
tokenizer.fit_on_texts(train_data['comment_text'])

num_words = min(vocab_size, len(tokenizer.word_index))
start_token, end_token = num_words, num_words+1
train_data['comment_text'] = add_start_end_token(tokenizer.texts_to_sequences(train_data['comment_text']), start_token, end_token)
val_data['comment_text'] = add_start_end_token(tokenizer.texts_to_sequences(val_data['comment_text']), start_token, end_token)
test_data['comment_text'] = add_start_end_token(tokenizer.texts_to_sequences(test_data['comment_text']), start_token, end_token)

X_val = pad_sequences(val_data['comment_text'], maxlen=max(map(lambda x: len(x), val_data['comment_text'])))
X_test = pad_sequences(test_data['comment_text'], maxlen=max(map(lambda x: len(x), test_data['comment_text'])))
num_words += 3  # start token, end token, padding

embedding_dim = 300
glove_matrix = construct_glove_matrix(tokenizer.word_index, embedding_dim, num_words, vocab_size)

dropout_rate = 0.5
input_layer = keras.Input(shape=(None,))
output_layer = layers.Embedding(num_words, embedding_dim, embeddings_initializer=Constant(glove_matrix),
                                trainable=False)(input_layer)
output_layer = layers.GRU(512, dropout=dropout_rate)(output_layer)
output_layer = layers.Dense(128, activation='relu')(output_layer)
output_layer = layers.Dropout(dropout_rate)(output_layer)
output_layer = layers.Dense(1, activation='sigmoid')(output_layer)
model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy')

model_name = 'my_model'
for i in range(epoch):
    t = time.time()
    agg_loss = 0
    train_data = train_data.sample(frac=1)
    X_train = train_data['comment_text']
    Y_train = train_data['target']
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
            train_data.loc[:, model_name] = model.predict(X_train_eval)
            bias_metrics_df = compute_bias_metrics_for_model(train_data, identity_columns, model_name, target_name)
            overall_auc = calculate_overall_auc(train_data, model_name, target_name)
            score = get_final_metric(bias_metrics_df, overall_auc)
            print(bias_metrics_df)
            print('Score: %.4f, Overall AUC: %.4f (Training)' % (score, overall_auc))
        if nrows is not None or (nrows is None and i == epoch-1):
            val_data.loc[:, model_name] = model.predict(X_val)
            bias_metrics_df_val = compute_bias_metrics_for_model(val_data, identity_columns, model_name, target_name)
            overall_auc_val = calculate_overall_auc(val_data, model_name, target_name)
            score_val = get_final_metric(bias_metrics_df_val, overall_auc_val)
            print(bias_metrics_df_val)
            print('Score: %.4f, Overall AUC: %.4f (Validation) Time: %.4f' % (score_val, overall_auc_val, time.time()-t))
            if i == epoch - 1:
                draw_roc_curve(val_data, model_name, target_name)

if is_submit:
    pred = model.predict(X_test)
    with open(output_dir + 'submission.csv', 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(pred)):
            f.write(str(test_data['id'].loc[i]) + ',' + str(pred[i][0]) + '\n')
