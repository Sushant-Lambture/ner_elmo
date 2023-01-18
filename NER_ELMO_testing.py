# !python --version

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

data = pd.read_csv('train_final.csv') #, encoding="latin1")
# data=data.drop(['Unnamed: 0.1'],axis=1)
# data = data.drop(['POS'], axis =1)
# data = data.fillna(method="ffill")
data

words = set(list(data['word'].values))
a=words.add('PADword')
n_words = len(words)
n_words

label = list(set(data["label"].values))
n_tags = len(label)
n_tags

data['label'].value_counts()

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),s["label"].values.tolist())]
        self.grouped = self.data.groupby("Unnamed: 0").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Unnamed: 0: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)
# [('Thousands', 'O'), ('of', 'O'), ('demonstrators', 'O'), ('have', 'O'), ('marched', 'O'), ('through', 'O'), ('London', 'B-geo'), ('to', 'O'), ('protest', 'O'), ('the', 'O'), ('war', 'O'), ('in', 'O'), ('Iraq', 'B-geo'), ('and', 'O'), ('demand', 'O'), ('the', 'O'), ('withdrawal', 'O'), ('of', 'O'), ('British', 'B-gpe'), ('troops', 'O'), ('from', 'O'), ('that', 'O'), ('country', 'O'), ('.', 'O')]

sentences = getter.sentences
print(len(sentences))
# # 47959

largest_sen = max(len(sen) for sen in sentences)
print('biggest sentence has {} words'.format(largest_sen))

# %matplotlib inline
plt.hist([len(sen) for sen in sentences] , bins=50)
plt.show

max_len = 2
X = [[w[0]for w in s] for s in sentences]
new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("PADword")
    new_X.append(new_seq)
new_X[15]

# from keras.preprocessing.sequence import pad_sequence
from keras_preprocessing.sequence import pad_sequences
tags2index = {t:i for i,t in enumerate(label)}
y = [[tags2index[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post")
y[15]
#, value=tags2index['0']

from sklearn.model_selection import train_test_split
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from keras import backend as K

X_tr, X_te, y_tr, y_te = train_test_split(new_X, y, test_size=0.1, random_state=2018)

sess = tf.Session()
K.set_session(sess)
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

sess.run(tf.global_variables_initializer())
# sess.run(tf1.compat.v1.global_variables_initializer())

sess.run(tf.tables_initializer())

batch_size = 32
def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x,    tf.string)),"sequence_len": tf.constant(batch_size*[max_len])},
            signature="tokens",as_dict=True)["elmo"]

from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda

input_text = Input(shape=(max_len,), dtype=tf.string)
embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)

x = Bidirectional(LSTM(units=512, return_sequences=True,recurrent_dropout=0.2, dropout=0.2))(embedding)

x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,recurrent_dropout=0.2, dropout=0.2))(x)

x = add([x, x_rnn])  # residual connection to the first biLSTM

out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

model = Model(input_text, out)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

X_tr, X_val = X_tr[:10000*batch_size], X_tr[-2500*batch_size:]
y_tr, y_val = y_tr[:10000*batch_size], y_tr[-2500*batch_size:]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
history = model.fit(np.array(X_tr), y_tr,validation_data=(np.array(X_val), y_val),batch_size=batch_size, epochs=3, verbose=1)

# X_tr, X_val = X_tr[:10000*batch_size], X_tr[-2500*batch_size:]
# y_tr, y_val = y_tr[:10000*batch_size], y_tr[-2500*batch_size:]
# y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
# y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
# history = model.fit(np.array(X_tr), y_tr,batch_size=batch_size, epochs=1, verbose=1)

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report,accuracy_score
X_te = X_te[:500*batch_size]
# print(X_te)
test_pred = model.predict(np.array(X_te),verbose=1)

idx2tag = {i: w for w, i in tags2index.items()}
def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PADword", "O"))
        out.append(out_i)
    return out
def test2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            out_i.append(idx2tag[p].replace("PADword", "O"))
        out.append(out_i)
    return out
    
pred_labels = pred2label(test_pred)
test_labels = test2label(y_te) [:500*batch_size]

print(classification_report(test_labels, pred_labels))
print('f1_score==',f1_score(test_labels, pred_labels))
print('precision_score==',precision_score(test_labels, pred_labels))
print('recall_score==',recall_score(test_labels, pred_labels))
print('accuracy_score==',accuracy_score(test_labels, pred_labels))

i = 4
p = model.predict(np.array(X_te[i:i+batch_size]))[0]
p = np.argmax(p, axis=-1)
print("{:10} {:10}: ({})".format("Word", "Pred", "True"))
print("="*60)

for w, true, pred in zip(X_te[i], y_te[i], p):
    if w != "__PAD__":
        print("{:10}:{:10} ({})".format(w, label[pred], label[true]))

d = pd.read_csv(r"test_set_ran.csv") #, encoding="latin1")
# data=data.drop(['Unnamed: 0.1'],axis=1)
# data = data.drop(['POS'], axis =1)
# data = data.fillna(method="ffill")
d

words = set(list(d['word'].values))
n_words = len(words)
n_words

label = list(set(d["label"].values))
n_tags = len(label)
n_tags

d['label'].value_counts()

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),s["label"].values.tolist())]
        self.grouped = self.data.groupby("Unnamed: 0").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Unnamed: 0: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(d)
sent = getter.get_next()
print(sent)
# [('Thousands', 'O'), ('of', 'O'), ('demonstrators', 'O'), ('have', 'O'), ('marched', 'O'), ('through', 'O'), ('London', 'B-geo'), ('to', 'O'), ('protest', 'O'), ('the', 'O'), ('war', 'O'), ('in', 'O'), ('Iraq', 'B-geo'), ('and', 'O'), ('demand', 'O'), ('the', 'O'), ('withdrawal', 'O'), ('of', 'O'), ('British', 'B-gpe'), ('troops', 'O'), ('from', 'O'), ('that', 'O'), ('country', 'O'), ('.', 'O')]

sentences = getter.sentences
print(len(sentences))
# # 47959

largest_sen = max(len(sen) for sen in sentences)
print('biggest sentence has {} words'.format(largest_sen))

# %matplotlib inline
plt.hist([len(sen) for sen in sentences] , bins=50)
plt.show

max_len = 2
X = [[w[0]for w in s] for s in sentences]
new_XX = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("PADword")
    new_XX.append(new_seq)

print(new_XX[15])
len(new_XX)

from keras_preprocessing.sequence import pad_sequences
tags2index = {t:i for i,t in enumerate(label)}
yy= [[tags2index[w[1]] for w in s] for s in sentences]
yy= pad_sequences(maxlen=max_len, sequences=yy, padding="post")

print(yy[15])
len(yy)
#, value=tags2index['0']

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report,accuracy_score
test = new_XX[:268*batch_size]
# print(X_te)
test_pred = model.predict(np.array(test),verbose=1)

idx2tag = {i: w for w, i in tags2index.items()}
def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PADword", "O"))
        out.append(out_i)
    return out
def test2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            out_i.append(idx2tag[p].replace("PADword", "O"))
        out.append(out_i)
    return out
    
pred_labels = pred2label(test_pred)
test_labels = test2label(yy) [:268*32]

print(classification_report(test_labels, pred_labels))
print('f1_score==',f1_score(test_labels, pred_labels))
print('precision_score==',precision_score(test_labels, pred_labels))
print('recall_score==',recall_score(test_labels, pred_labels))
print('accuracy_score==',accuracy_score(test_labels, pred_labels))

i = 25
p = model.predict(np.array(test[i:i+batch_size]))[0]
p = np.argmax(p, axis=-1)
print("{:10} {:10}: ({})".format("Word", "Pred", "True"))
print("="*60)

for w, true, pred in zip(test[i], y_te[i], p):
    if w != "__PAD__":
        print("{:10}:{:10} ({})".format(w, label[pred], label[true]))
