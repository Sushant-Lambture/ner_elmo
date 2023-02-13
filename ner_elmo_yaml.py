# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Training Dataset
data = pd.read_csv(r'/home/sushant/env/files/final.csv')
data.rename({'text':'word','labels':'label'},axis=1,inplace=True)
print(data)

# Test Dataset
d = pd.read_csv(r"/home/sushant/env/files/test_set_ran.csv")
print(d)

words = set(list(data['word'].values))
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

sentences = getter.sentences
print(len(sentences))

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

# Data Split
from sklearn.model_selection import train_test_split
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from keras import backend as K

X_tr, X_te, y_tr, y_te = train_test_split(new_X, y, test_size=0.01, random_state=2018)

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

# Model Training
X_tr = X_tr[:11500*batch_size]
y_tr= y_tr[:11500*batch_size]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
history = model.fit(np.array(X_tr), y_tr,batch_size=batch_size, epochs=2, verbose=1)

## Model Saving 
 
# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
    
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
  
# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = tf.keras.models.model_from_yaml(loaded_model_yaml)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Testing on the data which is present in the train dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report,accuracy_score
X_te = X_te[:110*batch_size]
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
test_labels = test2label(y_te) [:110*batch_size]

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


## TESTING ON TEST DATASET

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

sentences = getter.sentences
print(len(sentences))

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

# print(yy[15])
# len(yy)

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