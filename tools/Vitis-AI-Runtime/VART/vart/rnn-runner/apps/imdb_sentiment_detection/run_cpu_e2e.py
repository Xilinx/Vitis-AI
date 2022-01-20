"""
Copyright 2021 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from os import environ
import json
import string
import numpy as np
import sys
import pandas as pd
import tensorflow as tf
import datetime
from tensorflow.python import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk import word_tokenize
from tensorflow.python.keras import Model

import time

current_dir = os.path.dirname(__file__)
model_filename = os.path.join(current_dir, "data", "LSTM.h5")
data_filename = os.path.join(current_dir, "data", "imdb.npz")
word_dict_path = os.path.join(current_dir, "data", "imdb_word_index.json")
predict_file = os.path.join(current_dir, "data", "IMDB.csv")

# set the seed
np.random.seed(7)
# load the dataset top n words only
top_words = 5000
max_review_length = 500

def preprocessing(filename, maxlen, top_words, word_dict_path=None):
    noisy_char = '!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    table = str.maketrans(noisy_char, ' '*len(noisy_char))
    data = pd.read_csv(filename, sep=',')
    data.head()
    word_dict = json.load(open(word_dict_path, 'r'))
    X = []
    for review in data['review'].values:
        review = review.translate(table).replace("  ", " ")
        seq = []
        for word in word_tokenize(review):
            if word.lower() not in word_dict:
                continue
            index = word_dict[word.lower()] + 3
            if index >= top_words:
                continue
            seq.append(index)
        X.append(seq)

    X = pad_sequences(X, maxlen=maxlen)
    tf.logging.info("input shape: {}".format(X.shape))
    return X

print ("Load model")
model = load_model(model_filename, compile = False)
print ("Model ready")
model.summary()

pre_begin = datetime.datetime.now()
is_train = False
# writing the train model and getting input data
if environ.get('RESULT_DIR') is not None:
    output_model_folder = os.path.join(os.environ["RESULT_DIR"], "model")
    output_model_path = os.path.join(output_model_folder, model_filename)
else:
    output_model_folder = "model_bak"
    output_model_path = os.path.join("model", model_filename)

os.makedirs(output_model_folder, exist_ok=True)

if environ.get('DATA_DIR') is not None:
    data_folder = os.environ["DATA_DIR"]
    data_path = os.path.join(data_folder, data_filename)
else:
    data_folder = os.getcwd()
    data_path = os.path.join(data_folder, data_filename)
# set the seed
np.random.seed(7)
# load the dataset top n words only
top_words = 5000
(_, _), (X_test, y_test) = imdb.load_data(path=data_path, num_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
np.random.seed(7)
# load the dataset top n words only
pre_end = datetime.datetime.now()

a = np.array(y_test)
a = a.reshape([a.shape[0],1])
print("running on tensorflow cpu")

t1 = time.time()

embedding_vecor_length = 32
embd_model = Model(inputs=model.input, outputs=model.get_layer("embedding").output)

ebd_begin = datetime.datetime.now()
embd_output = embd_model.predict(X_test, batch_size = 1)
ebd_end = datetime.datetime.now()

original_lstm = Model(inputs=model.input, outputs=model.get_layer('lstm').output)
cpu_lstm_begin = datetime.datetime.now()
original_lstm_output = original_lstm.predict(X_test, batch_size = 1024)
cpu_lstm_end = datetime.datetime.now()

lstm_model2 = Sequential()
lstm_model2.add(Dense(1, activation='sigmoid'))
lstm_model2.build((1, 100))
layer_dict_fix = dict([(layer.name, layer) for layer in model.layers])
lstm_model2.layers[0].set_weights(layer_dict_fix['dense'].get_weights())

dense_begin = datetime.datetime.now()
lstm_output_dense = lstm_model2.predict(original_lstm_output, batch_size = 8)
dense_end = datetime.datetime.now()

t2 = time.time()
print("preprocessing time = {:.2f} s".format((pre_end - pre_begin).total_seconds()))
ebd_time = (ebd_end - ebd_begin).total_seconds()
print("embedding time = {:.2f} s".format(ebd_time))
cpu_lstm_time = (cpu_lstm_end - cpu_lstm_begin).total_seconds() - (ebd_end - ebd_begin).total_seconds()
print("cpu lstm time = {:.2f} s".format(cpu_lstm_time))
dense_time = (dense_end - dense_begin).total_seconds()
print("dense lstm time = {:.2f} s".format(dense_time))
print("total cpu time = {:.2f} s".format(ebd_time + cpu_lstm_time + dense_time))
print("total cpu2 time = {:.2f} s".format(t2-t1))

print("Accuracy = {:.2f} %".format(100 * np.sum(np.where(lstm_output_dense > 0.5, 1, 0) == a)/a.size))
