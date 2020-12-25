"""
Copyright 2019 Xilinx Inc.

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
import ctypes
from ctypes import *
from os import environ
import json
import string
import numpy
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

#from imdb_reader import load_data
#import nltk
#nltk.download('punkt')
model_filename = "model/LSTM.h5"
data_filename = "model/imdb.npz"
word_dict_path = "model/imdb_word_index.json"
predict_file = "model/IMDB.csv"
output_predict_file = 'model/predictions.txt'

# set the seed
numpy.random.seed(7)
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
print(model.summary())
if not os.path.exists(word_dict_path):
    imdb.get_word_index(path=word_dic_path)
print("start preprocessing")

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
numpy.random.seed(7)
# load the dataset top n words only
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(path=data_path, num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
numpy.random.seed(7)
# load the dataset top n words only
print("preprocessing over")
pre_end = datetime.datetime.now()


a = numpy.array(y_test)
a = a.reshape([a.shape[0],1])
print("running on tensorflow cpu")
#cpu_begin = datetime.datetime.now()
#cpu_result = model.predict(X_test, batch_size = 1)
#cpu_end = datetime.datetime.now()

t1 = time.time()

embedding_vecor_length = 32
embd_model = Model(inputs=model.input, outputs=model.get_layer("embedding").output)

ebd_begin = datetime.datetime.now()
embd_output = embd_model.predict(X_test, batch_size = 1)
ebd_end = datetime.datetime.now()
print('Embedding forward done:', ebd_end - ebd_begin)

original_lstm = Model(inputs=model.input, outputs=model.get_layer('lstm').output)
cpu_lstm_begin = datetime.datetime.now()
original_lstm_output = original_lstm.predict(X_test, batch_size = 1)
cpu_lstm_end = datetime.datetime.now()

lstm_model2= Sequential()
lstm_model2.add(Dense(1, activation='sigmoid'))
lstm_model2.build((1, 100))
layer_dict_fix = dict([(layer.name, layer) for layer in model.layers])
lstm_model2.layers[0].set_weights(layer_dict_fix['dense'].get_weights())
dense_in = numpy.zeros((25000, 100), dtype=numpy.float32)
dense_begin = datetime.datetime.now()
lstm_output_dense = lstm_model2.predict(dense_in, batch_size = 1)
dense_end = datetime.datetime.now()
t2 = time.time()
print ("preprocessing time", (pre_end - pre_begin).total_seconds())
ebd_time = (ebd_end - ebd_begin).total_seconds()
print ("embedding time", ebd_time)
cpu_lstm_time = (cpu_lstm_end - cpu_lstm_begin).total_seconds() - (ebd_end - ebd_begin).total_seconds()
print ("cpu lstm time", cpu_lstm_time)
dense_time = (dense_end - dense_begin).total_seconds()
print ("dense lstm time", dense_time)
print ("total cpu time", ebd_time + cpu_lstm_time + dense_time)
print ("total cpu2 time", t2-t1)
#print (tf.__version__)
