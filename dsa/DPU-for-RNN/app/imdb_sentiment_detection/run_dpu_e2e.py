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
import dpu4rnn_py
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

core = 1
#from keras.utils import CustomObjectScope
#from keras.initializers import glorot_uniform

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
Y = []

with open("model/sentiment.json",'r') as load_f:
  load_dict = json.load(load_f)
in_pos = load_dict[0]['lstm_in_float2fix']
out_pos = load_dict[0]['lstm_out_fix2float']

def quanti_convert_float_to_int16(data, fix_pos):
    amp = 2**fix_pos
    max = 2**(16-1)

    output = data * amp
    #print('step 1:', output[:32])
    output = numpy.clip(output, -max, max - 1)
    #output = numpy.where(numpy.logical_and(output < 0, (output - numpy.floor(output)) == 0.5),
    #        numpy.ceil(output), numpy.round(output)) #
    output = numpy.floor(output)
    #print('step 2:', output[:32])
    output = output.astype(numpy.int16)
    #print('step 3:', output[:32])
    return output


def quanti_convert_int16_to_float(data, fix_pos):
    amp = 2**fix_pos
    output = data.astype(numpy.float32)
    output = data / amp
    return output


def preprocessing(filename, maxlen, top_words, word_dict_path=None):
    noisy_char = '!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    table = str.maketrans(noisy_char, ' '*len(noisy_char))
    data = pd.read_csv(filename, sep=',') 
    data.head()
    word_dict = json.load(open(word_dict_path, 'r'))
    X = []

    for sentiment in data['sentiment'].values:
        Y.append(sentiment)
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
#with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
#    model = load_model(model_filename)
model = load_model(model_filename, compile = False)
print ("Model ready")
print (model.summary())
if not os.path.exists(word_dict_path):
    imdb.get_word_index(path=word_dic_path)
print("start preprocessing")
t1 = time.time()
pre_begin = datetime.datetime.now()
#X_test = preprocessing(predict_file, max_review_length, top_words, word_dict_path)
########################################
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
#(X_train, y_train), (X_test, y_test) = load_data(path=data_path, num_words=top_words)

(X_train, y_train), (X_test, y_test) = imdb.load_data(path=data_path, num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print("preprocessing over")
pre_end = datetime.datetime.now()
a = numpy.array(y_test)
a = a.reshape([a.shape[0],1])


print("Init Dense model")
lstm_model2= Sequential()
lstm_model2.add(Dense(1, activation='sigmoid'))
lstm_model2.build((1, 100))
layer_dict_fix = dict([(layer.name, layer) for layer in model.layers])
lstm_model2.layers[0].set_weights(layer_dict_fix['dense'].get_weights())
print("Dense ready")
print("Init Hardware")
#model_lstm = dpu4rnn_py.dpu4rnn.create("sentiment")

thread_lstm = []
batch = []
total_batch = 0
for i in range(core):
    thread_lstm.append(dpu4rnn_py.dpu4rnn.create("sentiment", i))
    batch.append(thread_lstm[i].getBatch())
    total_batch += batch[i]

print("Hardware ready")
hybrid_begin = datetime.datetime.now()

#print(model.get_weights())
embedding_vecor_length = 32
print ("Embedding start")
permute_layer_model = Model(inputs=model.input, outputs=model.get_layer("embedding").output)
#print(permute_layer_model.get_layer("embedding").get_weights())
ebd_begin = datetime.datetime.now()
pd = permute_layer_model.predict(X_test, batch_size = 8)
#print(X_test[:10])
#print('pd:', pd.shape)
#print(pd[0, :32])
ebd_end = datetime.datetime.now()
num = pd.shape[0]
print ("Embedding over")

#X_test = []
cv1b = datetime.datetime.now()
xxx = quanti_convert_float_to_int16(pd.reshape(num * 16000), in_pos).reshape((num, 16000))
cv1e = datetime.datetime.now()

#pd = []
inputs = []
#print (layer_dict_fix['lstm'].input_shape)
out_np = numpy.ones((xxx.shape[0], 100), dtype=numpy.int16)
print ("Start LSTMlayer")
begin = datetime.datetime.now()

#numpy.savetxt("lstm_in.txt", xxx[0].astype(numpy.int16), fmt='%0d')

#for ind, x in enumerate(xxx):
#    model_lstm.run(x.flatten(), 16000 * 2, out_np[ind], 500)

frame_num = 500
frame_size = 16000 * 2
count = 0
while count < num:
    if (count + total_batch) <= num:
        for i in range(len(batch)):
            thread_lstm[i].run(xxx[count:count+batch[i]].flatten(), frame_size*batch[i], out_np[count:count+batch[i]], frame_num, batch[i])
            count = count + batch[i]
    elif num - count > batch[0]:
        thread_lstm[0].run(xxx[count:count+batch[0]].flatten(), frame_size*batch[0], out_np[count:count+batch[0]], frame_num, batch[0])
        rest_batch = num-count-batch[0]
        thread_lstm[1].run(xxx[count+batch[0]:].flatten(), frame_size*rest_batch, out_np[count+batch[0]:], frame_num, rest_batch)
        count = num
    else:
        rest_batch = num - count
        thread_lstm[0].run(xxx[count:].flatten(), frame_size*rest_batch, out_np[count:], frame_num, rest_batch)
        count = count + rest_batch


end = datetime.datetime.now()
print ("LSTM over")
cv2b = datetime.datetime.now()
lstmlayerout = quanti_convert_int16_to_float(out_np, out_pos)
#print('lstmlayerout:', lstmlayerout)
cv2e = datetime.datetime.now()

dense_begin = datetime.datetime.now()
lstm_output_dense = lstm_model2.predict(lstmlayerout, batch_size = 8)
dense_end = datetime.datetime.now()
hybrid_end = datetime.datetime.now()
t2 = time.time()
lacc = []
#print(lstm_output_dense)
for ind,j in enumerate (lstm_output_dense):
    if (lstm_output_dense[ind] >0.5):
        lacc.append(1)
    else:
        lacc.append(0)
lacc = numpy.array(lacc).reshape(len(lacc), 1)
countall = lacc.shape[0]
counterr = 0
for  ind, i in enumerate(lacc):
    if (lacc[ind] != a[ind]):
        counterr = counterr  +1
hybrid_end = datetime.datetime.now()
print ("====== IMDB Sentiment Detection Results ======")
print ("Accuracy: ", (countall -counterr)/countall)
print ("preprocessing time", (pre_end - pre_begin).total_seconds())
print ("embedding time", (ebd_end - ebd_begin).total_seconds())
print ("convert1", (cv1e - cv1b).total_seconds())
print ("total hw time", (end - begin).total_seconds())
print ("convert2", (cv2e - cv2b).total_seconds())
print ("dense lstm time", (dense_end - dense_begin).total_seconds())
print ("total hybrid time", (hybrid_end - hybrid_begin).total_seconds())
print ("num", out_np.shape[0])
print ("Avg hw time", (end - begin).total_seconds() / out_np.shape[0])
print ()
print ("total hybrid time:", t2-t1)
