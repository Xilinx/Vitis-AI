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

import pandas as pd
import numpy as np
import time

core = 1

filename = "model/car_rental_training_data.csv"

data = pd.read_csv(filename, sep=';')
data.head()

complain_data = data[['Customer_Service', 'Satisfaction']]

#print(complain_data.count())

from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import tensorflow as tf

import json
import datetime
import dpu4rnn_py

thread_lstm = []
batch = []
total_batch = 0
for i in range(core):
    thread_lstm.append(dpu4rnn_py.dpu4rnn.create("satisfaction", i))
    batch.append(thread_lstm[i].getBatch())
    total_batch += batch[i]

with open("model/satisfaction.json",'r') as load_f:
  load_dict = json.load(load_f)
in_pos = load_dict[0]['lstm_in_float2fix']
out_pos = load_dict[0]['lstm_out_fix2float']

max_features = 500
hidden_size = 100

def quanti_convert_float_to_int16(data, fix_pos):
    amp = 2**fix_pos
    max = 2**(16-1)

    output = data * amp
    output = np.clip(output, -max, max - 1)
    #output = numpy.where(numpy.logical_and(output < 0, (output - numpy.floor(output)) == 0.5),
    #        numpy.ceil(output), numpy.round(output)) #
    output = np.floor(output)
    output = output.astype(np.int16)
    return output


def quanti_convert_int16_to_float(data, fix_pos):
    amp = 2**fix_pos
    output = data.astype(np.float32)
    output = data / amp
    return output


for idx, row in complain_data.iterrows():
  row[0] = row[0].replace('rt',' ')

tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(complain_data['Customer_Service'].values)
X = tokenizer.texts_to_sequences(complain_data['Customer_Service'].values)

maxlen = 50
X = pad_sequences(X, maxlen=maxlen)

Y = complain_data['Satisfaction'].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

#print(X_train.shape,Y_train.shape)
#print(X_test.shape,Y_test.shape)

embedding_vector_length = 32

model = Sequential()
model.add(layers.Embedding(max_features, embedding_vector_length, input_length=maxlen))
model.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.LSTM(100, recurrent_activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

filename = './model/complain_model.h5'
is_training = False
if is_training:
  model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=64)

  # Evaluate the model
  scores = model.evaluate(X_test, Y_test, verbose=0)
  print("Evaluation Accuracy: %.2f%%" % (scores[1]*100))
  model.save(filename, save_format='tf')
else:
  model.load_weights(filename)

t1=time.time()

lstm_upstream = tf.keras.Model(inputs=model.input, outputs=model.get_layer('max_pooling1d').output)  
lstm_input = lstm_upstream.predict(X_test, batch_size=8)
#print(lstm_input.shape)

batches = lstm_input.shape[0]
quantized_lstm_input = quanti_convert_float_to_int16(lstm_input.reshape(batches * 25*32), in_pos).reshape((batches, 25*32))
lstm_output = np.zeros((batches, 25*100), dtype = np.int16)
#np.savetxt("satis_in.txt", quantized_lstm_input.astype(np.int16), fmt='%0d')

# use the multi-batch
count = 0
frame_num = 25
frame_size = frame_num * 32 * 2
while count < batches:
    if (count + total_batch) <= batches:
        for i in range(len(batch)):
            thread_lstm[i].run(quantized_lstm_input[count:count+batch[i]].flatten(), frame_size*batch[i], lstm_output[count:count+batch[i]], frame_num, batch[i])
            count = count + batch[i]
    elif batches - count > batch[0]:
        thread_lstm[0].run(quantized_lstm_input[count:count+batch[0]].flatten(), frame_size*batch[0], lstm_output[count:count+batch[0]], frame_num, batch[0])
        rest_batch = batches-count-batch[0]
        thread_lstm[1].run(quantized_lstm_input[count+batch[0]:].flatten(), frame_size*rest_batch, lstm_output[count+batch[0]:], frame_num, rest_batch)
        count = batches
    else:
        rest_batch = batches - count
        thread_lstm[0].run(quantized_lstm_input[count:].flatten(), frame_size*rest_batch, lstm_output[count:], frame_num, rest_batch)
        count = count + rest_batch

#for index, x in enumerate(quantized_lstm_input):
#  cxxlib.run(x.flatten(), 2*25*32, lstm_output[index], 25)

#np.savetxt("satis_out.txt", lstm_output.astype(np.int16), fmt='%0d')
lstm_output = quanti_convert_int16_to_float(lstm_output, out_pos)
# Use last frame.
lstm_output = lstm_output.reshape((batches, 25, 100))[:, -1, :]
#print('lstm_input:', lstm_input)
#print('lstm_output:', lstm_output.shape, lstm_output)

lstm_downstream = Sequential()
lstm_downstream.add(layers.Dense(1, activation='sigmoid'))
lstm_downstream.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_downstream.build((1, 100))
lstm_downstream.layers[0].set_weights(model.get_layer('dense').get_weights())
score = lstm_downstream.evaluate(lstm_output, Y_test, verbose=0)
t2=time.time()
print('Accuracy:', score[1])
print('E2E Time:', t2-t1)
