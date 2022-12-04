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

import pandas as pd
import numpy as np
import time
import os

current_dir = os.path.dirname(__file__)
filename = os.path.join(current_dir, "data", "car_rental_training_data.csv")

data = pd.read_csv(filename, sep=';')
data.head()

complain_data = data[['Customer_Service', 'Satisfaction']]

from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import tensorflow as tf

import datetime

max_features = 500
hidden_size = 100

for idx, row in complain_data.iterrows():
  row[0] = row[0].replace('rt',' ')

tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(complain_data['Customer_Service'].values)
X = tokenizer.texts_to_sequences(complain_data['Customer_Service'].values)

maxlen = 50
X = pad_sequences(X, maxlen=maxlen)

Y = complain_data['Satisfaction'].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

embedding_vector_length = 32

model = Sequential()
model.add(layers.Embedding(max_features, embedding_vector_length, input_length=maxlen))
model.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.LSTM(100, recurrent_activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

filename = os.path.join(current_dir, 'data', 'complain_model.h5')

is_training = False
if is_training:
  model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=64)
  # Evaluate the model
  scores = model.evaluate(X_test, Y_test, verbose=0)
  print("Evaluation Accuracy: %.2f%%" % (scores[1]*100))
  model.save(filename, save_format='tf')
else:
  model.load_weights(filename)

from tensorflow.python.keras import backend as K
import datetime as dt
#t1=time.time()
# layers: [Embedding, Conv1D, MaxPooling1D, LSTM, Dense]
#print("x_test size:", X_test.shape)
lstm_upstream = K.function([model.layers[0].input], [model.layers[2].output])
lstm_input = lstm_upstream([X_test])[0]

batches = lstm_input.shape[0]
lstm_output = np.zeros((batches, 100))
lstm_tmp = np.zeros((1, 100))
lstm = K.function([model.layers[3].input], [model.layers[3].output])
t1=time.time()
for index, x in enumerate(lstm_input):
    lstm_input_batch1 = x.reshape(1,25,32)
    lstm_output[index] = lstm(lstm_input_batch1)[0]

#lstm = K.function([model.layers[3].input], [model.layers[3].output])
#lstm_start = dt.datetime.now()
#lstm_output = lstm([lstm_input])[0]
#lstm_finish = dt.datetime.now()
#print('lstm foward time(secs):', (lstm_finish - lstm_start).total_seconds())
#lstm_out = lstm_output_batch1.reshape((batches, 25, 100))
lstm_downstream = Sequential()
lstm_downstream.add(layers.Dense(1, activation='sigmoid'))
lstm_downstream.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_downstream.build((1, 100))
lstm_downstream.layers[0].set_weights(model.get_layer('dense').get_weights())
score = lstm_downstream.evaluate(lstm_output, Y_test, verbose=0)
t2=time.time()
print('Accuracy:', score[1])
print('E2E Time:', t2-t1)
#print(tf.__version__)
