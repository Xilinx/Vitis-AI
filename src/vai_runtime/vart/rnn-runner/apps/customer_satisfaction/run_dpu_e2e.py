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
import json
import datetime
import time
import numpy as np
import pandas as pd
import os
from os import environ, getenv

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import layers
from tensorflow.python.keras import Sequential
from tensorflow.python import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
import vart
import xir


current_dir = os.path.dirname(__file__)
filename = os.path.join(current_dir, "data", "car_rental_training_data.csv")

data = pd.read_csv(filename, sep=';')
data.head()

complain_data = data[['Customer_Service', 'Satisfaction']]

# Generate rnn graph
runners = []
batches = []

device_name = getenv("TARGET_DEVICE", default="").upper()
assert(device_name in ("U50LV", "U25")), "TARGET_DEVICE should be U50LV/U25"

num_cores = 1
if device_name == "U50LV":
    num_cores = 2
    models = [os.path.join(current_dir, "data", file)
              for file in ["compiled_batch_3.xmodel", "compiled_batch_4.xmodel"]]
elif device_name == "U25":
    num_cores = 1
    models = [os.path.join(current_dir, "data", file)
              for file in ["compiled_batch_1.xmodel"]]

num_sequences = 25
output_seq_dim = 100

# Setup the Runners
for i in range(num_cores):
    print("[INFO] Creating Runner on core:{}".format(i))
    graph = xir.Graph.deserialize(models[i])
    runners.append(vart.Runner.create_runner(
        graph.get_root_subgraph(), "run"))
    inputTensors = runners[i].get_input_tensors()
    outputTensors = runners[i].get_output_tensors()
    batch_size, num_frames, runner_in_seq_len = tuple(inputTensors[0].dims)
    _, _, runner_out_seq_len = tuple(outputTensors[0].dims)
    batches.append(batch_size)

out_pos = graph.get_root_subgraph().get_attr('output_fix2float')
in_pos = graph.get_root_subgraph().get_attr('input_float2fix')

max_features = 500

def quanti_convert_float_to_int16(data, fix_pos):
    amp = 2**fix_pos
    max = 2**(16-1)
    output = data * amp
    output = np.clip(output, -max, max - 1)
    output = np.floor(output)
    output = output.astype(np.int16)
    return output


def quanti_convert_int16_to_float(data, fix_pos):
    amp = 2**fix_pos
    output = data.astype(np.float32)
    output = data / amp
    return output


for idx, row in complain_data.iterrows():
    row[0] = row[0].replace('rt', ' ')

tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(complain_data['Customer_Service'].values)
X = tokenizer.texts_to_sequences(complain_data['Customer_Service'].values)

maxlen = 50
X = pad_sequences(X, maxlen=maxlen)

Y = complain_data['Satisfaction'].values
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)

# print(X_train.shape,Y_train.shape)
# print(X_test.shape,Y_test.shape)

embedding_vector_length = 32

model = Sequential()
model.add(layers.Embedding(max_features,
          embedding_vector_length, input_length=maxlen))
model.add(layers.Conv1D(filters=32, kernel_size=3,
          padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.LSTM(100, recurrent_activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

filename = os.path.join(current_dir, 'data', 'complain_model.h5')
is_training = False
if is_training:
    model.fit(X_train, Y_train, validation_data=(
        X_test, Y_test), epochs=20, batch_size=64)

    # Evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Evaluation Accuracy: %.2f%%" % (scores[1]*100))
    model.save(filename, save_format='tf')
else:
    model.load_weights(filename)

t1 = time.time()

lstm_upstream = tf.keras.Model(
    inputs=model.input, outputs=model.get_layer('max_pooling1d').output)
lstm_input = lstm_upstream.predict(X_test, batch_size=8)
# print(lstm_input.shape)

num_records = lstm_input.shape[0]
quantized_lstm_input = quanti_convert_float_to_int16(
    lstm_input.reshape(num_records * 25*32), in_pos).reshape((num_records, 25*32))
lstm_output = np.zeros((num_records, 25*100), dtype=np.int16)


runner_idx = count = 0

outputTensors = runners[0].get_output_tensors()
_, _, runner_out_seq_len = tuple(outputTensors[0].dims)

quantized_input = quantized_lstm_input.view()
out_np = lstm_output.view()

print("[INFO] Running execute_async..")

while count < len(quantized_input):
    batch_size = batches[runner_idx]
    input_data = quantized_input[count:count+batch_size]
    batch_size = input_data.shape[0]

    input_data = input_data.reshape(batch_size, num_sequences, runner_in_seq_len)
    output_data = np.empty((batch_size, num_sequences, runner_out_seq_len), dtype=np.int16)
    job_id = runners[runner_idx].execute_async([input_data], [output_data], True)
    runners[runner_idx].wait(job_id)
    out_np[count:count+batch_size, ...] = output_data[..., :output_seq_dim].reshape(
            batch_size, num_sequences*output_seq_dim)

    count += batch_size
    runner_idx = (runner_idx + 1) % num_cores

while runners:
    del runners[0]

lstm_output = quanti_convert_int16_to_float(lstm_output, out_pos)
lstm_output = lstm_output.reshape((num_records, 25, 100))[:, -1, :]

lstm_downstream = Sequential()
lstm_downstream.add(layers.Dense(1, activation='sigmoid'))
lstm_downstream.compile(loss='binary_crossentropy',
                        optimizer='adam', metrics=['accuracy'])
lstm_downstream.build((1, 100))
lstm_downstream.layers[0].set_weights(model.get_layer('dense').get_weights())
score = lstm_downstream.evaluate(lstm_output, Y_test, verbose=0)
t2 = time.time()
print('Accuracy:', score[1])
print('E2E Time:', t2-t1)
