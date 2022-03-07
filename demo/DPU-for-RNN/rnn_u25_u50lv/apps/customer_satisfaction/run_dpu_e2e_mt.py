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

import concurrent.futures
import json
import datetime
import numpy as np
import pandas as pd
from os import environ, getenv
import os

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import layers
from tensorflow.python.keras import Sequential
from tensorflow.python import keras

from time import time
import tensorflow as tf
import vart
import xir

parser = ArgumentParser()
parser.add_argument('-n', "--num-runners", default=4, type=int)
args = parser.parse_args()

current_dir = os.path.dirname(__file__)
filename = os.path.join(current_dir, "data", "car_rental_training_data.csv")

data = pd.read_csv(filename, sep=';')
data.head()

complain_data = data[['Customer_Service', 'Satisfaction']]

# Generate rnn graph
runners = []
batches = []

num_sequences = 25
input_seq_dim = 32
output_seq_dim = 100

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

# Setup the Runners
for runner_idx in range(args.num_runners):
    graph = xir.Graph.deserialize(models[runner_idx % num_cores])
    runners.append(vart.Runner.create_runner(graph.get_root_subgraph(), "run"))
    inputTensors = runners[runner_idx].get_input_tensors()
    batch_size = tuple(inputTensors[0].dims)[0]
    batches.append(batch_size)

out_pos = graph.get_root_subgraph().get_attr('output_fix2float')
in_pos = graph.get_root_subgraph().get_attr('input_float2fix')

vocab_size = 500

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

tokenizer = Tokenizer(num_words=vocab_size, split=' ')
tokenizer.fit_on_texts(complain_data['Customer_Service'].values)
X = tokenizer.texts_to_sequences(complain_data['Customer_Service'].values)

max_sentence_len = 50
X = pad_sequences(X, maxlen=max_sentence_len)

Y = complain_data['Satisfaction'].values
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)

embedding_vector_length = 32

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_vector_length,
                           input_length=max_sentence_len))
model.add(layers.Conv1D(filters=32, kernel_size=3, padding='same',
                        activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.LSTM(100, recurrent_activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()

filename = os.path.join(current_dir, 'data', 'complain_model.h5')
is_training = False
if is_training:
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20,
              batch_size=64)
    # Evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Evaluation Accuracy: %.2f%%" % (scores[1]*100))
    model.save(filename, save_format='tf')
else:
    model.load_weights(filename)

t1 = time()

lstm_upstream = tf.keras.Model(
    inputs=model.input, outputs=model.get_layer('max_pooling1d').output)
lstm_input = lstm_upstream.predict(X_test, batch_size=8)

num_records = lstm_input.shape[0]
# print("num_sequences: ", lstm_input.shape[1])
# print("input_seq_dim: ", lstm_input.shape[2])
# print("output_seq_dim: ", 100)

quantized_lstm_input = quanti_convert_float_to_int16(
    lstm_input.reshape(num_records*num_sequences*input_seq_dim), in_pos
    ).reshape((num_records, num_sequences*input_seq_dim))
lstm_output = np.zeros((num_records, num_sequences*output_seq_dim),
                       dtype=np.int16)

# use the multi-batch
inputTensors = runners[0].get_input_tensors()
outputTensors = runners[0].get_output_tensors()
_, _, runner_in_seq_len = tuple(inputTensors[0].dims)
_, _, runner_out_seq_len = tuple(outputTensors[0].dims)

quantized_input = quantized_lstm_input.view()
out_np = lstm_output.view()

def run(input_data, batch_size, count, num_sequences, runner_out_seq_len,
        output_seq_dim, runner, out_np):
    # print("[INFO] Filling {} -> {}".format(count, count+batch_size))
    input_data = input_data.reshape(batch_size, num_sequences, runner_in_seq_len)
    output_data = np.empty((batch_size, num_sequences, runner_out_seq_len), dtype=np.int16)
    job_id = runner.execute_async([input_data], [output_data], True)
    runner.wait(job_id)
    out_np[count:count+batch_size, ...] = output_data[..., :output_seq_dim].reshape(
            batch_size, num_sequences*output_seq_dim)

runner_idx = count = 0
futures = []
print("[INFO] Total number of records: {}".format(num_records))
dpu_start = time()
with concurrent.futures.ThreadPoolExecutor() as executor:
    while count < len(quantized_input):
        batch_size = batches[runner_idx]
        input_data = quantized_input[count:count+batch_size]
        batch_size = input_data.shape[0]
        futures.append(executor.submit(
            run, input_data, batch_size, count, num_sequences,
            runner_out_seq_len, output_seq_dim, runners[runner_idx], out_np))
        count += batch_size
        runner_idx = (runner_idx+1)%args.num_runners

concurrent.futures.wait(futures)
print("[INFO] DPU Time: {}".format(time() - dpu_start))

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
t2 = time()

print('[INFO] Accuracy: {}'.format(score[1]))
print('[INFO] E2E Time: {}'.format(t2-t1))
