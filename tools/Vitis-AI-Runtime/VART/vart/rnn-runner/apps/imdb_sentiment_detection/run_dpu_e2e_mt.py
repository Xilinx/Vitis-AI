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
import os
from os import environ, getenv
import json
import numpy as np
import pandas
import tensorflow as tf
import datetime
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk import word_tokenize
from tensorflow.python.keras import Model
import time
import vart
import xir

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-n', "--num-runners", default=4, type=int)
args = parser.parse_args()

current_dir = os.path.dirname(__file__)
model_filename = os.path.join(current_dir, "data", "LSTM.h5")
data_filename = os.path.join(current_dir, "data", "imdb.npz")
word_dict_path = os.path.join(current_dir, "data", "imdb_word_index.json")
predict_file = os.path.join(current_dir, "data", "IMDB.csv")
output_predict_file = os.path.join(current_dir, "data", 'predictions.txt')
# set the seed
np.random.seed(7)

# load the dataset top n words only
top_words = 5000
max_review_length = 500
Y = []


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


def preprocessing(filename, maxlen, top_words, word_dict_path=None):
    noisy_char = '!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    table = str.maketrans(noisy_char, ' '*len(noisy_char))
    data = pandas.read_csv(filename, sep=',')
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


print("Load model")
model = load_model(model_filename, compile=False)
print("Model ready")
model.summary()
if not os.path.exists(word_dict_path):
    imdb.get_word_index(path=word_dict_path)
print("start preprocessing")
t1 = time.time()
pre_begin = datetime.datetime.now()
# X_test = preprocessing(predict_file, max_review_length,
#                        top_words, word_dict_path)
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
np.random.seed(7)

# load the dataset top n words only
top_words = 5000
# (X_train, y_train), (X_test, y_test) = load_data(path=data_path,
#                                                  num_words=top_words)

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    path=data_path, num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print("preprocessing over")
pre_end = datetime.datetime.now()
a = np.array(y_test)
a = a.reshape([a.shape[0], 1])


print("Init Dense model")
lstm_model2 = Sequential()
lstm_model2.add(Dense(1, activation='sigmoid'))
lstm_model2.build((1, 100))
layer_dict_fix = dict([(layer.name, layer) for layer in model.layers])
lstm_model2.layers[0].set_weights(layer_dict_fix['dense'].get_weights())
print("Dense ready")
print("Init Hardware")
# model_lstm = dpu4rnn_py.dpu4rnn.create("sentiment")

# Generate rnn graph
embedding_vecor_length = 32
runners = []
batches = []

num_sequences = max_review_length
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

print("Hardware ready")
hybrid_begin = datetime.datetime.now()

# print(model.get_weights())
print("Embedding start")
permute_layer_model = Model(
    inputs=model.input, outputs=model.get_layer("embedding").output)
# print(permute_layer_model.get_layer("embedding").get_weights())
ebd_begin = datetime.datetime.now()
lstm_input = permute_layer_model.predict(X_test, batch_size=8)
ebd_end = datetime.datetime.now()
num_records = lstm_input.shape[0]
print("Embedding over")

cv1b = datetime.datetime.now()
quantized_lstm_input = quanti_convert_float_to_int16(lstm_input.reshape(
    num_records * 16000), in_pos).reshape((num_records, 16000))
cv1e = datetime.datetime.now()
inputs = []
lstm_output = np.zeros((quantized_lstm_input.shape[0], 100), dtype=np.int16)
print("Start LSTMlayer")
begin = datetime.datetime.now()

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
    out_np[count:count+batch_size, ...] = output_data[:, -1, :output_seq_dim].reshape(
            batch_size, output_seq_dim)

runner_idx = count = 0
futures = []
print("[INFO] Total number of records: {}".format(num_records))
dpu_start = time.time()
with concurrent.futures.ThreadPoolExecutor() as executor:
    while count < len(quantized_input):
        batch_size = batches[runner_idx]
        input_data = quantized_input[count:count+batch_size]
        batch_size = input_data.shape[0]
        futures.append(executor.submit(
            run, input_data, batch_size, count, num_sequences,
            runner_out_seq_len, output_seq_dim, runners[runner_idx], out_np))
        count += batch_size
        runner_idx = (runner_idx + 1) % args.num_runners

concurrent.futures.wait(futures)
print("[INFO] DPU Time: {}".format(time.time() - dpu_start))

while runners:
    del runners[0]

end = datetime.datetime.now()
print("LSTM over")
cv2b = datetime.datetime.now()
lstmlayerout = quanti_convert_int16_to_float(lstm_output, out_pos)
cv2e = datetime.datetime.now()

dense_begin = datetime.datetime.now()
lstm_output_dense = lstm_model2.predict(lstmlayerout, batch_size=8)
dense_end = datetime.datetime.now()
hybrid_end = datetime.datetime.now()
t2 = time.time()
lacc = []

for ind, j in enumerate(lstm_output_dense):
    if (lstm_output_dense[ind] > 0.5):
        lacc.append(1)
    else:
        lacc.append(0)
lacc = np.array(lacc).reshape(len(lacc), 1)
countall = lacc.shape[0]
counterr = 0
for ind, i in enumerate(lacc):
    if (lacc[ind] != a[ind]):
        counterr = counterr + 1
hybrid_end = datetime.datetime.now()

print("====== IMDB Sentiment Detection Results ======")
print("Accuracy: ", (countall - counterr)/countall)
print("preprocessing time", (pre_end - pre_begin).total_seconds())
print("embedding time", (ebd_end - ebd_begin).total_seconds())
print("convert1", (cv1e - cv1b).total_seconds())
print("total hw time", (end - begin).total_seconds())
print("convert2", (cv2e - cv2b).total_seconds())
print("dense lstm time", (dense_end - dense_begin).total_seconds())
print("total hybrid time", (hybrid_end - hybrid_begin).total_seconds())
print("num", out_np.shape[0])
print("Avg hw time", (end - begin).total_seconds() / out_np.shape[0])
print()
print("total hybrid time:", t2-t1)
