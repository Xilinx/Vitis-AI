import numpy as np
import os
import argparse
from os import environ

parser = argparse.ArgumentParser(description='Lstm tensorflow quantizer test')
parser.add_argument('--quant_mode',
                    type=str, 
                    default='calib', 
                    help='Lstm tensorflow quantization mode, calib for calibration of quantization, test for evaluation of quantized model')
parser.add_argument('--subset_len',
                    type=int,
                    default=None,
                    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
args = parser.parse_args()

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.eager import context

from tf_nndct.graph import builder
from tf_nndct.quantization.api import tf_quantizer

#tf.keras.backend.set_floatx('float64')

model_path = "./pretrained.h5"
# set the seed
np.random.seed(7)

# load the dataset top n words only
top_words = 5000
(X_train,y_train), (X_test,y_test) = imdb.load_data(num_words=top_words)

# truncate and pad input sequences
if args.subset_len:
    subset_len = args.subset_len
    assert subset_len <= X_test.shape[0]
    X_test = X_test[:subset_len]
    y_test = y_test[:subset_len]

max_review_length = 500
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model definition
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100, implementation=1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights(model_path)

if args.quant_mode == 'calib' or args.quant_mode == 'test':
    single_batch_data = X_test[:1, ]
    #input_signature = tf.TensorSpec(single_batch_data.shape[1:], tf.int32)
    input_signature = tf.TensorSpec(single_batch_data.shape, tf.int32)
    print('Start quantizer creation...', flush=True)
    quantizer = tf_quantizer(model, 
                             input_signature, 
                             quant_mode = args.quant_mode,
                             bitwidth = 16)
    rebuilt_model = quantizer.quant_model
    print('Start quantization inference...', flush=True)

    batch_size = 50
    correct_count = 0
    #calib_iters = int(len(X_test)//batch_size/20)
    #eval_iters = int(len(X_test)//batch_size)
    #iters = calib_iters if args.quant_mode == 'calib' else eval_iters
    iters = int(len(X_test)//batch_size)
    for j in range(iters + 1):
        start = j * batch_size
        end = min(start + batch_size, len(X_test))
        output = rebuilt_model(X_test[start: end])
        print('Iteration {} is done...'.format(j))
        for o in range(len(output)):
            correct = output.numpy()[o][0] > 0.5
            if correct == y_test[start+o]:
                correct_count = correct_count + 1
    print("Evaluation Accuracy: %.2f%%" % (correct_count/len(y_test)*100))

    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
    elif args.quant_mode == 'test':
        # dump xmodel data
        quantizer.dump_xmodel()
        quantizer.dump_rnn_outputs_by_timestep(X_test[:1])
else:
    # evaluate the float model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Evaluation Accuracy: %.2f%%" % (scores[1]*100))

