# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a LSTM on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.
"""

from __future__ import print_function
# import g3
import numpy as np

import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper


def print_model_sparsity(pruned_model):
  """Prints sparsity for the pruned layers in the model.

  Model Sparsity Summary
  --
  prune_lstm_1: (kernel, 0.5), (recurrent_kernel, 0.6)
  prune_dense_1: (kernel, 0.5)

  Args:
    pruned_model: keras model to summarize.

  Returns:
    None
  """
  def _get_sparsity(weights):
    return 1.0 - np.count_nonzero(weights) / float(weights.size)

  print("Model Sparsity Summary ({})".format(pruned_model.name))
  print("--")
  for layer in pruned_model.layers:
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      prunable_weights = layer.layer.get_prunable_weights()
      if prunable_weights:
        print("{}: {}".format(
            layer.name, ", ".join([
                "({}, {})".format(weight.name,
                                  str(_get_sparsity(K.get_value(weight))))
                for weight in prunable_weights
            ])))
  print("\n")


max_features = 20000
maxlen = 100  # cut texts after this number of words
batch_size = 32

print("Loading data...")
(x_train,
 y_train), (x_test,
            y_test) = keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train), "train sequences")
print(len(x_test), "test sequences")

print("Pad sequences (samples x time)")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

print("Build model...")
model = keras.models.Sequential()
model.add(keras.layers.Embedding(max_features, 128, input_length=maxlen))
model.add(keras.layers.LSTM(128))  # try using a GRU instead, for fun
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation("sigmoid"))

model = prune.prune_low_magnitude(model, pruning_schedule.PolynomialDecay(
    initial_sparsity=0.3, final_sparsity=0.7, begin_step=1000, end_step=3000))

# try using different optimizers and different optimizer configs
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
print_model_sparsity(model)

print("Train...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=3,
          callbacks=[pruning_callbacks.UpdatePruningStep()],
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print_model_sparsity(model)
print("Test score:", score)
print("Test accuracy:", acc)
