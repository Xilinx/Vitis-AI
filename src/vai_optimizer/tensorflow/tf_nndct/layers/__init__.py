"""Nndct layers API."""

from tf_nndct.utils import tf_utils
if tf_utils.is_tf_version_less_than('2.9.0'):
  from tf_nndct.layers.recurrent import LSTM
  from tf_nndct.layers.recurrent import LSTMCell
else:
  from keras.layers import LSTM
  from keras.layers import LSTMCell

from tf_nndct.layers.array import Identity

from tf_nndct.layers.math import Dense
from tf_nndct.layers.math import Add
from tf_nndct.layers.math import Multiply
from tf_nndct.layers.math import Sigmoid
from tf_nndct.layers.math import Tanh
