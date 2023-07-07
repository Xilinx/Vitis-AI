from .base import TorchQuantProcessor, dump_xmodel
from .rnn import LSTMTorchQuantProcessor, RNNQuantProcessor
from .utils import prepare_quantizable_module, replace_relu6_with_relu
#from .rnn import LSTMQuantizer
from .adaquant import AdvancedQuantProcessor
