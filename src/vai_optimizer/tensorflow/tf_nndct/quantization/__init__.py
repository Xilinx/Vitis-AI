from tf_nndct.utils import tf_utils

# No longer support LSTM after TF 2.9.0
if tf_utils.is_tf_version_less_than('2.9.0'):
  from .tf_qstrategy import *
  from .tf_qconfig import *
  from .quantizer import *
  from .api import *
