import math
from nndct_shared.base import NNDCT_OP

def calculate_op_scale(rec, node):
  scale = 1.0
  if node.op.type in [NNDCT_OP.MEAN]:
    max_factor =  math.ceil(math.log(rec * 128,2))
    diff = 1.0
    multi_factor = 0.0
    shift_factor = 0.0
    for shift_factor_ in range(max_factor):
      factor = round((2 ** shift_factor_)/rec)
      diff_ = abs(factor / (2 ** shift_factor_) - 1/rec)
      if diff_ < diff:
        multi_factor = factor
        diff = diff_
        shift_factor = shift_factor_
    scale = rec * multi_factor / (2 ** shift_factor)
  return scale

def get_avgpool_dpu_coeff(kernel):
  scale = 1.0
  if kernel == [3, 3]:
    scale = 9.0 * 7.0 / 64.0
  elif kernel == [5, 5]:
    scale = 25.0 * 10.0 / 256.0
  elif kernel in [[6, 6], [3, 6], [6, 3]]:
    scale = 36.0 * 7.0 / 256.0
  elif kernel == [7, 7]:
    scale = 49.0 * 21.0 / 1024.0
  elif kernel == [14, 14]:
    scale = 196.0 * 21.0 / 4096.0
  else:
    rec = kernel[0] * kernel[1]
    max_factor =  math.ceil(math.log(rec * 128,2))
    diff = 1.0
    multi_factor = 0.0
    shift_factor = 0.0
    for shift_factor_ in range(max_factor):
      factor = round((2 ** shift_factor_)/rec)
      diff_ = abs(factor / (2 ** shift_factor_) - 1/rec)
      if diff_ < diff:
        multi_factor = factor
        diff = diff_
        shift_factor = shift_factor_
    scale = rec * multi_factor / (2 ** shift_factor)

  return scale
