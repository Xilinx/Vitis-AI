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