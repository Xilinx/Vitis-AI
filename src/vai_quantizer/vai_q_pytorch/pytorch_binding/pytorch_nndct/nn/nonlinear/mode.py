class ApproxModes(object):
  NO_APPROX = 'no_approx'
  EXP_POLY = 'exp_poly'
  EXP_LUT = 'exp_lut'
  IP_V70_BERT = 'nndct_ip_v70_bert_qat'

def is_no_approx(mode):
  return mode == ApproxModes.NO_APPROX

def is_exp_poly(mode):
  return mode == ApproxModes.EXP_POLY

def is_exp_lut(mode):
  return mode == ApproxModes.EXP_LUT

def is_ip_v70_bert(mode):
  return mode == ApproxModes.IP_V70_BERT

def available_modes():
  return [ApproxModes.NO_APPROX, ApproxModes.EXP_POLY, ApproxModes.EXP_LUT]
