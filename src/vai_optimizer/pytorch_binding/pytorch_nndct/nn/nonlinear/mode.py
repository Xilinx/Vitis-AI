class ApproxModes(object):
  NO_APPROX = 'no_approx'
  EXP_POLY = 'exp_poly'
  EXP_LUT = 'exp_lut'
  QIO = 'quant_input_output'

def is_no_approx(mode):
  return mode == ApproxModes.NO_APPROX

def is_exp_poly(mode):
  return mode == ApproxModes.EXP_POLY

def is_exp_lut(mode):
  return mode == ApproxModes.EXP_LUT

def is_quant_input_output(mode):
  return mode == ApproxModes.QIO

def available_modes():
  return [ApproxModes.NO_APPROX, ApproxModes.EXP_POLY, ApproxModes.EXP_LUT]
