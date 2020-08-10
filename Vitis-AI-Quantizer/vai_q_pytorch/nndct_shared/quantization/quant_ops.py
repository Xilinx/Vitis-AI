import numpy as np

#for calibration process
def max(data, name='', quantizer=None):
  return data.max()

def min(data, name='', quantizer=None):
  return data.min()

def quant_diff_s(data, bitwidth, range, round_method=2, name='',
                 quantizer=None):
  raise NotImplementedError("please implement the diffs operation")

#for quantization process
def __amplify_data(data, max, amp, method=2):
  #1 for floor, 2 for dpu round; use number, not amplified
  data = data * amp
  '''
  if method == 1:
    data = np.floor(data * amp)
    data = np.clip(data, -max, max - 1)
  elif method == 2:
    data = data * amp
    data = np.clip(data, -max, max - 1)
    data = np.where(
        np.logical_and(data < 0, (data - np.floor(data)) == 0.5), np.ceil(data),
        np.round(data))
  '''
  return data

def normal_quant_neuron(data,
                        maxamps=[[32768], [2048]],
                        strides=[-1],
                        round_method=2,
                        keep_scale=True,
                        name='',
                        quantizer=None,
                        on_gpu=True,
                        as_int=False):
  #integer need not keep scale as precondition
  if as_int:
    keep_scale = False
  if len(strides) == 1:
    data = __amplify_data(
        data, maxamps[0][0], maxamps[1][0], method=round_method)
    if keep_scale:
      data = data / maxamps[1][0]
  else:
    org_shape = data.shape
    flatten_data = data.flatten()
    pos = 0
    for idx, s in enumerate(strides):
      flatten_data[pos:pos + s] = __amplify_data(
          flatten_data[pos:pos + s],
          maxamps[0][idx],
          maxamps[1][idx],
          method=round_method)
      if keep_scale:
        flatten_data[pos:pos + s] = flatten_data[pos:pos + s] / maxamps[1][idx]
      pos += s
    data = flatten_data.reshape(org_shape)
  #return integer or origional dtype
  if as_int:
    assert all(m == maxamps[0][0]
               for m in maxamps[0]), "all max limitation should be the same"
    if maxamps[0][0] == 2**7:
      return data.astype(np.int8)
    elif maxamps[0][0] == 2**15:
      return data.astype(np.int16)
    else:
      raise TypeError("unexpected max found " + str(maxamps[0][0]))
  else:
    return data

def nonlin(data, alpha, signed):
  if signed:
    return np.clip(data, -alpha, alpha)
  else:
    return np.clip(data, 0, alpha)

def pact_quant_neuron(data,
                      bitw,
                      bita,
                      alpha_init_value=None,
                      signed=False,
                      trainable=True,
                      warmup=False,
                      name='',
                      tensor_type='act',
                      quantizer=None):
  raise NotImplementedError("please implement the pact_quant_neuron operation")

def graffitist_quant_neuron(data, bn, fp, method=2, name=''):
  raise NotImplementedError(
      "please implement the lowbit_quant_neuron operation")
