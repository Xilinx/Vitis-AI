from tensorflow.python.ops.signal import fft_ops

def rfft(input_tensor, fft_length=None, name=None):
  return fft_ops.rfft(input_tensor, fft_length, name)

def irfft(input_tensor, fft_length=None, name=None):
  return fft_ops.irfft(input_tensor, fft_length, name)

def ifft(input, name=None):
  return fft_ops.ifft(input, name)
