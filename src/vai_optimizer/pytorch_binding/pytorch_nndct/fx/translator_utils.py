

def get_meta_info(meta, info_type):
  return getattr(meta, info_type)


def convert_dtype(dtype):
  r"""convert torch dtype to nndct dtype"""
  return {
      'torch.float': 'float32',
      'torch.float32': 'float32',
      'torch.double': 'float64',
      'torch.int': 'int32',
      'torch.long': 'int64'
  }.get(dtype.__str__(), dtype)


def convert_shape(shape):
  return list(shape)