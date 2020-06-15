
def tensor_out_in_axis(tensor):
  nndct_layouts = {2: 'OI', 4: 'OHWI'}
  data_format = nndct_layouts[tensor.ndim]
  out_axis = data_format.index('O')
  in_axis = data_format.index('I')
  return out_axis, in_axis
