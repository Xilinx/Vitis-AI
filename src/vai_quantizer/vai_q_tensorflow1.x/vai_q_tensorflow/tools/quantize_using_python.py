def get_pos_overflow(f_tensor, bit_width=8):
    x_min = f_tensor.min()
    x_max = f_tensor.max()
    #  Use 0.5 as a guard
    lower_bound = -2**(bit_width - 1) - 0.5
    upper_bound = 2**(bit_width - 1) - 0.5
    step = max(x_min / lower_bound, x_max / upper_bound)
    if step == 0:
        # Set pos to 127(max of uint8) for all zero
        return 127
    else:
        pos = np.floor(np.log2(1 / step))
        # print("overflow: min: {}, max: {}, pos: {}".format(x_min, x_max, pos))
    return pos

def quantize(f_tensor, pos, bit_width=8):
    step = 2**(-pos)
    lower_bound = -np.power(2, bit_width - 1) * step
    upper_bound = np.power(2, bit_width - 1) * step - step
    q_tensor = np.round(f_tensor / step) * step
    q_tensor = lower_bound * (q_tensor < lower_bound) \
                + q_tensor * (q_tensor >= lower_bound) # it should be >=, not > (xieyi)
    q_tensor = upper_bound * (q_tensor > upper_bound) \
                + q_tensor * (q_tensor <= upper_bound) # it should be <=, not < (xieyi)
    # import pdb; pdb.set_trace()
    return q_tensor

def get_pos_diffs(f_tensor, bit_width=8, pos_range=5):

    pos_overflow = get_pos_overflow(f_tensor, bit_width)
    if pos_overflow == 127:
      # Set pos to 127(max of uint8) for all zero
      return 127

    pos_diffs = pos_overflow
    diff_min = 999999999999
    for i in range(pos_range):
      tmp_pos = pos_overflow +i
      q_tensor = quantize(f_tensor, tmp_pos, bit_width=bit_width)
      diff = np.sum((q_tensor - f_tensor)**2)
      print("index: {} diff: {} curr_pos: {}".format(i, diff, tmp_pos))
      if diff < diff_min:
        diff_min = diff
        pos_diffs = tmp_pos
    # print("pos_diffs: {} pos_overflow: {}".format(pos_diffs, pos_overflow))
    return pos_diffs
