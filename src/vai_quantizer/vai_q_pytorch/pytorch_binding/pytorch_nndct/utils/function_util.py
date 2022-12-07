import torch
import numpy as np

# 1/sqrt(x)
def invsqrt(number):
    x2 = number.astype(np.float32)
    x2 = x2 * 0.5
    y  = number.astype(np.float32)
    threehalfs = 1.5
    i = y.view(np.int32)
    i  = 0x5f3759df - ( i >> 1 )
    y = i.view(np.float32)
    y  = y * ( threehalfs - ( x2 * y * y ) )
    y  = y * ( threehalfs - ( x2 * y * y ) )
    y  = y * ( threehalfs - ( x2 * y * y ) )
    y  = y * ( threehalfs - ( x2 * y * y ) )
    return y

