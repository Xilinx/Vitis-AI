# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow.compat.v1.train import MomentumOptimizer
from tensorflow.compat.v1.train import RMSPropOptimizer
from tensorflow.compat.v1.train.experimental import MixedPrecisionLossScaleOptimizer
from tensorflow.compat.v1.train import FtrlOptimizer
from tensorflow.compat.v1.train import ProximalGradientDescentOptimizer
from tensorflow.compat.v1.train import ProximalAdagradOptimizer
from tensorflow.compat.v1.train import AdadeltaOptimizer
from tensorflow.compat.v1.train import AdagradDAOptimizer
from tensorflow.compat.v1.train import AdagradOptimizer
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.python.training.optimizer import Optimizer
from tf1_nndct.optimization.states import MASKS

opts = [
      MomentumOptimizer, RMSPropOptimizer, MixedPrecisionLossScaleOptimizer, 
      FtrlOptimizer, ProximalAdagradOptimizer, ProximalGradientDescentOptimizer, 
      AdadeltaOptimizer, AdagradDAOptimizer, AdagradOptimizer, AdamOptimizer, Optimizer]

apply_gradients_fn_ids = set()
MASKS_APPLIED_TO_GRADIENTS_KEY = "_mask_applied_to_gradients"


def wrap_apply_gradients(opt) -> None:
  apply_gradients_fn_ids.add(id(opt.apply_gradients))
  inner_apply_gradients = opt.apply_gradients
  def apply_gradients_wrapper(self, grads_and_vars, global_step=None, name=None):
    if not hasattr(self, MASKS_APPLIED_TO_GRADIENTS_KEY) or not getattr(self, MASKS_APPLIED_TO_GRADIENTS_KEY):
      for idx, (grad, var) in enumerate(grads_and_vars):
        mask_key = var.name.split(":")[0]
        if mask_key in MASKS:
          mask = MASKS[mask_key]
          grad = tf.math.multiply(grad, tf.convert_to_tensor(mask))
          grads_and_vars[idx] = (grad, var)
      setattr(self, MASKS_APPLIED_TO_GRADIENTS_KEY, True)
    return inner_apply_gradients(self, grads_and_vars, global_step, name)

  opt.apply_gradients = apply_gradients_wrapper

for opt in opts:
  wrap_apply_gradients(opt)
