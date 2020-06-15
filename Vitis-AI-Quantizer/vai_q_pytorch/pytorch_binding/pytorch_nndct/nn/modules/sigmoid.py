import torch
import numpy as np
from nndct_shared.quantization import maybe_get_quantizer, process_inputs_and_params, post_quant_process
from nndct_shared.utils import NndctOption
from .sigmoid_table import *
from .fix_ops import NndctSigmoidTableLookup
import pytorch_nndct.utils as py_utils
__all__ = ['Sigmoid']

SIGMOID_TABLE = deephi_sigmoid_table()

class deephi_Sigmoid(torch.nn.modules.Sigmoid):
  r"""DeePhi Sigmoid operation"""

  def __init__(self):
    super(deephi_Sigmoid, self).__init__()
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.valid_inputs = None
    self.valid_output = None
    self.node = None

  def forward(self, input):

    [input], _ = process_inputs_and_params(
        self.node,
        self.quant_mode,
        self.quantizer,
        inputs=[input],
        valid_inputs=self.valid_inputs)

    if NndctOption.nndct_quant_off.value:
      output = super().forward(input)
    elif self.quant_mode > 0:
      output = torch.empty_like(input)
      input_name = self.node.in_nodes[0]
      fragpos = self.quantizer.get_bnfp(input_name, False)[1]
      NndctSigmoidTableLookup(input.cuda(),
                              SIGMOID_TABLE.table.cuda(),
                              output.cuda(),
                              fragpos)
    else:
      output = super().forward(input)

    [output] = post_quant_process(self.node, 
                                  self.valid_output, 
                                  [output],
                                  [output, output])
    return output


@py_utils.register_quant_op
def Sigmoid(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Sigmoid(*args, **kwargs)
  return deephi_Sigmoid(*args, **kwargs)
