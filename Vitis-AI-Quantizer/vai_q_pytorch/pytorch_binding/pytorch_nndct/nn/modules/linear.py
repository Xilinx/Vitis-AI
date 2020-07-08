import torch
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
from .fix_ops import NndctScale

__all__ = ['Linear']

class deephi_Linear(torch.nn.modules.linear.Linear):
  r"""DeePhi Linear operation, support float and double"""

  def __init__(self, *args, **kwards):
    super(deephi_Linear, self).__init__(*args, **kwards)
    self.valid_inputs = None
    self.valid_output = None
    self.params_name = None
    self.bias_valid_inputs = None
    self.bias_valid_output = None
    self.node = None
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.param_quantized = False
    self.need_quant_output = True

  def forward(self, input):
    if self.bias is not None:
      params = [self.weight, self.bias]
    else:
      params = [self.weight]

    [input], __ = process_inputs_and_params(
        self.node,
        self.quant_mode,
        self.quantizer,
        inputs=[input],
        valid_inputs=self.valid_inputs,
        params=[],
        param_names=[])
    if (not self.param_quantized):
      __, __ = process_inputs_and_params(
          self.node,
          self.quant_mode,
          self.quantizer,
          inputs=[],
          valid_inputs=[],
          params=params,
          param_names=self.params_name)
      self.param_quantized = True

    scalei = 1
    scalew = 1
    
    # enlarge input if it is too small
    fragposi = self.quantizer.get_bnfp(self.node.in_nodes[0], False)[1]
    if fragposi is not None and fragposi > 12:
      scalei = (1 << fragposi)
      NndctScale(input, scalei)

    # enlarge weights if it is too small
    fragposw = self.quantizer.get_bnfp(self.params_name[0], False)[1]
    if fragposw is not None and fragposw > 12:
      scalew = (1 << fragposw)
      NndctScale(self.weight, scalew)

    # enlarge bias along with input and weights
    if self.bias is not None:
      if scalei > 1 or scalew > 1:
        NndctScale(self.bias, (scalei * scalew))

    output = super().forward(input)

    # shrink back output and bias
    if scalei > 1 or scalew > 1:
      NndctScale(output, 1.0/(scalei * scalew))
      if self.bias is not None:
        NndctScale(self.bias, 1.0/(scalei * scalew))
    # shrink back weights
    if scalew > 1:
      NndctScale(self.weight, 1.0/scalew)
    # shrink back input
    if scalei > 1:
      NndctScale(input, 1.0/scalei)

    if (self.need_quant_output):
      [output] = post_quant_process(self.node, self.valid_output, [output],
                                    [output, output])

    return output
  
@py_utils.register_quant_op
def Linear(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode == None:
    return torch.nn.Linear(*args, **kwargs)
  return deephi_Linear(*args, **kwargs)

