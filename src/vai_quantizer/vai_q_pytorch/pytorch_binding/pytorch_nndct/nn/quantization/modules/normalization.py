import torch

from torch import nn

from pytorch_nndct.nn.nonlinear import approx
from pytorch_nndct.nn.nonlinear import mode
from nndct_shared.utils import NndctOption

class LayerNorm(nn.LayerNorm):

  _FLOAT_MODULE = nn.LayerNorm

  def __init__(self,
               normalized_shape,
               eps=1e-5,
               elementwise_affine=True,
               approx_mode=mode.ApproxModes.NO_APPROX,
               rt_spec = None,
               **kwargs):
    super(LayerNorm, self).__init__(normalized_shape, eps, elementwise_affine, **kwargs)

    self.approx_mode = approx_mode
    self.norm_dims = list(range(-len(self.normalized_shape), 0))
    assert rt_spec, 'Runtime spec must be provided for quantized module'
    self.rt_spec = rt_spec

    if elementwise_affine:
      self.weight_quantizer = rt_spec.get_weight_quantizer('weight')
      self.bias_quantizer = rt_spec.get_weight_quantizer('bias')

  @property
  def is_quantized(self):
    return True

  def forward(self, input):
    if mode.is_ip_v70_bert(self.approx_mode):
      quantized_weight = self.weight_quantizer(self.weight)
      quantized_bias = self.bias_quantizer(
          self.bias) if self.bias is not None else None
      return torch.nn.functional.layer_norm(input, 
                                            self.normalized_shape,
                                            quantized_weight,
                                            quantized_bias,
                                            self.eps)
    else:
      input = input.to(torch.bfloat16)
      mean = torch.mean(input, self.norm_dims, keepdim=True)
      var = torch.var(
          input, self.norm_dims, unbiased=False, keepdim=True) + self.eps

      var = var.to(torch.bfloat16)
      if mode.is_no_approx(self.approx_mode):
        inverse_std = 1 / torch.sqrt(var)
      else:
        inverse_std = approx.isqrt_approx_walcyzk(var.to(torch.float32))
        inverse_std = inverse_std.to(torch.bfloat16)
      out = input - mean
      out = out * inverse_std
      # NOTE: Mul_add(a_bf16, b_bf16, c_fp32)
      out = out.bfloat16().float() * (self.weight.bfloat16().float())
      out = out + self.bias
      return out.bfloat16().float()

  @classmethod
  def from_float(cls, mod, rt_spec):
    """Create a quantized module from a float module."""
    assert rt_spec, 'Runtime spec must be provided for quantized module.'
    assert type(mod) == cls._FLOAT_MODULE, \
        '{}.from_float() only accepts {}, but got {}'.format(
          cls.__name__, cls._FLOAT_MODULE, type(mod))
    
    if rt_spec.config:
      approx_mode = rt_spec.config.approx_mode
    elif NndctOption.nndct_ip_v70_bert_qat.value:
      approx_mode = mode.ApproxModes.IP_V70_BERT
    else:
      raise ValueError("approx_mode could nor be None")

    norm = cls(
        mod.normalized_shape,
        mod.eps,
        mod.elementwise_affine,
        approx_mode=approx_mode,
        rt_spec=rt_spec
    )

    norm.weight = mod.weight
    norm.bias = mod.bias
    return norm

  def extra_repr(self):
    return '{}, approx_mode={}'.format(super(LayerNorm, self).extra_repr(), self.approx_mode)
