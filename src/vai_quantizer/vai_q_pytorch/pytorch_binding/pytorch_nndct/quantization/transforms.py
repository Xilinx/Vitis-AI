# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from torch import nn
try:
  from torch.nn.modules.conv import _ConvTransposeNd
except ImportError:
  from torch.nn.modules.conv import _ConvTransposeMixin as _ConvTransposeNd

from pytorch_nndct.nn import quantization as nnq
from pytorch_nndct.quantization import module_transform
from pytorch_nndct.utils import fusion
from pytorch_nndct.utils import module_util as mod_util

Transform = module_transform.Transform
NodeMatch = module_transform.NodeMatch
NodePattern = module_transform.NodePattern

def quantize_conv_bn(conv, bn, spec):
  """Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of (nn.Conv2d, nn.Conv3d)
        bn: nn.BatchNorm2d or nn.BatchNorm3d instance that needs
            to be fused with the conv
        spec: Runtime specification used to initialize the fused module.

    Return:
      The fused module.
  """

  foldable_patterns = {
      (nn.Conv2d, nn.BatchNorm2d):
          nnq.QuantizedConvBatchNorm2d,
      (nn.Conv3d, nn.BatchNorm3d):
          nnq.QuantizedConvBatchNorm3d,
      (nn.ConvTranspose2d, nn.BatchNorm2d):
          nnq.QuantizedConvTransposeBatchNorm2d,
      (nn.ConvTranspose3d, nn.BatchNorm3d):
          nnq.QuantizedConvTransposeBatchNorm3d,
  }

  cls_pair = (type(conv), type(bn))
  assert cls_pair in foldable_patterns
  assert(conv.training == bn.training),\
      "Conv and BN both must be in the same mode (train or eval)."

  assert conv.training

  assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
  assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
  assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'

  fused_cls = foldable_patterns[cls_pair]
  #print('fuse', cls_pair, '->', fused_cls)
  return fused_cls.from_float(conv, bn, spec)

class _FuseConvNdBatchNorm(Transform):

  def replace(self, model, match):
    conv_match, bn_match = match.inputs[0].node, match.node
    conv_name, bn_name = conv_match.name, bn_match.name
    conv, bn = conv_match.module, bn_match.module

    fusion.fuse_conv_bn(conv, bn)
    mod_util.replace_modules(model, bn_name, nn.Identity())

class FuseConv2dBatchNorm(_FuseConvNdBatchNorm):

  def pattern(self):
    return NodePattern(
        'BatchNorm2d', inputs=[NodePattern('Conv2d|ConvTranspose2d')])

class FuseConv3dBatchNorm(_FuseConvNdBatchNorm):

  def pattern(self):
    return NodePattern(
        'BatchNorm3d', inputs=[NodePattern('Conv3d|ConvTranspose3d')])

class QuantizeConvNdBatchNorm(Transform):

  def replace(self, model, match):
    # Fuse (Conv2d, BatchNorm2d) and (Conv3d, BatchNorm3d)
    conv_match, bn_match = match.inputs[0].node, match.node
    conv_name, bn_name = conv_match.name, bn_match.name
    conv, bn = conv_match.module, bn_match.module
    #transposed = True if isinstance(conv, _ConvTransposeNd) else False
    conv_bn = quantize_conv_bn(conv, bn, conv_match.spec)
    mod_util.replace_modules(model, [conv_name, bn_name], conv_bn)
    return {bn_name: conv_name + '.bn'}

class QuantizeConv2dBatchNorm(QuantizeConvNdBatchNorm):

  def pattern(self):
    return NodePattern(
        'BatchNorm2d', inputs=[NodePattern('Conv2d|ConvTranspose2d')])

class QuantizeConv3dBatchNorm(QuantizeConvNdBatchNorm):

  def pattern(self):
    return NodePattern(
        'BatchNorm3d', inputs=[NodePattern('Conv3d|ConvTranspose3d')])

class QuantizeConvNd(Transform):

  def pattern(self):
    return NodePattern('Conv1d|Conv2d|Conv3d|ConvTranspose2d|ConvTranspose3d')

  def replace(self, model, match):
    float_to_qat = {
        nn.Conv1d: nnq.QuantizedConv1d,
        nn.Conv2d: nnq.QuantizedConv2d,
        nn.Conv3d: nnq.QuantizedConv3d,
        nn.ConvTranspose2d: nnq.QuantizedConvTranspose2d,
        nn.ConvTranspose3d: nnq.QuantizedConvTranspose3d,
    }
    matched_module = match.node.module
    qat_cls = float_to_qat[type(matched_module)]
    mod_util.replace_modules(
        model, match.node.name,
        qat_cls.from_float(matched_module, match.node.spec))

class QuantizeLinear(Transform):

  def pattern(self):
    return NodePattern('Linear')

  def replace(self, model, match):
    mod_util.replace_modules(
        model, match.node.name,
        nnq.QuantizedLinear.from_float(match.node.module, match.node.spec))

class ReplacePooling2d(Transform):

  def pattern(self):
    return NodePattern('AvgPool2d|AdaptiveAvgPool2d')

  def replace(self, model, match):
    op = match.node.op
    attrs = {name: op.get_config(name) for name in op.configs}

    replacement_map = {
        nn.AvgPool2d: nnq.DPUAvgPool2d,
        nn.AdaptiveAvgPool2d: nnq.DPUAdaptiveAvgPool2d,
    }
    pool2d = replacement_map[type(match.node.module)](**attrs)
    mod_util.replace_modules(model, match.node.name, pool2d)

class ReplaceLeakyReLU(Transform):

  def pattern(self):
    return NodePattern('LeakyReLU')

  def replace(self, model, match):
    op = match.node.op
    attrs = {name: op.get_config(name) for name in op.configs}
    relu = nnq.DPULeakyReLU(*attrs)
    mod_util.replace_modules(model, match.node.name, relu)

class ReplaceSoftmax(Transform):

  def pattern(self):
    return NodePattern('Softmax')

  def replace(self, model, match):
    rt_cfg = match.node.spec.config
    softmax = nnq.Softmax(match.node.module.dim, rt_cfg.approx_mode,
                          rt_cfg.approx_degree, rt_cfg.exp_table_size)
    mod_util.replace_modules(model, match.node.name, softmax)

class ReplaceGELU(Transform):

  def pattern(self):
    return NodePattern('GELU')

  def replace(self, model, match):
    op = match.node.op
    attrs = {name: op.get_config(name) for name in op.configs}
    rt_cfg = match.node.spec.config
    gelu = nnq.GELU(
        **attrs,
        approx_mode=rt_cfg.approx_mode,
        approx_degree=rt_cfg.approx_degree)
    mod_util.replace_modules(model, match.node.name, gelu)

class ReplaceSigmoid(Transform):

  def pattern(self):
    return NodePattern('Sigmoid')

  def replace(self, model, match):
    rt_cfg = match.node.spec.config
    sigmoid = nnq.Sigmoid(rt_cfg.approx_mode,
                          rt_cfg.approx_degree, rt_cfg.exp_table_size)
    mod_util.replace_modules(model, match.node.name, sigmoid)

class ReplaceTanh(Transform):

  def pattern(self):
    return NodePattern('Tanh')

  def replace(self, model, match):
    rt_cfg = match.node.spec.config
    tanh = nnq.Tanh(rt_cfg.approx_mode,
                    rt_cfg.approx_degree, rt_cfg.exp_table_size)
    mod_util.replace_modules(model, match.node.name, tanh)

class ReplaceLayerNorm(Transform):

  def pattern(self):
    return NodePattern('LayerNorm')

  def replace(self, model, match):
    mod_util.replace_modules(
        model, match.node.name,
        nnq.LayerNorm.from_float(match.node.module, match.node.spec))
