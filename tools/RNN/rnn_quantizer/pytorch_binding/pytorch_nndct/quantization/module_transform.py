

#
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import copy
import pytorch_nndct.nn.qat as nnqat
import six
import torch

from pytorch_nndct.utils import module_util as mod_util

class NodeMatch(object):

  def __init__(self, node):
    self.node = node
    self.inputs = []

class NodePattern(object):
  """Defines a tree sub-graph pattern of nodes to match in a nndct graph.

  Examples:
    Matches a Conv+BN+ReLU6 and DepthwiseConv+BN+ReLU6 pattern.
    pattern = ModulePattern('ReLU', {'max_value': 6.0}, [
        ModulePattern('BatchNormalization', {}, [
            ModulePattern('Conv2D|DepthwiseConv2D', {} [])
        ])
    ])

    Matches multiple Conv2Ds feeding into a Concat.
    pattern = ModulePattern('Concat', {}, [
        ModulePattern('Conv2D', {}, []),
        ModulePattern('Conv2D', {}, [])
    ])
  """

  def __init__(self, module_name, inputs=None):
    """Construct pattern to match.

    Args:
      module_name: Type name of module. (such as Conv2d, Linear etc.)
      inputs: input modules to the module.
    """
    if inputs is None:
      inputs = []

    self.module_name = module_name
    self.inputs = inputs

  def __str__(self):
    return '{} <- [{}]'.format(self.module_name,
                               ', '.join([str(inp) for inp in self.inputs]))

def fuse_conv_bn(conv, bn, qconfig):
  """Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of torch.nn.Conv2d
        bn: torch.nn.BatchNorm2d instance that needs to be fused with the conv
        qconfig: Quantization config used to initialize the fused module.

    Return:
      The fused module.
  """

  assert(conv.training == bn.training),\
      "Conv and BN both must be in the same mode (train or eval)."

  assert conv.training

  assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
  assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
  assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
  return nnqat.ConvBatchNorm2d.from_float(conv, bn, qconfig)

def replace_modules(model, module_names, new_modules):
  if not isinstance(module_names, (tuple, list)):
    module_names = [module_names]
  if not isinstance(new_modules, (tuple, list)):
    new_modules = [new_modules]
  assert len(new_modules) <= len(module_names)

  modules = []
  for name in module_names:
    modules.append(mod_util.get_module(model, name))

  for handle_id, pre_hook_fn in modules[0]._forward_pre_hooks.items():
    new_modules[0].register_forward_pre_hook(pre_hook_fn)
    del modules[0]._forward_pre_hooks[handle_id]
  # Move post forward hooks of the last module to resulting fused module
  for handle_id, hook_fn in modules[-1]._forward_hooks.items():
    new_modules[-1].register_forward_hook(hook_fn)
    del modules[-1]._forward_hooks[handle_id]

  # Use Identity to fill in the missing modules.
  for i in range(len(module_names) - len(new_modules)):
    new_modules.append(torch.nn.Identity())
    new_modules[i].training = modules[0].training

  for i, name in enumerate(module_names):
    mod_util.set_module(model, name, new_modules[i])

@six.add_metaclass(abc.ABCMeta)
class Transform(object):
  """Defines a transform to be applied to a torch.nn.Module object.

  A transform is a combination of 'Find + Replace' which describes how to find
  a pattern of layers in a model, and what to replace those layers with.

  A pattern is described using `LayerPattern`. The replacement function receives
  a `LayerNode` which contains the matched layers and should return a
  `LayerNode` which contains the set of layers which replaced the matched
  layers.
  """

  @abc.abstractmethod
  def pattern(self):
    """Return the `LayerPattern` to find in the model graph."""
    raise NotImplementedError()

  @abc.abstractmethod
  def replace(self, model, match):
    """Generate a replacement sub-graph for the matched sub-graph.

    The fundamental constraint of the replacement is that the replacement
    sub-graph should consume the same input tensors as the original sub-graph
    and also produce a final list of tensors which are same in number and shape
    as the original sub-graph. Not following this could crash model creation,
    or introduce bugs in the new model graph.


    Args:
      model: The replacement performed on.
      match: Matched sub-graph based on `self.pattern()`.
    """
    raise NotImplementedError()

class FuseAndQuantizeConv2dBatchNorm(Transform):

  def pattern(self):
    return NodePattern('BatchNorm2d', inputs=[NodePattern('Conv2d')])

  def replace(self, model, match):
    # Fuse Conv2d and BatchNorm2d
    conv_match, bn_match = match.inputs[0].node, match.node
    conv_name, bn_name = conv_match.name, bn_match.name
    conv, bn = conv_match.module, bn_match.module
    #print('Fusing {} and {}'.format(conv_name, bn_name))
    conv_bn = fuse_conv_bn(conv, bn, conv_match.qconfig)
    replace_modules(model, [conv_name, bn_name], conv_bn)
    return {bn_name: conv_name + '.bn'}

class QuantizeLinear(Transform):

  def pattern(self):
    return NodePattern('Linear')

  def replace(self, model, match):
    replace_modules(model, match.node.name,
        nnqat.Linear.from_float(match.node.module, match.node.qconfig))

class ReplaceAdaptiveAvgPool2d(Transform):

  def pattern(self):
    return NodePattern('AdaptiveAvgPool2d')

  def replace(self, model, match):
    op = match.node.op
    attrs = {name: op.get_config(name) for name in op.configs}
    #print('Replace AdaptiveAvgPool2d with AvgPool2d using', attrs)
    avgpool2d = nnqat.AvgPool2d(**attrs)
    replace_modules(model, match.node.name, avgpool2d)

class ModuleTransformer(object):
  """Matches patterns to apply transforms a torch.nn.Module."""

  def __init__(self, model, model_topo, transforms):
    """Construct ModelTransformer.

    Args:
      model: A torch.nn.Module object to be transformed.
      transforms: List of transforms to be applied to the model.
    """
    self.model = model
    self.model_topo = model_topo
    self.transforms = transforms

  def _match_node_with_inputs(self, node, pattern, matched_nodes):
    """Match pattern at this node, and continue to match at its inputs."""
    if node.name in matched_nodes:
      return None

    # Only match module nodes.
    if not node.module:
      return None

    matched = False
    if pattern.module_name == '*':
      matched = True
    else:
      module_names = pattern.module_name.split('|')
      for module_name in module_names:
        if node.module._get_name() == module_name:
          matched = True
          break

    if not matched:
      return None

    match = NodeMatch(node)

    if len(pattern.inputs) == 0:
      # Leaf node in pattern.
      return match

    if len(pattern.inputs) != len(node.inputs):
      return None
    for i, input_pattern in enumerate(pattern.inputs):
      input_node = self.model_topo.node(node.inputs[i])
      input_match = self._match_node_with_inputs(input_node, pattern.inputs[i],
                                                 matched_nodes)
      if not input_match:
        return None
      match.inputs.append(input_match)

    return match

  def _match(self, pattern, strict=False):
    self._update_node_module()
    if not strict:
      self._rebuild_topo()

    matches = []
    matched_nodes = set()
    for node in self.model_topo.nodes:
      if node.name in matched_nodes:
        continue

      node_match = self._match_node_with_inputs(node, pattern, matched_nodes)
      if node_match:
        self._save_matched_nodes(node_match, matched_nodes)
        matches.append(node_match)
    return matches

  def _save_matched_nodes(self, match, matched_nodes):
    matched_nodes.add(match.node.name)
    for input_match in match.inputs:
      self._save_matched_nodes(input_match, matched_nodes)

  def _update_node_module(self):
    for node in self.model_topo.nodes:
      try:
        node.module = mod_util.get_module(self.model, node.name)
      except AttributeError:
        pass

  def _rebuild_topo(self):
    """Rebuild topology of the model by skiping Identity module.

    Some sub modules of the model may be replaced by torch.nn.Identity in
    transform and these Identity modules can cause patterns to fail to be
    matched.

    For eaxmple, the original topology is Conv2d -> BatchNorm2d -> Add,
    after being transformed by fusing Conv and Bn, the topology would be like
    ConvBatchNorm2d -> Identity -> Add. In this case, the pattern
    NodePattern("Add", inputs=["ConvBatchNorm2d", "ConvBatchNorm2d"]) can not
    be matched.

    Call this function before doing pattern matching
    to make sure no matches are missed.
    """

    topo = self.model_topo

    node_to_inputs = {}
    for node in topo.nodes:
      node_to_inputs[node.name] = node.inputs

    for node in topo.nodes:
      for i, inp in enumerate(node.inputs):
        input_node = topo.node(inp)
        while type(input_node.module) == torch.nn.Identity:
          input_node = topo.node(node_to_inputs[inp][0])
        node.inputs[i] = input_node.name
        #print('Skip identity: {} <- {}'.format(node.name, input_node.name))

  def transform(self, inplace=False):
    """Transforms the torch.nn.Module by applying all the specified transforms.

    Returns:
      The transformed torch.nn.Module.
    """
    model = self.model if inplace else copy.deepcopy(self.model)

    replace_map = {}

    for transform in self.transforms:
      matches = self._match(transform.pattern())

      for match in matches:
        orig_to_transformed = transform.replace(model, match)
        if not orig_to_transformed:
          continue

        for key in orig_to_transformed.keys():
          if key in replace_map:
            raise RuntimeError(
                'Mapping module "{}" to more than one transformed module'.format(key))
        replace_map.update(orig_to_transformed)

    return model, replace_map
