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

import abc
import copy

from torch import nn

from pytorch_nndct.nn.modules import functional
from pytorch_nndct.utils import logging
from pytorch_nndct.utils import module_util as mod_util

def quantize_input(module, index, quantizer):
  """Insert a quantizer for quantizing the input of the module.

    The input module is modified inplace with added quantizer module
    and forward_pre_hooks.

    Args:
      module: Input module that we want to quantize.
      index: The index of module's inputs to be quantized.
      quantizer: Module of quantizer to be added.
    """

  quantizer_name_template = 'input%d_quantizer'

  def _forward_pre_hook(self, input):
    """Forward hook that calls quantizer on the input"""
    quantized_input = []
    for i, inp in enumerate(input):
      quantizer_name = quantizer_name_template % i
      if hasattr(self, quantizer_name):
        quantized_input.append(getattr(self, quantizer_name)(inp))
      else:
        quantized_input.append(inp)
    return tuple(quantized_input)

  #TODO(yuwang): Support torch.nn.Sequential
  module.add_module(quantizer_name_template % index, quantizer)
  # Register quantizer as the last entry in the hook list.
  # All forward pre hooks are preserved and will be executed before the quantizer.
  quantizer_flag_name = '_input_quantizer_hook_registered'
  if not hasattr(module, quantizer_flag_name):
    setattr(module, quantizer_flag_name, True)
    handle = module.register_forward_pre_hook(_forward_pre_hook)
    module._forward_pre_hooks.move_to_end(handle.id, last=True)

# TODO(yuwang): Maybe support multiple outputs like quantize_input ?
def quantize_output(module, quantizer):
  """Insert a quantizer for quantizing the output of the module.

    The input module is modified inplace with added quantizer module
    and forward_hooks

    Args:
      module: Input module that we want to quantize.
      quantizer: Module of quantizer to be added.
    """

  def _forward_hook(self, input, output):
    """Forward hook that calls quantizer on the output"""
    return self.output_quantizer(output)

  def _forward_hook_max(self, input, output):
    """Forward hook that calls quantizer on the output for functional.Max."""
    quantized_values = self.output_quantizer(output[0])
    # (values, indices)
    return (quantized_values, output[1])

  quantizer_flag_name = '_output_quantizer_hook_registered'
  if hasattr(module, quantizer_flag_name):
    raise RuntimeError('Insert multiple quantizers to a module: {}'.format(
        type(module)))

  #TODO(yuwang): Support torch.nn.Sequential
  module.add_module('output_quantizer', quantizer)
  setattr(module, quantizer_flag_name, True)
  # Register quantizer as the first entry in the hook list.
  # All post forward hooks are preserved and will be executed after the quantizer.
  if isinstance(module, functional.Max):
    handle = module.register_forward_hook(_forward_hook_max)
  else:
    handle = module.register_forward_hook(_forward_hook)
  module._forward_hooks.move_to_end(handle.id, last=False)

def insert_quantizer(model_topo):
  """Insert quantizer for quantizing input/output of a module.
  The quantization of weight/bias is handled by quantized module itself.
  """

  quantized_modules = set()
  for node in model_topo.nodes:
    rt_spec = node.spec
    if not rt_spec:
      continue

    if node.name in quantized_modules:
      continue

    logging.vlog(
        3, 'Inserting quantizer for node {}: {}'.format(node.name, rt_spec))
    quantized_modules.add(node.name)
    for index, quantizer in enumerate(rt_spec.input_quantizers):
      quantize_input(node.module, index, quantizer)

    output_quantizers = rt_spec.output_quantizers
    if len(output_quantizers) > 1:
      raise NotImplementedError('Multiple outputs tensor not supported yet.')

    if output_quantizers:
      quantize_output(node.module, output_quantizers[0])


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

class Transform(abc.ABC):
  """Defines a transform to be applied to a nn.Module object.

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

class ModuleTransformer(object):
  """Matches patterns to apply transforms a nn.Module."""

  def __init__(self, model, model_topo, transforms):
    """Construct ModelTransformer.

    Args:
      model: A nn.Module object to be transformed.
      transforms: List of transforms to be applied to the model.
    """
    self.model = model
    self.model_topo = model_topo
    self.transforms = transforms

  def _match_node_with_inputs(self, node, model_topo, pattern, matched_nodes):
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
      input_node = model_topo.node(node.inputs[i])
      input_match = self._match_node_with_inputs(input_node, model_topo,
                                                 pattern.inputs[i],
                                                 matched_nodes)
      if not input_match:
        return None
      match.inputs.append(input_match)

    return match

  def _match(self, model_topo, pattern):
    matches = []
    matched_nodes = set()
    for node in model_topo.nodes:
      if node.name in matched_nodes:
        continue

      node_match = self._match_node_with_inputs(node, model_topo, pattern,
                                                matched_nodes)
      if node_match:
        self._save_matched_nodes(node_match, matched_nodes)
        matches.append(node_match)
    return matches

  def _save_matched_nodes(self, match, matched_nodes):
    matched_nodes.add(match.node.name)
    for input_match in match.inputs:
      self._save_matched_nodes(input_match, matched_nodes)

  def _rebuild_topo(self, model, topo):
    """Rebuild topology of the given model.

    1. Skiping Identity module.
    Some sub modules of the model may be replaced by nn.Identity in
    transform and these Identity modules can cause patterns to fail to be
    matched.

    For eaxmple, the original topology is Conv2d -> BatchNorm2d -> Add,
    after being transformed by fusing Conv and Bn, the topology would be like
    ConvBatchNorm2d -> Identity -> Add. In this case, the pattern
    NodePattern("Add", inputs=["ConvBatchNorm2d", "ConvBatchNorm2d"]) can not
    be matched.

    2. Updating node's module. The node's corresponding module may change
    after the transformation. Assign the actual module to the node.

    Call this function before doing pattern matching
    to make sure no matches are missed.
    """

    node_to_inputs = {}
    for node in topo.nodes:
      node_to_inputs[node.name] = node.inputs

    for node in topo.nodes:
      for i, inp in enumerate(node.inputs):
        input_node = topo.node(inp)
        while type(input_node.module) == nn.Identity:
          input_node = topo.node(node_to_inputs[inp][0])
        node.inputs[i] = input_node.name

    # Update node's module.
    for node in topo.nodes:
      try:
        node.module = mod_util.get_module(model, node.name)
      except AttributeError:
        pass

  def transform(self, excluded_nodes=None, inplace=False):
    """Transforms the nn.Module by applying all the specified transforms.

    Returns:
      The transformed nn.Module, the topology of the model after transformation
      and a mapping which maps the original module to the replaced module.
    """
    model = self.model if inplace else copy.deepcopy(self.model)
    model_topo = self.model_topo

    replace_map = {}
    for transform in self.transforms:
      self._rebuild_topo(model, model_topo)

      matches = self._match(model_topo, transform.pattern())
      for match in matches:
        if excluded_nodes and match.node.name in excluded_nodes:
          continue

        orig_to_transformed = transform.replace(model, match)
        if not orig_to_transformed:
          continue

        for key in orig_to_transformed.keys():
          if key in replace_map:
            raise RuntimeError(
                'Mapping module "{}" to more than one transformed module'
                .format(key))
        replace_map.update(orig_to_transformed)

    # Make sure the topo is up to date.
    # import pdb
    # pdb.set_trace()
    self._rebuild_topo(model, model_topo)
    insert_quantizer(model_topo)

    return model, model_topo, replace_map
