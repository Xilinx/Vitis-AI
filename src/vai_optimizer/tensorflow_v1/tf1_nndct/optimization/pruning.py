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
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tf1_nndct.optimization.utils import group_conv_nodes, find_weight_nodes, is_matmul, \
  is_conv, is_depthwise_conv, is_concat, is_weighted_node, calculate_flops, get_input_node_name, \
  topo_sort, find_ancestor_target_nodes
from tf1_nndct.optimization.sensitivity import SensAnalyzer, GroupSensitivity, Sensitivity
from tf1_nndct.optimization.spec import PruningSpec
from tf1_nndct.optimization.constant import OpType
from tf1_nndct.optimization.states import MASKS
import numpy as np
import math
from copy import deepcopy
from tensorflow.python.client.session import SessionInterface
from typing import Callable, List, Mapping, Tuple
import multiprocessing as mp
import traceback
import os


ctx = mp.get_context("spawn")
BASE_DIR = ".vai"
if not os.path.exists(BASE_DIR):
  os.makedirs(BASE_DIR)


class PruningDesc(object):
  def __init__(self) -> None:
    self.removed_inputs: List[int] = []
    self.removed_outputs: List[int] = []
    self.in_depth: int = 0
    self.out_depth: int = 0


def _do_eval(eval_fn: Callable[[tf.compat.v1.GraphDef], float], input_queue: mp.Queue, output_queue: mp.Queue):
    idx, gpu_id, frozen_graph_def = input_queue.get()
    try:
      with tf.device(gpu_id):
        score = eval_fn(frozen_graph_def)
    except Exception as e:
      traceback.print_exc()
      output_queue.put(e)
    else:
      output_queue.put((idx, score))


def _submit_to_subprocess(
    idx: int, sess: SessionInterface, output_node_names: List[str],
    eval_fn: Callable[[tf.compat.v1.GraphDef], float], input_queue: mp.Queue, 
    output_queue: mp.Queue, weights: Mapping[str, np.ndarray] = None, gpu_id: str = '/CPU:0') -> mp.Process:
  frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
      sess,
      sess.graph.as_graph_def(),
      output_node_names
  )
  if weights is not None:
    for node_def in frozen_graph_def.node:
      if node_def.name in weights:
        node_def.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(weights[node_def.name]))
  p = ctx.Process(target=_do_eval, args=(eval_fn, input_queue, output_queue))
  p.start()
  input_queue.put((idx, gpu_id, frozen_graph_def))
  return p


class IterativePruningRunner(object):
  def __init__(
      self, model_name: str, sess: SessionInterface, 
      input_specs: Mapping[str, tf.TensorSpec], 
      output_node_names: List[str], excludes: List[str]=[]) -> None:
    self._model_name = model_name
    self._sess = sess
    self._weights = {
        var.name.split(":")[0] : sess.run(var) for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
    self._output_node_names = output_node_names
    self._graph_def = tf.compat.v1.graph_util.extract_sub_graph(
        sess.graph_def, output_node_names)
    self._input_specs = input_specs
    self._excludes = excludes
    self._sens_analyzer = None

  def _fill_in_weights(self, weights: Mapping[str, np.ndarray]) -> None:
    for var in tf.get_collection('variables'):
      var_name = var.name.split(":")[0]
      if var_name in weights:
        self._sess.run(tf.assign(var, weights[var_name]))

  def ana(
      self, eval_fn: Callable[[tf.compat.v1.GraphDef], float], 
      gpu_ids: List[str]=['/GPU:0'], checkpoint_interval: int = 10) -> None:
    sens_path = os.path.join(BASE_DIR, self._model_name + '.sens')
    self._sens_analyzer = SensAnalyzer()
    input_queue = ctx.Queue(maxsize=len(gpu_ids))
    output_queue = ctx.Queue(maxsize=len(gpu_ids))

    if os.path.exists(sens_path):
      self._sens_analyzer.load(sens_path)
    else:
      groups = group_conv_nodes(self._graph_def, self._excludes)
      input_queue = ctx.Queue(maxsize=len(gpu_ids))
      output_queue = ctx.Queue(maxsize=len(gpu_ids))

      p = _submit_to_subprocess(
          0, self._sess, self._output_node_names, eval_fn, input_queue, output_queue, gpu_id=gpu_ids[0])
      ret = output_queue.get()
      assert not isinstance(ret, Exception), "Error occurred during call eval_fn"
      base_score = ret[1]
      p.join()
      for g in groups:
        group_sens = GroupSensitivity(g, [Sensitivity(i * 0.1) for i in range(10)])
        group_sens.sens[0].val = base_score
        self._sens_analyzer.add_group_sens(group_sens)

    unfinished_specs = self._sens_analyzer.unfinished_specs()
    processes: List[mp.Process] = [None for _ in unfinished_specs]

    for _ in gpu_ids:
      output_queue.put(None)

    num_unfinished_tasks = 0
    num_finished_tasks = 0
    for idx, (sens, spec) in enumerate(unfinished_specs):
      ret = output_queue.get()
      assert not isinstance(ret, Exception), "Error occurred during call eval_fn"
      if ret is not None:
        num_unfinished_tasks -= 1
        num_finished_tasks += 1
        if num_finished_tasks % checkpoint_interval == 0:
          self._sens_analyzer.save(sens_path)
        i, score = ret
        unfinished_specs[i][0].val = score
        if processes[i] is not None:
          processes[i].join()
      _, weights, _ = self._prune(spec)
      processes[idx] = _submit_to_subprocess(
          idx, self._sess, self._output_node_names, eval_fn, 
          input_queue, output_queue, weights, gpu_id=gpu_ids[idx % len(gpu_ids)])
      num_unfinished_tasks += 1
    
    for _ in range(num_unfinished_tasks):
      ret = output_queue.get()
      assert not isinstance(ret, Exception), "Error occurred during call eval_fn"
      if ret is not None:
        idx, score = ret
        unfinished_specs[idx][0].val = score
        if processes[idx] is not None:
          processes[idx].join()
    self._sens_analyzer.save(sens_path)

  def _get_spec_by_sparsity(self, sparsity: float, max_attemp: int) -> PruningSpec:
    idx = 0
    flops_tolerance = 1e-2
    min_th = 1e-5
    max_th = 1 - min_th

    base_flops = calculate_flops(self.get_slim_graph_def(), self._input_specs)
    expected_flops = (1 - sparsity) * base_flops
    prev_spec = None
    cur_spec = None
    while idx < max_attemp:
      idx += 1
      threshold = (min_th + max_th) / 2
      cur_spec = self._sens_analyzer.generate_spec_by_threshold(threshold)
      if prev_spec and prev_spec == cur_spec:
        continue

      shape_tensors, _, masks = self._prune(cur_spec)
      current_flops = calculate_flops(self.get_slim_graph_def(shape_tensors, masks), self._input_specs)
      error = abs(base_flops - expected_flops) / base_flops
      if error < flops_tolerance:
        break

      if current_flops < expected_flops:
        max_th = threshold
      else:
        min_th = threshold
      prev_spec = cur_spec
    return cur_spec

  def prune(self, sparsity: float=None, threshold: float=None, max_attemp: int=10) -> Tuple[Mapping[str, TensorProto], Mapping[str, np.ndarray]]:
    assert sparsity is not None or threshold is not None
    if sparsity is not None:
      shape_tensors, weights, masks = self._prune(self._get_spec_by_sparsity(sparsity, max_attemp))
    else:
      shape_tensors, weights, masks = self._prune(self._sens_analyzer.generate_spec_by_threshold(threshold))
    self._fill_in_weights(weights)
    MASKS.update(masks)
    return shape_tensors, masks

  def _prune(self, spec: PruningSpec) -> Tuple[Mapping[str, TensorProto], Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    graph_def = deepcopy(self._graph_def)
    weights = deepcopy(self._weights)

    node_def_map = {n.name: n for n in graph_def.node}
    node_pruning_descs = {n.name: PruningDesc() for n in graph_def.node}
    for group_spec in spec.group_specs:
      weight_node = find_weight_nodes(node_def_map[group_spec.nodes[0]], node_def_map)[0]
      out_depth = weight_node.attr['shape'].shape.dim[3].size
      remain_out_depth = int(out_depth * (1 - group_spec.sparsity))
      remain_out_depth = math.ceil(remain_out_depth / spec.channel_divisible) * spec.channel_divisible
      remain_out_depth = min(max(remain_out_depth, 2), out_depth)
      node_pruning_descs[group_spec.nodes[0]].out_depth = remain_out_depth
      node_pruning_descs[group_spec.nodes[0]].removed_outputs = self._get_channel_indices_to_remove(
          weights[weight_node.name], -1, remain_out_depth)

      for i in range(1, len(group_spec.nodes)):
        node_pruning_descs[group_spec.nodes[i]].out_depth = node_pruning_descs[group_spec.nodes[0]].out_depth
        node_pruning_descs[group_spec.nodes[i]].removed_outputs.extend(
            node_pruning_descs[group_spec.nodes[0]].removed_outputs)
    
    self._shape_inference(graph_def, node_def_map, node_pruning_descs)
    shape_tensors = self._update_shape_tensor(graph_def, node_def_map, node_pruning_descs)
    masks = self._update_weights(graph_def, node_def_map, node_pruning_descs, weights)
    return shape_tensors, weights, masks

  def _shape_inference(self, graph_def: tf.compat.v1.GraphDef, node_def_map: Mapping[str, tf.compat.v1.GraphDef], node_pruning_descs: Mapping[str, PruningDesc]) -> None:
    for node_def in topo_sort(graph_def):
      if len(node_def.input) == 0:
        continue
      pruning_desc = node_pruning_descs[node_def.name]

      if is_matmul(node_def):
        input_pruning_desc = node_pruning_descs[get_input_node_name(node_def.input[0])]
        if len(input_pruning_desc.removed_outputs) == 0:
          continue
        weight_node = find_weight_nodes(node_def, node_def_map)[0]
        input_node_out_depth = (input_pruning_desc.out_depth + len(input_pruning_desc.removed_outputs))
        pruning_desc.removed_inputs = []
        offset = 0
        while offset < weight_node.attr['shape'].shape.dim[0].size:
          pruning_desc.removed_inputs.extend([c + offset for c in input_pruning_desc.removed_outputs])
          offset += input_node_out_depth
      elif is_conv(node_def):
        input_pruning_desc = node_pruning_descs[get_input_node_name(node_def.input[0])]
        if len(input_pruning_desc.removed_outputs) == 0:
          continue
        pruning_desc.removed_inputs.extend(input_pruning_desc.removed_outputs)
      elif is_depthwise_conv(node_def):
        input_pruning_desc = node_pruning_descs[get_input_node_name(node_def.input[0])]
        if len(input_pruning_desc.removed_outputs) == 0:
          continue
        weight_node = find_weight_nodes(node_def, node_def_map)[0]
        group_size = weight_node.attr['shape'].shape.dim[-1].size
        pruning_desc.removed_inputs = [i for i in input_pruning_desc.removed_outputs]
        pruning_desc.removed_outputs = []
        for i in input_pruning_desc.removed_outputs:
          pruning_desc.removed_outputs.extend(list(range(i * group_size, (i + 1) * group_size)))
        pruning_desc.out_depth = input_pruning_desc.out_depth * group_size
      elif is_concat(node_def):
        pruning_desc.removed_outputs = []
        pruning_desc.out_depth = 0
        offset = 0
        for inpt in node_def.input:
          input_pruning_desc = node_pruning_descs[get_input_node_name(inpt)]
          pruning_desc.out_depth += input_pruning_desc.out_depth
          pruning_desc.removed_outputs.extend([c + offset for c in input_pruning_desc.removed_outputs])
          offset += (input_pruning_desc.out_depth + len(input_pruning_desc.removed_outputs))
      else:
        input_pruning_desc = node_pruning_descs[get_input_node_name(node_def.input[0])]
        if len(input_pruning_desc.removed_outputs) == 0:
          continue
        pruning_desc.removed_inputs = [i for i in input_pruning_desc.removed_outputs]
        pruning_desc.removed_outputs = [i for i in pruning_desc.removed_inputs]
        pruning_desc.out_depth = input_pruning_desc.out_depth

  def _update_shape_tensor(
      self, graph_def: tf.compat.v1.GraphDef, node_def_map: Mapping[str, tf.compat.v1.GraphDef], 
      node_pruning_descs: Mapping[str, PruningDesc]) -> Mapping[str, TensorProto]:
    ret = {}
    for node_def in graph_def.node:
      if node_def.name not in node_pruning_descs:
        continue
      pruning_desc = node_pruning_descs[node_def.name]
      if node_def.op == OpType.Mul and node_def.name.endswith("dropout/mul_1"):
        for const_node in find_ancestor_target_nodes(node_def_map, node_def.input[1], [OpType.Const], [], True):
          if const_node.name.endswith("dropout/Shape"):
            shape_tensor = tf.make_ndarray(const_node.attr['value'].tensor)
            shape_tensor[-1] -= len(pruning_desc.removed_outputs)
            ret[const_node.name] = tf.make_tensor_proto(shape_tensor)
    return ret

  def _update_weights(
      self, graph_def: tf.compat.v1.GraphDef, node_def_map: Mapping[str, tf.compat.v1.GraphDef], 
      node_pruning_descs: Mapping[str, PruningDesc], weights: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    masks = {}
    for node_def in graph_def.node:
      if not is_weighted_node(node_def) or node_def.name not in node_pruning_descs:
        continue
      pruning_desc = node_pruning_descs[node_def.name]
      if len(pruning_desc.removed_inputs) == 0 and len(pruning_desc.removed_outputs) == 0:
        continue

      processed_weights = set()
      for weight_node in find_weight_nodes(node_def, node_def_map):
        if weight_node.name in processed_weights:
          continue
        if weight_node.name in weights:
          weight_array = weights[weight_node.name]
          mask_array = np.ones_like(weight_array)
        elif weight_node.op == OpType.Const:
          weight_array = tf.make_ndarray(weight_node.attr['value'].tensor)
          if weight_array.size == 0:
            continue
          mask_array = np.ones_like(weight_array)
        else:
          continue
        masks[weight_node.name] = mask_array
        dim_size = len(weight_array.shape)
        if dim_size == 1:
          for idx in pruning_desc.removed_outputs:
            weight_array[idx] = 0
            mask_array[idx] = 0
        else:
          for idx in pruning_desc.removed_inputs:
            weight_array[..., idx, :] = 0
            mask_array[..., idx, :] = 0
          if not is_depthwise_conv(node_def):
            for idx in pruning_desc.removed_outputs:
              weight_array[..., idx] = 0
              mask_array[..., idx] = 0
        processed_weights.add(weight_node.name)
    return masks

  def _get_channel_indices_to_remove(self, weight: np.ndarray, axis: int, remain_depth: int) -> List[int]:
    weight = abs(weight)
    dim_size = len(weight.shape)
    sums = []
    axis = axis % dim_size
    if axis == 0:
      for i in range(weight.shape[0]):
        sums.append(weight[i, ...].sum())
    elif axis == dim_size - 1:
      for i in range(weight.shape[dim_size - 1]):
        sums.append(weight[..., i].sum())
    else:
      for i in range(weight.shape[axis]):
        sums.append(weight[slice(0, axis), i, slice(axis + 1, dim_size)])
    sums = np.array(sums)
    sorted_sum_indices = np.argsort(sums)
    return list(sorted_sum_indices[0: weight.shape[axis] - remain_depth])

  def _get_slim_ndarray(self, array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dim_size = len(array.shape)
    assert dim_size in [1, 2, 4]

    output_to_remove = []
    for i in range(mask.shape[-1]):
      if mask[..., i].sum() == 0:
        output_to_remove.append(i)
    array = np.delete(array, output_to_remove, axis=-1)
    if dim_size == 4 or dim_size == 2:
      input_to_remove = []
      for i in range(mask.shape[-2]):
        if mask[..., i, :].sum() == 0:
          input_to_remove.append(i)
      array = np.delete(array, input_to_remove, axis=-2)
    return array    

  def get_slim_graph_def(self, shape_tensors: Mapping[str, TensorProto]=None, masks: Mapping[str, np.ndarray]=None) -> tf.compat.v1.GraphDef:
    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        self._sess,
        self._graph_def,
        self._output_node_names
    )
    frozen_graph_def_map = {node.name: node for node in frozen_graph_def.node}
    if shape_tensors is not None:
      for k, tensor_proto in shape_tensors.items():
        if k in frozen_graph_def_map:
          node_def = frozen_graph_def_map[k]
          node_def.attr['value'].tensor.CopyFrom(tensor_proto)

    if masks is not None:
      for k, mask in masks.items():
        if k in frozen_graph_def_map:
          node_def = frozen_graph_def_map[k]
          array = tf.make_ndarray(node_def.attr['value'].tensor)
          node_def.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(self._get_slim_ndarray(array, mask)))

    return frozen_graph_def
