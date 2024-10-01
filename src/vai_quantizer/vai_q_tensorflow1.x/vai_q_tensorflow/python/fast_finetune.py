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

import os
import shutil
import argparse
import importlib
import sys
import time
import tempfile
from copy import deepcopy
from progressbar import ProgressBar

import tensorflow as tf
from tensorflow.core.framework import types_pb2, graph_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client.session import Session
from tensorflow.python import Graph

from .utils import *
from .quantize_graph import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.contrib.graph_editor as ge


def _gen_trainable_graph(quant_graph_def, q_config):
  """Convert frozen graph to trainable graph for fast finetune"""
  name_to_nodes = get_name_to_nodes_map(quant_graph_def)
  name_to_inputnodes = get_name_to_input_nodes(quant_graph_def)

  node_vars = {}

  # Find constants that need conversion
  convert_names = []

  for node in quant_graph_def.node:
    if not ((node.op == "Conv2D" or node.op == "Conv3D") or \
            (node.op == "DepthwiseConv2d" or node.op == "DepthwiseConv2dNative") or \
            (node.op == "MatMul" or node.op == "Dense") or \
            (node.op == "BiasAdd" or node.op == "Add" or node.op == "AddV2")):
      continue  # skip ops that is out of the white list

    node_weights = []

    for in_node in name_to_inputnodes[node.name]:
      if not (in_node.op == "FixNeuron" and in_node.name.endswith("/wquant")):
        continue  # skip act inputs of this node

      const_name = in_node.name[:-7]  # filter "/wquant"
      if (const_name in name_to_nodes and
          name_to_nodes[const_name].op == "Const"):
        node_weights.append(const_name)

    if len(node_weights):
      convert_names.extend(node_weights)

    node_vars[node.name] = node_weights

  # Convert constants to variables
  const_var_name_pairs = []

  graph = tf.Graph()
  with graph.as_default():
    tf.graph_util.import_graph_def(quant_graph_def, name='')

    with tf.Session() as sess:
      progress = ProgressBar()
      for index in progress(range(0, len(convert_names))):
        name = convert_names[index]
        #print("INFO: Fast Finetune {} variables #{} {}".format(
        #      len(convert_names), index, name))

        tensor = graph.get_tensor_by_name('{}:0'.format(name))
        tensor_as_numpy_array = sess.run(tensor)

        var_shape = tensor.get_shape()
        var_name = '{}_var'.format(name)
        var = tf.compat.v1.get_variable(
            name=var_name,
            dtype='float32',
            shape=var_shape,
            initializer=tf.constant_initializer(tensor_as_numpy_array))

        const_var_name_pairs.append((name, var_name))

    for const_name, var_name in const_var_name_pairs:
      const_op = graph.get_operation_by_name(const_name)
      var_reader_op = graph.get_operation_by_name(var_name + '/read')
      ge.swap_outputs(ge.sgv(const_op), ge.sgv(var_reader_op))

    #with tf.Session() as sess:
    #  sess.run(tf.compat.v1.global_variables_initializer())
    #  tf.compat.v1.train.Saver().save(sess, os.path.join(q_config.output_dir,
    #                                              "decent_debug/ft_training.ckpt"))

    #  frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
    #                                           sess.graph_def, q_config.output_nodes)
    #  save_pb_file(frozen_graph_def, os.path.join(q_config.output_dir,
    #                                              "decent_debug/ft_training.pb"))

    return graph, node_vars


def _get_target_modules(quant_graph_def):
  """Transform quantized graph to modules for fast finetune"""

  def _be_fix_neuron(node):
    '''is FixNeuron node or not'''
    if node.op == "FixNeuron":
      return True
    else:
      return False

  def _be_compute_node(node, name_to_nodes):
    '''is compute node or not, compute node contains weights'''
    if _be_fix_neuron(node):
      return False
    for inname in node.input:
      if inname not in name_to_nodes:
        continue  # will ignore multiply outputs node here
      innode = name_to_nodes[inname]
      if _be_fix_neuron(innode) and inname.endswith("wquant"):
        return True
    return False

  target_modules = []

  name_to_nodes = get_name_to_nodes_map(quant_graph_def)
  name_to_outputnodes = get_name_to_output_nodes(quant_graph_def)

  traveled_nodes = []  # avoid record overlaped
  for node in quant_graph_def.node:
    if _be_compute_node(node, name_to_nodes) and (node.name
                                                  not in traveled_nodes):
      node_cur = node
      module = [node_cur.name]  # module[0] is start node

      while (len(name_to_outputnodes[node_cur.name]) == 1):
        nodex = name_to_outputnodes[node_cur.name][0]

        if (_be_fix_neuron(nodex)):
          # note this FixNeuron is skiped
          break
        else:
          node_cur = nodex
          module.append(node_cur.name)  # module[-1] is end node

      traveled_nodes.extend(module)

      target_modules.append(module)  # module may contains only one node

  return target_modules


def _get_refine_modules(input_graph_def, target_modules):
  """Use float graph to align the modules for fast finetune"""
  node_names = []
  for node in input_graph_def.node:
    node_names.append(node.name)

  refine_modules = []

  for module in target_modules:
    if (module[0] in node_names) and (module[-1] in node_names):
      refine_modules.append(module)
    else:
      print('INFO: Fast Finetune ignored module', module[0], '->', module[-1])

  return refine_modules


def _get_module_ionames(graph_def, modules):
  """Get the input and output nodes name for modules"""

  def _weights_identity(node, name_to_inputnodes):
    if node.op == "Identity" and len(node.input) == 1:
      in_nodes = name_to_inputnodes[node.name]
      for in_node in in_nodes:
        if in_node.op == "Const":
          return True
    return False

  name_to_nodes = get_name_to_nodes_map(graph_def)
  name_to_inputnodes = get_name_to_input_nodes(graph_def)
  name_to_outputnodes = get_name_to_output_nodes(graph_def)

  module_ionames = []

  for module in modules:
    layer_in_nodes = name_to_inputnodes[module[0]]
    layer_in = [ # may be multiple nodes, and we should
      # ignore weights node or identity in float graph,
      # ignore weights fix neuron in quant graph also
      node.name for node in layer_in_nodes
      if not (node.op == "Const" or \
              _weights_identity(node, name_to_inputnodes) or \
             (node.op == "FixNeuron" and node.name.endswith("/wquant")))
    ]
    # merge input fix neuron in quant graph
    for i, name in enumerate(layer_in):
      node = name_to_nodes[name]
      if (len(node.input) == 1 and node.op == "FixNeuron" and \
                                   node.name.endswith("/aquant")):
        layer_in[i] = node.input[0]

    act_node = name_to_nodes[module[-1]]
    act_out_nodes = name_to_outputnodes[act_node.name]
    # merge output fix neuron in quant gragh
    if (len(act_out_nodes) == 1 and act_out_nodes[0].op == "FixNeuron" and \
                                    act_out_nodes[0].name.endswith("/aquant")):
      act_node = act_out_nodes[0]
    act_out = [act_node.name]  # always only one node

    # append layer and activation to the list
    module_ionames.append({'layer': layer_in, 'act': act_out})

  return module_ionames


def _get_module_io(input_graph_def, input_fn, q_config, s_config, module_inodes,
                   module_onodes):
  """Get the input and output tensors of input model for training"""
  input_graph = tf.Graph()
  with input_graph.as_default():
    tf.graph_util.import_graph_def(input_graph_def, name='')

    input_tensors = [
        op.outputs[0]
        for op in input_graph.get_operations()
        if op.type == 'Placeholder'
    ]

    module_itensors = [
        input_graph.get_tensor_by_name(name + ':0') for name in module_inodes
    ]
    module_otensors = [
        input_graph.get_tensor_by_name(name + ':0') for name in module_onodes
    ]

    module_io = []

    with Session(graph=input_graph, config=s_config) as sess:
      for it in range(0, q_config.calib_iter):
        inputs = input_fn(it)
        feed_dict = gen_feed_dict(input_tensors, inputs)
        module_inputs, module_outputs = sess.run(
            [module_itensors, module_otensors], feed_dict)

        # each element is module's inputs and outputs in this iteration
        module_io.append({'layer': module_inputs, 'act': module_outputs})

    return module_io


def _get_module_variables(node_vars, module):
  """Get the variables within the module for training"""
  variable_names = []

  for node_name in module:
    if (node_name in node_vars and len(node_vars[node_name])):
      for var_name in node_vars[node_name]:
        variable_names.append('{}_var:0'.format(var_name))

  return variable_names


def _fast_ft_epoch(sess,
                   input_fn,
                   q_config,
                   s_config,
                   input_tensors,
                   module_io,
                   quant_feed,
                   place_holder,
                   loss,
                   learning_rate=None,
                   lr=0,
                   train=None):
  """Do 1 epoch inference or training for fast finetune"""
  epoch_loss = 0

  progress = ProgressBar()
  for it in progress(range(0, q_config.calib_iter)):
    # Feed dict contains model inputs,
    # finetune module inputs, outputs and so on
    inputs = input_fn(it)
    feed_dict = gen_feed_dict(input_tensors, inputs)
    for i, tensor in enumerate(quant_feed):
      feed_dict[tensor] = module_io[it]['layer'][i]
    feed_dict[place_holder] = module_io[it]['act'][0]

    # Compute loss only or train variables
    if learning_rate is None or lr <= 0 or train is None:
      l = sess.run(loss, feed_dict)
    else:
      feed_dict[learning_rate] = lr
      _, l = sess.run([train, loss], feed_dict)

    epoch_loss += l

  return epoch_loss / q_config.calib_iter


def fast_ft(input_graph_def,
            input_fn,
            q_config,
            s_config,
            quant_graph_def,
            temp_path,
            fast_ft_mode=1,
            fast_ft_epochs=1,
            fast_ft_lr=1e-6,
            fast_ft_lrcoef=1.0):
  """Fast Finetune workflow"""
  input_names = q_config.input_nodes
  output_names = q_config.output_nodes

  target_modules = _get_target_modules(quant_graph_def)
  refine_modules = _get_refine_modules(input_graph_def, target_modules)

  float_module_ionames = _get_module_ionames(input_graph_def, refine_modules)
  quant_module_ionames = _get_module_ionames(quant_graph_def, refine_modules)

  print("INFO: Fast Finetune is generating trainable graph...")
  #ft_graph_def = ConvertConstantsToVariables(quant_graph_def, q_config)
  ft_graph, node_vars = _gen_trainable_graph(quant_graph_def, q_config)

  for index, module in enumerate(refine_modules):
    print("INFO: Fast Finetuning module({}/{}): {} -> {}".format(
        index + 1, len(refine_modules), module[0], module[-1]))

    variables_names = _get_module_variables(node_vars, module)
    if (len(variables_names) == 0):
      print("INFO: Skip this module because no variables found.",
            "nodes in the module are", [name for name in module],
            "nodes with weights are", [name for name in node_vars])
      continue

    float_ionames = float_module_ionames[index]
    quant_ionames = quant_module_ionames[index]
    if (fast_ft_mode == 0 and
        len(float_ionames['layer']) != len(quant_ionames['layer'])):
      print(
          "WARN: Skip this module because float inputs {}->{} do not match quant inputs {}->{}"
          .format(float_ionames['layer'], float_ionames['act'],
                  quant_ionames['layer'], quant_ionames['act']))
      continue

    module_io = _get_module_io(input_graph_def, input_fn, q_config, s_config,
                               float_ionames['layer'], float_ionames['act'])

    quant_feed = [
        ft_graph.get_tensor_by_name(name + ':0')
        for name in quant_ionames['layer']
        if fast_ft_mode == 0
    ]

    with ft_graph.as_default():
      trainable_variables = tf.compat.v1.trainable_variables()
      train_variables = [
          var for var in trainable_variables if var.name in variables_names
      ]

      input_tensors = [
          ft_graph.get_tensor_by_name('{}:0'.format(name))
          for name in input_names
      ]
      fetch_tensor = ft_graph.get_tensor_by_name(quant_ionames['act'][0] + ':0')
      place_holder = tf.compat.v1.placeholder(tf.float32,
                                              fetch_tensor.get_shape())
      loss = tf.reduce_mean(tf.square(fetch_tensor - place_holder))

      learning_rate = tf.compat.v1.placeholder(tf.float32, shape=())
      train = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
          loss, var_list=train_variables)
      saver = tf.compat.v1.train.Saver(trainable_variables)

      with Session(graph=ft_graph, config=s_config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        checkpoint_path = os.path.join(temp_path, 'fast_finetune-ckpt')

        # Save variables at first time, later restore variables
        if (os.path.exists(checkpoint_path + '.index') == False):
          saver.save(sess, checkpoint_path, write_meta_graph=False)
        else:
          saver.restore(sess, tf.compat.v1.train.latest_checkpoint(temp_path))

        # Compute original loss
        best_loss = _fast_ft_epoch(sess, input_fn, q_config, s_config,
                                   input_tensors, module_io, quant_feed,
                                   place_holder, loss)
        print("INFO: Fast Finetuning module({}/{}): best_loss {}".format(
            index + 1, len(refine_modules), best_loss))

        # Adjust initial learning rate
        lrcoef = 1.0 if best_loss <= 3.0 else fast_ft_lrcoef
        lr = max(1e-10, min(1e-1, fast_ft_lr * lrcoef))

        for epoch in range(0, fast_ft_epochs):
          # Train and get loss of current epoch
          epoch_loss = _fast_ft_epoch(sess, input_fn, q_config, s_config,
                                      input_tensors, module_io, quant_feed,
                                      place_holder, loss, learning_rate, lr,
                                      train)
          print(
              "INFO: Fast Finetuning epoch {}/{} best_loss {} epoch_loss {} (lr {})"
              .format(epoch + 1, fast_ft_epochs, best_loss, epoch_loss, lr))

          # Continue or restore
          if (epoch_loss < best_loss):
            saver.save(sess, checkpoint_path, write_meta_graph=False)
            best_loss = epoch_loss
          else:
            saver.restore(sess, tf.compat.v1.train.latest_checkpoint(temp_path))
            print(
                "INFO: Fast Finetuning epoch {}/{} restored parameters from {}"
                .format(epoch + 1, fast_ft_epochs, checkpoint_path))
            break

        if (index + 1 == len(refine_modules)):
          frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
              sess, sess.graph_def, output_names)

          #save_pb_file(frozen_graph_def, os.path.join(q_config.output_dir,
          #                  "decent_debug/fast_finetune-modules{}_{}.pb"
          #                  .format(len(refine_modules), index+1)))

          print("INFO: Fast Finetune Done.")
          return frozen_graph_def

  print("INFO: Fast Finetune None.")
  return quant_graph_def
