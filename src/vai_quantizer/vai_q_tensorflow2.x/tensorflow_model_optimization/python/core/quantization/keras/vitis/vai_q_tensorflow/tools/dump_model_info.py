import os, sys
import copy

import logging

# import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
# from tensorflow.core.framework.tensor_pb2 import TensorProto
# from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto


"""
remove nodes before and include `target_node` then set input of graph as a
placeholder node
"""

tf.app.flags.DEFINE_string('input_pb', '', 'input pb file')
tf.app.flags.DEFINE_string('model_info', './model_info.txt', 'save model info')
tf.app.flags.DEFINE_boolean('debug', False, 'if log debug info')
FLAGS = tf.app.flags.FLAGS


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CONV_TYPE = {"Conv2D","DepthwiseConv2dNative"}
BIASADD_TYPE = {"Add", "BiasAdd", "AddV2"}
ACTIVATION_TYPE = {"Relu", "Relu6"}
IGNORE_TYPE = {"FixNeuron", "Identity", "Placeholder"}
WEIGHT_TYPE = {"Const"}

INDENT = "    "

def _get_real_node_name(name):
  return name.split(":")[0]

def _parse_input_graph(graph_path):
  with gfile.FastGFile(graph_path,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

def _get_name2node_map(graph_def):
  name_to_node = {}
  for node in graph_def.node:
    name_to_node[node.name] = node
  return name_to_node

def _get_name2output_map(graph_def, name_to_node):
  name_to_output_node = {}
  for node in graph_def.node:
    if len(node.input) > 0:
      for n in node.input:
        in_name = _get_real_node_name(n)
        output_nodes = name_to_output_node.get(in_name, [])
        output_nodes.append(node)
        name_to_output_node[in_name] = output_nodes
  return name_to_output_node

def get_node_attr(node, indent):
  def _get_attr_val(attr_val):
    if attr_val.b:
      return attr_val.b
    if attr_val.f:
      return attr_val.f
    if attr_val.i:
      return attr_val.i
    if attr_val.s:
      return attr_val.s
    if attr_val.type:
      return attr_val.type
    if hasattr(attr_val, "list") and attr_val.list:
      return _get_attr_val(attr_val.list)
    return None
  attr_info = ""
  for k, v in node.attr.items():
    # print(k, v)
    logging.debug(k, v)
    attr_info += "{} {}:{}\n".format(indent, k, _get_attr_val(v))
  return attr_info


def get_node_pos(node):
  return node.attr["quantize_pos"].i

def write_info(info, path):
  with open(path, 'w') as f:
    f.writelines(info)

def conv_add_relu(node, visited_node, name_to_node, name_to_output_node):
  if node.op not in CONV_TYPE:
    # logging.warning(str(sys._getframe().f_lineno) + " node {} op type is {}".format(node.name, node.op))
    return None

  conv_node = node
  is_match = False
  ## match pattern conv + bias + relu
  out_level_1 = name_to_output_node[conv_node.name]
  if len(out_level_1) > 1:
    logging.warning(sys._getframe().f_lineno + " node {} has {} outputs, is more than one ".format(conv_node.name, len(out_level_1)))
  biasadd = out_level_1[0]
  if biasadd.op in BIASADD_TYPE:
    out_level_2 = name_to_output_node[biasadd.name]
    if len(out_level_2) > 1:
      logging.warning(str(sys._getframe().f_lineno) + " node {} has {} outputs, is more than one ".format(biasadd.name, len(out_level_2)))
    act_node = out_level_2[0]
    if act_node.op in ACTIVATION_TYPE:
      is_match = True

  if not is_match:
    return None
  in_quant = name_to_node[_get_real_node_name(conv_node.input[0])]
  w_quant = name_to_node[_get_real_node_name(conv_node.input[1])]
  w = name_to_node[_get_real_node_name(w_quant.input[0])]
  b_quant = name_to_node[_get_real_node_name(biasadd.input[1])]
  b = name_to_node[_get_real_node_name(b_quant.input[0])]
  out_quant = name_to_output_node[_get_real_node_name(act_node.name)][0]
  visited_node |= {conv_node.name, biasadd.name, act_node.name, out_quant.name,
          w_quant.name, b_quant.name, w.name, b.name}

  pattern_info = []
  node_names = "pattern [Conv + bias + relu]: ({})\n".format(" + ".join([conv_node.name, biasadd.name, act_node.name]))
  input_nodes = INDENT + "input nodes: {} \n".format(", ".join(conv_node.input))

  output_node_names = [node.name for node in name_to_output_node[out_quant.name]]
  output_node_names = INDENT + "output nodes: {} \n".format(", ".join(output_node_names))

  in_pos = get_node_pos(in_quant)
  w_pos = get_node_pos(w_quant)
  b_pos = get_node_pos(b_quant)
  out_pos = get_node_pos(out_quant)
  pos_info = INDENT + "inpos: {}    w_pos: {}    b_pos: {}    out_pos:{}\n".format(in_pos,
          w_pos, b_pos, out_pos)

  conv_attr = INDENT + "attr: \n"
  # conv_attr = conv_attr + get_node_attr(conv_node, INDENT*2)

  infos = [node_names, input_nodes, output_node_names, pos_info, conv_attr]
  for info in infos:
    pattern_info.append(info)
  return pattern_info

def conv_add(node, visited_node, name_to_node, name_to_output_node):
  if node.op not in CONV_TYPE:
    return None

  conv_node = node
  is_match = False
  ## match pattern conv + bias + relu
  out_level_1 = name_to_output_node[conv_node.name]
  if len(out_level_1) > 1:
    logging.warning(sys._getframe().f_lineno + " node {} has {} outputs, is more than one ".format(conv_node.name, len(out_level_1)))
  biasadd = out_level_1[0]
  if biasadd.op in BIASADD_TYPE:
    is_match = True

  if not is_match:
    return None
  in_quant = name_to_node[_get_real_node_name(conv_node.input[0])]
  w_quant = name_to_node[_get_real_node_name(conv_node.input[1])]
  w = name_to_node[_get_real_node_name(w_quant.input[0])]
  b_quant = name_to_node[_get_real_node_name(biasadd.input[1])]
  b = name_to_node[_get_real_node_name(b_quant.input[0])]
  out_quant = name_to_output_node[_get_real_node_name(biasadd.name)][0]
  visited_node |= {conv_node.name, biasadd.name, out_quant.name, w_quant.name, b_quant.name, w.name, b.name}

  pattern_info = []
  node_names = "pattern [Conv + bias ]: ({})\n".format(" + ".join([conv_node.name, biasadd.name]))
  input_nodes = INDENT + "input nodes: {} \n".format(", ".join(conv_node.input))

  output_node_names = ""
  if out_quant.name in name_to_output_node:
    output_node_names = [node.name for node in name_to_output_node[out_quant.name]]
    output_node_names = INDENT + "output nodes: {} \n".format(", ".join(output_node_names))

  in_pos = get_node_pos(in_quant)
  w_pos = get_node_pos(w_quant)
  b_pos = get_node_pos(b_quant)
  out_pos = get_node_pos(out_quant)
  pos_info = INDENT + "inpos: {}    w_pos: {}    b_pos: {}    out_pos:{}\n".format(in_pos,
          w_pos, b_pos, out_pos)

  conv_attr = INDENT + "attr: \n"
  # conv_attr = conv_attr + get_node_attr(conv_node, INDENT*2)

  infos = [node_names, input_nodes, output_node_names, pos_info, conv_attr]
  for info in infos:
    pattern_info.append(info)
  return pattern_info

def conv_relu(node, visited_node, name_to_node, name_to_output_node):
  ## TODO
  return None

def regular_node(node, visited_node, name_to_node, name_to_output_node):
  node_attr = INDENT + "attr: \n"
  # node_attr = node_attr + get_node_attr(node, INDENT*2)

  node_names = "op type {}: ({})\n".format(node.op, node.name)

  input_nodes = INDENT + "input nodes: {} \n".format(", ".join(node.input))

  output_node_names = [n.name for n in name_to_output_node[node.name]]
  output_node_names = INDENT + "output nodes: {} \n".format(", ".join(output_node_names))

  pattern_info = []
  infos = [node_names, input_nodes, output_node_names, node_attr]
  for info in infos:
    pattern_info.append(info)
  visited_node |= {node.name}
  return pattern_info

def main():
  if FLAGS.debug:
    logging.info("using logging level DEBUG")
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
  else:
    logging.info("using logging level INFO")
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

  logging.info("Parsing file :{}".format(FLAGS.input_pb))
  graph_def = _parse_input_graph(FLAGS.input_pb)
  name_to_node = _get_name2node_map(graph_def)
  name_to_output_node = _get_name2output_map(graph_def, name_to_node)

  visited_node = set()
  ignore_type = ACTIVATION_TYPE | WEIGHT_TYPE | IGNORE_TYPE
  model_info = []
  for node in graph_def.node:
    if node.name in visited_node:
      continue
    if node.op in ignore_type:
      continue
    logging.debug("processing node {}".format(node.name))
    parse_function = [conv_add_relu, conv_add, conv_relu, regular_node]
    for func in parse_function:
      pattern_info = func(node, visited_node, name_to_node, name_to_output_node)
      if pattern_info:
        model_info.extend(pattern_info)
        break
    # print(len(visited_node))
  for node in graph_def.node:
    if node.name not in visited_node and node.op not in IGNORE_TYPE:
      # logging.info("node {} is not processed".format(node.name))
      print("node {} is not processed".format(node.name))
  write_info(model_info, FLAGS.model_info)



if __name__ == '__main__':
  main()
