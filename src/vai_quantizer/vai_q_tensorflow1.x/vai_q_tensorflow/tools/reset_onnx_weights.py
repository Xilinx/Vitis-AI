import onnx
from onnx import numpy_helper

import numpy as np

model_name = "mlperf_resnet50"
# model_name = "sd"
model_path = "./{}/quantized.onnx".format(model_name)
save_path = "./{}_reset_weights.onnx".format(model_name)

model = onnx.load(model_path)
graph = model.graph

inits = graph.initializer
init_map = {}
for w in inits:
  init_map[w.name] = w

nodes = graph.node
node_map = {}
for node in nodes:
  node_map[node.name] = node
# w_0 = inits[0]
# node_0 = graph.node[0]

def get_real_node_name(name):
  return name.split(":")[0]

def get_input_node(node):
  input_node_name = node.input[0]
  input_node_name = get_real_node_name(input_node_name)
  return node_map[input_node_name]

qdq_init_map = {}
for node in graph.node:
  if node.op_type == 'DequantizeLinear' :
    dq_node = node
    tensor_scale = init_map[dq_node.input[1]]
    q_node = get_input_node(dq_node)
    if q_node.input[0] in init_map:
      tensor_w = init_map[q_node.input[0]]
      np_w = numpy_helper.to_array(tensor_w)
      np_scale = numpy_helper.to_array(tensor_scale)
      q_w = (np_w / np_scale).round()
      print(q_w.min(), q_w.max())
      q_w = np.clip(q_w, -128, 127)
      print(q_w.min(), q_w.max())
      print()
      qdq_w = q_w * np_scale
      tensor_w_qdq = numpy_helper.from_array(qdq_w, tensor_w.name)
      qdq_init_map[tensor_w.name] = tensor_w_qdq
      # print(qdq_w)
      # import pdb; pdb.set_trace()

for init in graph.initializer:
  if init.name in qdq_init_map:
    # print("reset weights: ", init.name)
    init.CopyFrom(qdq_init_map[init.name])

# import pdb; pdb.set_trace()
onnx.save_model(model, save_path)
