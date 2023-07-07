import tensorflow as tf
import copy

def rename_node(graph_def, src_name, target_name):
  modified_graph_def = tf.GraphDef()
  for node in graph_def.node:
    new_node = copy.deepcopy(node)
    if node.name == src_name:
      print("rename node {} to {} ".format(src_name, target_name))
      new_node.name = target_name
      modified_graph_def.node.extend([new_node])
      continue
    for i, in_name in enumerate(node.input):
      if node.input[i] == src_name:
        print("rename node's input", node.name)
        new_node.input[i] = target_name
      else:
        new_node.input[i] = in_name
    modified_graph_def.node.extend([new_node])
  return modified_graph_def

