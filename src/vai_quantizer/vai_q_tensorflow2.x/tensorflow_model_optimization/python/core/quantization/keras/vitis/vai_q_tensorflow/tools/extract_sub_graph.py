import tensorflow as tf
tf1 = tf.compat.v1

tf1.app.flags.DEFINE_string('model_name',
        '', 'TensorFlow \'GraphDef\' file to load.')
FLAGS = tf1.app.flags.FLAGS

org_model_dir = "./models/pb_models/"
org_model_name = FLAGS.model_name + ".pb"

dst_model_dir = "./models/extracted_sub_graph"
dst_model_name = org_model_name

def load_graph_def(graph_def_path):
  with tf1.gfile.FastGFile(graph_def_path,'rb') as f:
    graph_def = tf1.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

org_model_path = org_model_dir + org_model_name
graph_def = load_graph_def(org_model_path)

# graph_def = tf1.graph_util.extract_sub_graph(graph_def, ["Identity"])
id_node = graph_def.node[-1]
id_node.input.remove("^NoOp")
softmax_node = id_node.input[0]
# import pdb; pdb.set_trace()
graph_def = tf1.graph_util.extract_sub_graph(graph_def, [softmax_node])
graph_def.node.extend([id_node])

for node in graph_def.node:
  print(node.name, node.op)
  for inp in node.input:
    print("    ", inp)
tf.io.write_graph(graph_def, dst_model_dir, dst_model_name, as_text=False)
