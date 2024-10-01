import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_name = "uv_model"
######################################
if model_name == "y_model":
  path = "./y_mode_savedmodel"; filename = "y_model.pb"; shape_1 = [None, 192, 320, 1]; shape_2 = [None, 192, 320, 2]
######################################
elif model_name == "uv_model":
  path = "./uv_model"; filename = "uv_model.pb"; shape_1 = [None, 192, 320, 1]; shape_2 = [None, 192, 320, 2];
######################################

## need enable eager model
tf.compat.v1.enable_eager_execution()
loaded_1 = tf.saved_model.load_v2(path)
infer = loaded_1.signatures["serving_default"]

# find inputs placeholder
print(infer.inputs)
print(infer.outputs)

f = tf.function(infer)
import pdb; pdb.set_trace()
## input_38 and input_39 need parse from above
if model_name == "y_model":
  f2 = f.get_concrete_function(
          input_38=tf.TensorSpec(shape=shape_1, dtype=tf.float32),
          input_39=tf.TensorSpec(shape=shape_2, dtype=tf.float32))
elif model_name == "uv_model":
  f2 = f.get_concrete_function(
          input_4=tf.TensorSpec(shape=shape_1, dtype=tf.float32),
          input_5=tf.TensorSpec(shape=shape_2, dtype=tf.float32))


f3 = convert_variables_to_constants_v2(f2)
graph_def = f3.graph.as_graph_def()


# remove NoOp
dst_graph = []
no_op_names = []
for n in graph_def.node:
  if n.op == "NoOp":
    print(n.name, n.op)
    no_op_names.append(n.name)
    continue
print()
print()
for n in graph_def.node:
  if n.name in no_op_names:
      continue
  for inp in n.input:
    for no_op in no_op_names:
      if no_op in inp:
        n.input.remove(inp)
        print(n.name, n.op, n.input)
  dst_graph.append(n)
print()
print()
del graph_def.node[:]
for n in dst_graph:
  graph_def.node.append(n)

# Export frozen graph
with tf.io.gfile.GFile(filename, 'wb') as f:
       f.write(graph_def.SerializeToString())
