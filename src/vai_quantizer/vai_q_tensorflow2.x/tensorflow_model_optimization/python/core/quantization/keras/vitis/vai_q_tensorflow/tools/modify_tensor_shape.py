import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto

os.environ["CUDA_VISIBLE_DEVICES"]="1"

src_graph_pb_path = "../../pruned_resnet_v1.5/quantize_results/quantize_eval_model.pb"
dst_graph_pb_path = './quantize_eval_model.pb'

with gfile.FastGFile(src_graph_pb_path,'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  for node in graph_def.node:
    print(node.name)
    if node.name == "resnet_model/Mean/scale_value":
      tensor_content = node.attr["value"].tensor.tensor_content
      tensor_shape = [x.size for x in node.attr["value"].tensor.tensor_shape.dim]
      import pdb; pdb.set_trace()
      tensor_shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=s) for s
          in tensor_shape])
      tensor_dtype = node.attr["value"].tensor.dtype

with tf.Session() as sess:
  print("load graph")
  with gfile.FastGFile(dst_graph_pb_path,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    for node in graph_def.node:
      print(node.name)
      if node.name == "resnet_model/Mean/scale_value":
        # my_value = np.array([1.0048828125], dtype=np.float)
        # Make graph node
        # tensor_content = my_value.tobytes()
        # dt = tf.as_dtype(my_value.dtype).as_datatype_enum
        # tensor_shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=s) for s
        #     in my_value.shape])
        tensor_proto = TensorProto(tensor_content=tensor_content,
                                  tensor_shape=tensor_shape,
                                  dtype=tensor_dtype)
        node.attr["value"].tensor.Clear()
        node.attr["value"].tensor.CopyFrom(tensor_proto)
    tf.io.write_graph(graph_def, "./debug", "quantize_eval_model.pb", as_text=False)
