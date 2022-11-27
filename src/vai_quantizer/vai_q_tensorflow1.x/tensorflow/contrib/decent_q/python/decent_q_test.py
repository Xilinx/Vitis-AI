"""Tests the decent_q."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver as saver_lib
from tensorflow.contrib.decent_q.python import decent_q
from tensorflow.contrib.decent_q.python.quantize_graph import QuantizeConfig


class Decent_qTest(test_util.TensorFlowTestCase):
  def _get_pb_path(self, start_str, dir_name=None):
    if dir_name is None:
      dir_name = self.get_temp_dir()
    else:
      dir_name = os.path.join(self.get_temp_dir(), dir_name)

    file_list = []
    for root, dirs, files in os.walk(dir_name):
      for f in files:
        if f.startswith(start_str):
          file_list.append(os.path.join(root, f))
    return file_list[0]

  def _parse_quant_info_from_file(self, dump_file):
    with open(dump_file, 'rb') as f:
      lines = f.readlines()
      return [int(m) for m in lines]

  def _parse_def_from_file(sefl, graph_def, graph_path):
    with gfile.GFile(graph_path, "rb") as f:
      graph_def.ParseFromString(f.read())
    return graph_def

  def _compose_config(self, is_quant=False):
    input_nodes = ["input"]
    output_nodes = ["relu/aquant"] if is_quant else ["relu"]
    input_shapes = [[-1, 4, 4, 3]]
    ignore_nodes = []
    nodes_bit = []
    q_config = QuantizeConfig(input_nodes=input_nodes,
                              output_nodes=output_nodes,
                              input_shapes=input_shapes,
                              ignore_nodes=ignore_nodes,
                              weight_bit=8,
                              activation_bit=8,
                              nodes_bit=nodes_bit,
                              method=0,
                              calib_iter=10,
                              output_dir=self.get_temp_dir(),
                              align_concat=0,
                              simulate_dpu=1)
    s_config = ConfigProto()
    s_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    s_config.graph_options.optimizer_options.opt_level = -1
    s_config.graph_options.rewrite_options.disable_meta_optimizer = True
    return q_config, s_config


  def _build_graph(self, is_freezed=True):
    images = array_ops.placeholder(dtypes.float32, shape=[None, 4, 4, 3],
            name="input")
    if is_freezed:
      w = constant_op.constant([[[[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]]]],
                                dtype=dtypes.float32,
                                shape=[1, 3, 3, 3],
                                name="w/read")
    else:
      w = variables.VariableV1([[[[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]]]],
                                dtype=dtypes.float32,
                                shape=[1, 3, 3, 3],
                                name="w")
    conv = nn.conv2d(images, filter=w, name="conv2d",
            strides=[1, 1, 1, 1], padding="SAME")
    if is_freezed:
      b = constant_op.constant([0.1, 0.2, 0.3], dtype=dtypes.float32,
              shape=[3], name="b/read")
    else:
      b = variables.VariableV1([0.1, 0.2, 0.3], dtype=dtypes.float32, shape=[3], name="b")
    bias = nn.bias_add(conv, b, name="bias")
    relu = nn.relu(bias, name="relu")
    return


  def _mock_input_fn(sefl, tensor_name, shape):
    def input_fn(iter):
      np.random.seed(1024)
      feed_dict = {}
      tensor = []
      # feed_dict[tensor_name] = np.random.random(shape)
      feed_dict[tensor_name] = np.ones(shape, dtype=np.float32)
      return feed_dict
    return input_fn

  def testQuantizeFrozen(self):
    input_graph_name = "original_float_graph.pb"
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      graph_def = ops.get_default_graph().as_graph_def()
      input_graph_path = os.path.join(self.get_temp_dir(), input_graph_name)
      with gfile.GFile(input_graph_path, "wb") as f:
        f.write(graph_def.SerializeToString())
    input_graph_def = graph_pb2.GraphDef()
    input_graph_def = self._parse_def_from_file(input_graph_def, input_graph_path)
    original_graph_node = [node.name for node in input_graph_def.node]

    q_config, s_config = self._compose_config()
    input_fn = self._mock_input_fn("input", [1, 4, 4, 3])
    decent_q.quantize_frozen(input_graph_def, input_fn, q_config, s_config)

    quant_graph_def = graph_pb2.GraphDef()
    quant_graph_path = self._get_pb_path("quantize_eval_model")
    quant_graph_def = self._parse_def_from_file(quant_graph_def, quant_graph_path)
    for node in quant_graph_def.node:
      if node.name not in original_graph_node:
        self.assertIn(node.name, ["input/aquant", "relu/aquant",
            "b/read/wquant", "w/read/wquant"])
        self.assertEqual(node.op, "FixNeuron")
        pos = node.attr["quantize_pos"].i
        if node.name == "input/aquant":
          self.assertEqual(6, pos)
        if node.name == "relu/aquant":
          self.assertEqual(4, pos)
        if node.name == "b/read/wquant":
          self.assertEqual(8, pos)
        if node.name == "w/read/wquant":
          self.assertEqual(7, pos)

    deploy_graph_def = graph_pb2.GraphDef()
    deploy_graph_path = self._get_pb_path("deploy_model")
    deploy_graph_def = self._parse_def_from_file(deploy_graph_def, deploy_graph_path)
    for node in deploy_graph_def.node:
      self.assertNotEqual("FixNeuron", node.op)

  def testQuantizeTrain(self):
    input_meta_name = "input_meta.meta"
    with ops.Graph().as_default():
      self._build_graph(is_freezed=False)
      graph_def = ops.get_default_graph().as_graph_def()
      input_meta_path = os.path.join(self.get_temp_dir(), input_meta_name)
      saver_lib.export_meta_graph(filename=input_meta_path)
      original_graph_node = [node.name for node in graph_def.node]

    meta_graph_def = MetaGraphDef()
    meta_graph_def = self._parse_def_from_file(meta_graph_def, input_meta_path)
    q_config, _ = self._compose_config()
    decent_q.quantize_train(meta_graph_def, q_config)

    output_meta_graph_def = MetaGraphDef()
    output_meta_graph_path = os.path.join(self.get_temp_dir(),
            "quantize_train/quantize_train.ckpt.meta")
    output_meta_graph_def = self._parse_def_from_file(output_meta_graph_def, output_meta_graph_path)
    quantize_train_graph_def = output_meta_graph_def.graph_def
    for node in quantize_train_graph_def.node:
      if node.name not in original_graph_node:
        self.assertEqual(node.op, "FixNeuron")

  def testQuantizeEval(self):
    input_meta_name = "original_meta.meta"
    input_meta_path = os.path.join(self.get_temp_dir(), input_meta_name)
    with ops.Graph().as_default():
      self._build_graph(is_freezed=False)
      graph_def = ops.get_default_graph().as_graph_def()
      saver_lib.export_meta_graph(filename=input_meta_path)

    original_meta_graph_def = MetaGraphDef()
    original_meta_graph_def = self._parse_def_from_file(original_meta_graph_def, input_meta_path)
    q_config, _ = self._compose_config()
    decent_q.quantize_train(original_meta_graph_def, q_config)

    quant_train_meta_graph_def = MetaGraphDef()
    quant_train_meta_graph_path = os.path.join(self.get_temp_dir(),
            "quantize_train/quantize_train.ckpt.meta")
    quant_train_meta_graph_def = self._parse_def_from_file(quant_train_meta_graph_def,
            quant_train_meta_graph_path)
    with session.Session() as sess:
      new_saver = saver_lib.import_meta_graph(quant_train_meta_graph_def)

      relu = sess.graph.get_tensor_by_name("relu/aquant:0")
      input_fn = self._mock_input_fn("input:0", [1, 4, 4, 3])
      init = variables.global_variables_initializer()
      sess.run(init)
      relu_val = sess.run([relu], feed_dict=input_fn(1))
    decent_q.quantize_evaluate(quant_train_meta_graph_def , q_config)
    quant_eval_meta_graph_def = MetaGraphDef()
    quant_eval_meta_graph_path = os.path.join(self.get_temp_dir(),
            "quantize_eval/quantize_eval.ckpt.meta")
    quant_eval_meta_graph_def = self._parse_def_from_file(quant_eval_meta_graph_def,
            quant_eval_meta_graph_path)
    eval_quant_pos = [node.attr["quantize_pos"].i for node in
            quant_eval_meta_graph_def.graph_def.node if node.op == "FixNeuron"]
    self.assertAllEqual([8, 7, 6, 4], eval_quant_pos)


  def testDeployCheckpoint(self):
    input_meta_name = "original_meta.meta"
    input_meta_path = os.path.join(self.get_temp_dir(), input_meta_name)
    q_config, _ = self._compose_config()
    with ops.Graph().as_default():
      self._build_graph(is_freezed=False)
      graph_def = ops.get_default_graph().as_graph_def()
      saver_lib.export_meta_graph(filename=input_meta_path)

    original_meta_graph_def = MetaGraphDef()
    original_meta_graph_def = self._parse_def_from_file(original_meta_graph_def, input_meta_path)
    decent_q.quantize_train(original_meta_graph_def, q_config)

    quant_train_meta_graph_def = MetaGraphDef()
    quant_train_meta_graph_path = os.path.join(self.get_temp_dir(),
            "quantize_train/quantize_train.ckpt.meta")
    quant_train_meta_graph_def = self._parse_def_from_file(quant_train_meta_graph_def,
            quant_train_meta_graph_path)
    with ops.Graph().as_default():
      new_saver = saver_lib.import_meta_graph(quant_train_meta_graph_def)
      with session.Session() as sess:
        w_t = sess.graph.get_tensor_by_name("w/read/wquant:0")
        b_t = sess.graph.get_tensor_by_name("b/read/wquant:0")
        relu_t = sess.graph.get_tensor_by_name("relu/aquant:0")
        input_fn = self._mock_input_fn("input:0", [1, 4, 4, 3])
        init = variables.global_variables_initializer()
        sess.run(init)
        eval_relu, eval_w, eval_b= sess.run([relu_t, w_t, b_t], feed_dict=input_fn(1))

        checkpoint_prefix = os.path.join(self.get_temp_dir(), "ckpt/saved_checkpoint")
        checkpoint_state_name = "checkpoint_state"
        checkpoint_path = new_saver.save(
            sess,
            checkpoint_prefix,
            global_step=0,
            latest_filename=checkpoint_state_name)
    q_config.output_nodes = ["relu/aquant"]
    decent_q.quantize_evaluate(quant_train_meta_graph_def , q_config)
    quant_eval_meta_graph_def = MetaGraphDef()
    quant_eval_meta_graph_path = os.path.join(self.get_temp_dir(),
            "quantize_eval/quantize_eval.ckpt.meta")
    quant_eval_meta_graph_def = self._parse_def_from_file(quant_eval_meta_graph_def,
          quant_eval_meta_graph_path)
    sess.close()
    decent_q.deploy_checkpoint(quant_eval_meta_graph_def, checkpoint_path, q_config)
    deploy_graph_def = graph_pb2.GraphDef()
    deploy_graph_path = os.path.join(self.get_temp_dir(),
            "deploy/deploy_model.pb")
    deploy_graph_def = self._parse_def_from_file(deploy_graph_def,
            deploy_graph_path)
    for node in deploy_graph_def.node:
      if node.name == "conv2d":
        # need to equal with quantize pos in quantize_eval_model.pb
        self.assertAllEqual(node.attr['ipos'].list.i, [8, 6])
        self.assertAllEqual(node.attr['wpos'].list.i, [8, 7])
        self.assertAllEqual(node.attr['bpos'].list.i, [8, 8])
        self.assertAllEqual(node.attr['opos'].list.i, [8, 4])
        deploy_w = tensor_util.MakeNdarray(node.attr['weights'].tensor)
        deploy_b = tensor_util.MakeNdarray(node.attr['bias'].tensor)
        self.assertNDArrayNear(deploy_w, eval_w, 1e-6)
        self.assertNDArrayNear(deploy_b, eval_b, 1e-6)

  def testInspect(self):
    input_graph_name = "original_float_graph.pb"
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      graph_def = ops.get_default_graph().as_graph_def()
      input_graph_path = os.path.join(self.get_temp_dir(), input_graph_name)
      with gfile.GFile(input_graph_path, "wb") as f:
        f.write(graph_def.SerializeToString())
    input_graph_def = graph_pb2.GraphDef()
    input_graph_def = self._parse_def_from_file(input_graph_def, input_graph_path)
    frozen_graph_path = os.path.join(self.get_temp_dir(),
            "quantize_eval_model.pb")
    decent_q.inspect(input_graph_def, frozen_graph_path)

  def testDump(self):
    input_graph_name = "original_float_graph.pb"
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      graph_def = ops.get_default_graph().as_graph_def()
      input_graph_path = os.path.join(self.get_temp_dir(), input_graph_name)
      with gfile.GFile(input_graph_path, "wb") as f:
        f.write(graph_def.SerializeToString())
    input_graph_def = graph_pb2.GraphDef()
    input_graph_def = self._parse_def_from_file(input_graph_def, input_graph_path)
    original_graph_node = [node.name for node in input_graph_def.node]

    q_config, s_config = self._compose_config()
    input_fn = self._mock_input_fn("input", [1, 4, 4, 3])
    decent_q.quantize_frozen(input_graph_def, input_fn, q_config, s_config)

    quant_graph_def = graph_pb2.GraphDef()
    quant_graph_path = os.path.join(self.get_temp_dir(),
            "quantize_eval_model.pb")
    quant_graph_def = self._parse_def_from_file(quant_graph_def, quant_graph_path)

    max_dump_batches = 1
    dump_float = 0
    decent_q.dump(quant_graph_def, input_fn, self.get_temp_dir(), max_dump_batches,
         dump_float, s_config)
    input_aquant_path = os.path.join(self.get_temp_dir(),
            "dump_results_0/input_aquant.txt")
    input_aquant = self._parse_quant_info_from_file(input_aquant_path)
    self.assertAllEqual([64]*48, input_aquant)

    relu_aquant_path = os.path.join(self.get_temp_dir(),
            "dump_results_0/relu_aquant.txt")
    relu_aquant = self._parse_quant_info_from_file(relu_aquant_path)
    self.assertAllEqual(relu_aquant, [40, 51, 62, 59, 75, 91, 59, 75,
                                      91, 40, 51, 62, 40, 51, 62, 59,
                                      75, 91, 59, 75, 91, 40, 51, 62,
                                      40, 51, 62, 59, 75, 91, 59, 75,
                                      91, 40, 51, 62, 40, 51, 62, 59,
                                      75, 91, 59, 75, 91, 40, 51, 62])


if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
  test.main()
