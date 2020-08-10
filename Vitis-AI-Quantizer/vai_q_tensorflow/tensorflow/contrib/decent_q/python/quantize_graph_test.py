"""Tests the quantize_graph.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.protobuf.config_pb2 import ConfigProto
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
from tensorflow.contrib.decent_q.python import quantize_graph as q_lib
from tensorflow.contrib.decent_q.python.ops import fix_neuron_ops


class QuantizeGraphTest(test_util.TensorFlowTestCase):
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
    q_config = q_lib.QuantizeConfig(input_nodes=input_nodes,
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


  def _build_graph(self, is_freezed=True, with_bias=True, with_fix_op=False):
    images = array_ops.placeholder(dtypes.float32, shape=[None, 4, 4, 3],
            name="input")
    if with_fix_op:
      images = fix_neuron_ops.fix_neuron(images, 1, 1, name="input/aquant")
    if is_freezed:
      w = constant_op.constant([[[[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]]]],
                                dtype=dtypes.float32,
                                shape=[1, 3, 3, 3],
                                name="w")
    else:
      w = variables.VariableV1([[[[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]],
                                [[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]]]],
                                dtype=dtypes.float32,
                                shape=[1, 3, 3, 3],
                                name="w")
    feature = nn.conv2d(images, filter=w, name="conv2d",
            strides=[1, 1, 1, 1], padding="SAME")
    if with_bias:
      if is_freezed:
        b = constant_op.constant([0.1, 0.2, 0.3], dtype=dtypes.float32, shape=[3], name="b")
      else:
        b = variables.VariableV1([0.1, 0.2, 0.3], dtype=dtypes.float32, shape=[3], name="b")
      feature = nn.bias_add(feature, b, name="bias")
    feature = nn.relu(feature, name="relu")
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


  def testQuantizeConfig(self):
    q_config, _ = self._compose_config()
    config_str = q_config.to_string()
    expected_str = 'input_nodes,input,output_nodes,relu,input_shapes,'\
            '-1*4*4*3,weight_bit,8,activation_bit,8,method,0,calib_iter,'\
            '10,output_dir,{},align_concat,0,simulate_dpu,1,'.format(self.get_temp_dir())
    self.assertEqual(config_str, expected_str.encode())

  def testConvertConstantsToVariables(self):
    q_config, _ = self._compose_config()
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      input_graph_def = ops.get_default_graph().as_graph_def()
    output_graph_def = q_lib.ConvertConstantsToVariables(input_graph_def, q_config)
    node_names = ["input", "w", "w/read", "conv2d", "b", "b/read", "bias", "relu"]
    var_names = ["w", "b"]
    for node in output_graph_def.node:
      self.assertIn(node.name, node_names)
      node_names.remove(node.name)
      if node.name in var_names:
        self.assertIn(node.op, "VariableV2")
    self.assertEqual(node_names, [])

  def testCreateOptimizedGraphDefo(self):
    q_config, _ = self._compose_config()
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      bias = ops.get_default_graph().get_tensor_by_name("bias:0")
      bias_idt = array_ops.identity(bias, name="bias_idt")
      input_graph_def = ops.get_default_graph().as_graph_def()
    output_graph_def = q_lib.CreateOptimizedGraphDef(input_graph_def, q_config)
    node_names = ["input", "w", "conv2d", "b", "bias", "relu"]
    for node in output_graph_def.node:
      self.assertIn(node.name, node_names)
      node_names.remove(node.name)
    self.assertEqual(node_names, [])

  def testRerouteTensor(self):
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      conv_t = ops.get_default_graph().get_tensor_by_name("conv2d:0")
      bias_t = ops.get_default_graph().get_tensor_by_name("bias:0")
      q_lib.RerouteTensor(conv_t, bias_t)

      for op in ops.get_default_graph().get_operations():
        if op.name == "relu":
          for in_t in op.inputs:
            self.assertEqual(in_t.name, "conv2d:0")

  def testAppendNode(self):
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      bias = ops.get_default_graph().get_operation_by_name("relu")
      consumers = {}
      consumers[bias] = 0
      id_node = node_def_pb2.NodeDef()
      id_node.name = "test_id"
      id_node.op = "Identity"
      id_node.input.append("bias")
      q_lib.AppendNode(id_node, consumers, existed=False)
      for op in ops.get_default_graph().get_operations():
        if op.name == "test_id":
          self.assertEqual(len(op.inputs), 1)
          self.assertEqual(op.inputs[0].name, "bias:0")
        if op.name == "relu":
          self.assertEqual(len(op.inputs), 1)
          self.assertEqual(op.inputs[0].name, "test_id:0")

  def testGetNodeConsumers(self):
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      dst_graph_def = ops.get_default_graph().as_graph_def()
      bias_node = [node for node in dst_graph_def.node if node.name == "bias"][0]
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True, with_bias=False)
      consumers = q_lib.GetNodeConsumers(bias_node, dst_graph_def)
      exprct_comsumers = {ops.get_default_graph().get_operation_by_name("relu"):0}
    self.assertEqual(consumers, exprct_comsumers)

  def testMergeNodesFromGraphDef(self):
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True)
      dst_graph_def = ops.get_default_graph().as_graph_def()
    with ops.Graph().as_default():
      self._build_graph(is_freezed=True, with_bias=False)
      q_lib.MergeNodesFromGraphDef(dst_graph_def)

      bias = ops.get_default_graph().get_operation_by_name("bias")
      self.assertEqual(len(bias.inputs), 2)
      self.assertEqual(bias.inputs[0].name, "conv2d:0")
      self.assertEqual(bias.inputs[1].name, "b:0")
      relu = ops.get_default_graph().get_operation_by_name("relu")
      self.assertEqual(relu.inputs[0].name, "bias:0")


  def testCreateQuantizeTrainingGraph(self):
    with ops.Graph().as_default():
      self._build_graph(is_freezed=False)
      original_op_names = [op.name for op in ops.get_default_graph().get_operations()]
      q_config, _ = self._compose_config()
      q_lib.CreateQuantizeTrainingGraph(config=q_config)
      quantized_ops = [op for op in ops.get_default_graph().get_operations()]
      diff_ops = [op for op in quantized_ops if op.name not in original_op_names]
      for op in diff_ops:
        self.assertEndsWith(op.name, "quant")
        self.assertEqual(op.type, "FixNeuron")

  def testCreateQuantizeEvaluationGraph(self):
    q_config, _ = self._compose_config()
    with ops.Graph().as_default():
      self._build_graph(is_freezed=False)
      original_op_names = [op.name for op in ops.get_default_graph().get_operations()]
      q_lib.CreateQuantizeTrainingGraph(config=q_config)
      with session.Session() as sess:
        relu = sess.graph.get_tensor_by_name("relu/aquant:0")
        input_fn = self._mock_input_fn("input:0", [1, 4, 4, 3])
        init = variables.global_variables_initializer()
        sess.run(init)
        relu_val = sess.run([relu], feed_dict=input_fn(1))

    with ops.Graph().as_default():
      self._build_graph(is_freezed=False)
      original_op_names = [op.name for op in ops.get_default_graph().get_operations()]
      q_lib.CreateQuantizeEvaluationGraph(config=q_config)
      true_pos = {"b/read/wquant":8, "w/read/wquant":7,
                  "input/aquant":6, "relu/aquant":4}
      for op in ops.get_default_graph().get_operations():
        if op.name not in original_op_names + ["init"]:
          self.assertEqual(op.type, "FixNeuron")
          self.assertEqual(op.get_attr("quantize_pos"), true_pos[op.name])

  def testCreateQuantizeDeployGraph(self):
    q_config, _ = self._compose_config()
    with ops.Graph().as_default():
      self._build_graph(is_freezed=False)
      original_op_names = [op.name for op in ops.get_default_graph().get_operations()]
      q_lib.CreateQuantizeTrainingGraph(config=q_config)
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
        new_saver = saver_lib.Saver()
        checkpoint_path = new_saver.save(
            sess,
            checkpoint_prefix,
            global_step=0,
            latest_filename=checkpoint_state_name)

    with ops.Graph().as_default():
      self._build_graph(is_freezed=False)
      original_op_names = [op.name for op in ops.get_default_graph().get_operations()]
      q_lib.CreateQuantizeEvaluationGraph(config=q_config)
      q_lib.CreateQuantizeDeployGraph(checkpoint=checkpoint_path,
              config=q_config)
    deploy_graph_def = graph_pb2.GraphDef()
    deploy_graph_path = self._get_pb_path("deploy_model")
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


if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
  test.main()
