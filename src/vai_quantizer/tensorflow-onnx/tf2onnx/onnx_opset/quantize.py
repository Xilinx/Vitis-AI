# SPDX-License-Identifier: Apache-2.0


"""
tensor
"""

import logging

import numpy as np
from onnx.onnx_pb import TensorProto

from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.utils import make_sure

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement,invalid-name

@tf_op(["FakeQuantWithMinMaxVarsPerChannel"])
class FakeQuantWithMinMaxVarsPerChannel:
    # see https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/fake-quant-with-min-max-vars-per-channel
    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        amin = np.array(node.inputs[1].get_tensor_value())
        amax = np.array(node.inputs[2].get_tensor_value())
        make_sure(
            len(amin) == len(amax),
            "min and max should have same dim.")

        narrow_range = node.get_attr("narrow_range").i
        num_bits = node.get_attr("num_bits").i
        make_sure(
            not narrow_range,
            "Unable to convert node FakeQuantWithMinMaxArgs with narrow_range=%r",
            narrow_range)
        make_sure(
            num_bits == 8,
            "Unable to convert node FakeQuantWithMinMaxArgs with "
            "num_bits=%r", num_bits)

        dtype = ctx.get_dtype(node.input[0])
        shape = ctx.get_shape(node.input[0])

        make_sure(
            len(shape) in [1, 2, 4],
            "Only support [d], [b, d], [b, h, w, d]")
        axis = -1
        #  idtype = TensorProto.UINT8
        idtype = TensorProto.INT8

        scale = (amax - amin) / (2 ** num_bits - 1)
        min_adj = np.around(amin / scale)

        pb_scale = ctx.make_const(
            utils.make_name("{}_scaley".format(node.name)),
            np.array(scale, dtype=np.float32),
        )
        #  zero = np.array(-min_adj, dtype=np.uint8)
        zero = np.array([0 for i in range(len(amin))], dtype=np.int8)
        #  make_sure(
        #      zero == -min_adj,
        #      "Cannot convert %s node %s with "
        #      "min=%r max=%r numbits=%r because zero_scale=%r "
        #      "is outside uint8 boundary",
        #      node.type, node.name, amin, amax, num_bits, -min_adj)
        zero_point = ctx.make_const(
            utils.make_name("{}_zpy".format(node.name)), zero)

        new_node = ctx.make_node(
            "QuantizeLinear", [node.input[0], pb_scale.name, zero_point.name],
            op_name_scope=node.name, attr={"axis": axis},
            shapes=[shape], dtypes=[idtype])
        output_name = new_node.output[0]
        ctx.replace_input(node, node.input[0], output_name, 0)

        ctx.remove_node(node.name)

        last_node = ctx.make_node(
            "DequantizeLinear", [new_node.output[0], pb_scale.name, zero_point.name],
            op_name_scope=node.name, attr={"axis": axis},
            shapes=[shape], dtypes=[dtype])
        ctx.replace_all_inputs(node.output[0], last_node.output[0])  # ops=ctx.get_nodes()


@tf_op(["FakeQuantWithMinMaxArgs", "FakeQuantWithMinMaxVars"])
class FakeQuantWithMinMaxArgs:
    # see https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/fake-quant-with-min-max-args
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # hack to make up for the missing onnx pack op
        if node.type == "FakeQuantWithMinMaxVars":
            utils.make_sure(node.inputs[1].is_scalar(), "%s node %s requires const scalar value for min",
                            node.type, node.name)
            utils.make_sure(node.inputs[2].is_scalar(), "%s node %s requires const scalar value for max",
                            node.type, node.name)
            amin = node.inputs[1].get_tensor_value()
            amax = node.inputs[2].get_tensor_value()
        else:
            amin = node.get_attr("min").f
            amax = node.get_attr("max").f
        narrow_range = node.get_attr("narrow_range").i
        num_bits = node.get_attr("num_bits").i

        make_sure(
            not narrow_range,
            "Unable to convert node FakeQuantWithMinMaxArgs with narrow_range=%r",
            narrow_range)
        make_sure(
            num_bits in [8,16,32],
            "Unable to convert node FakeQuantWithMinMaxArgs with "
            "num_bits=%r", num_bits)

        scale = (amax - amin) / (2 ** num_bits - 1)
        min_adj = np.around(amin / scale)

        dtype = ctx.get_dtype(node.input[0])
        shape = ctx.get_shape(node.input[0])
        axis = 1
        idtype_dict = {8:TensorProto.INT8, 16:TensorProto.INT16, 32:TensorProto.INT32}
        idtype = idtype_dict[num_bits] 

        pb_scale = ctx.make_const(
            utils.make_name("{}_scaley".format(node.name)),
            np.array(scale, dtype=np.float32))
        #  zero = np.array(-min_adj, dtype=np.uint8)

        dtype_dict = {8:np.int8, 16:np.int16, 32:np.int32}
        zero = np.array(0, dtype=dtype_dict[num_bits])
        #  make_sure(
        #      zero == -min_adj,
        #      "Cannot convert %s node %s with "
        #      "min=%r max=%r numbits=%r because zero_scale=%r "
        #      "is outside uint8 boundary",
        #      node.type, node.name, amin, amax, num_bits, -min_adj)
        zero_point = ctx.make_const(
            utils.make_name("{}_zpy".format(node.name)), zero)

        new_node = ctx.make_node(
            "QuantizeLinear", [node.input[0], pb_scale.name, zero_point.name],
            op_name_scope=node.name, attr={"axis": axis},
            shapes=[shape], dtypes=[idtype])
        output_name = new_node.output[0]
        ctx.replace_input(node, node.input[0], output_name, 0)

        ctx.remove_node(node.name)

        last_node = ctx.make_node(
            "DequantizeLinear", [new_node.output[0], pb_scale.name, zero_point.name],
            op_name_scope=node.name, attr={"axis": axis},
            shapes=[shape], dtypes=[dtype])
        ctx.replace_all_inputs(node.output[0], last_node.output[0])  # ops=ctx.get_nodes()

@tf_op(["FixNeuron"])
class DPUFixNeuron:
    # see https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/fake-quant-with-min-max-args
    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        pos = node.get_attr("quantize_pos").i

        num_bits = node.get_attr("bit_width").i

        make_sure(
            num_bits == 8,
            "Unable to convert node FakeQuantWithMinMaxArgs with "
            "num_bits=%r", num_bits)

        scale = 2 ** (-pos)

        dtype = ctx.get_dtype(node.input[0])
        shape = ctx.get_shape(node.input[0])
        axis = 1
        idtype = TensorProto.INT8

        pb_scale = ctx.make_const(
            utils.make_name("{}_scaley".format(node.name)),
            np.array(scale, dtype=np.float32))
        zero = np.array(0, dtype=np.int8)
        zero_point = ctx.make_const(
            utils.make_name("{}_zpy".format(node.name)), zero)

        new_node = ctx.make_node(
            "QuantizeLinear", [node.input[0], pb_scale.name, zero_point.name],
            op_name_scope=node.name, attr={"axis": axis},
            shapes=[shape], dtypes=[idtype])
        output_name = new_node.output[0]
        ctx.replace_input(node, node.input[0], output_name, 0)

        ctx.remove_node(node.name)

        last_node = ctx.make_node(
            "DequantizeLinear", [new_node.output[0], pb_scale.name, zero_point.name],
            op_name_scope=node.name, attr={"axis": axis},
            shapes=[shape], dtypes=[dtype])
        ctx.replace_all_inputs(node.output[0], last_node.output[0])  # ops=ctx.get_nodes()
