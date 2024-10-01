"""
 Copyright 2019 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import hashlib
import logging
import sys
import time

# from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple

import numpy as np
from tqdm import tqdm

use_xir_proto = False
try:
    import xir
except ImportError:
    use_xir_proto = True

from xnnc.core import CORE
from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import XModelNode, XModelNodeConst
from xnnc.ir.enums import RoundMode, PadMode, TargetType, Layout
from xnnc.optimizer import OptManager
from xnnc.tensor.xtensor import XTensorCompute as tc, XTensor

# from xnnc.utils.helper import Layout
from xnnc.proto.xir_pb2 import graph_proto_v2_pb2 as xir_proto


# package version
__version__ = "3.5.0"

# create logger
logger = logging.getLogger(__name__)
enable = False
if enable:
    logger.setLevel(logging.DEBUG)
    # create log folder
    log_dir = Path.home().joinpath(".xnnc_logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    # check number of log files and only keep the most recent 10 logs.
    MAXNUM = 10
    archives = sorted(log_dir.glob("*.log"), reverse=True)
    if len(archives) >= MAXNUM:
        for f in archives[MAXNUM - 1 :]:
            if f.exists():
                f.unlink()
    # create a log file
    fname = time.strftime("%Y%m%d-%H%M%S") + ".log"
    log_file = str(log_dir.joinpath(fname).resolve())
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.DEBUG)
    # create console handler and set level to DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter
    fmtr_precise = logging.Formatter(
        "%(name)s - %(lineno)d - %(levelname)s - %(message)s"
    )
    fmtr_brief = logging.Formatter("%(levelname)s - %(message)s")
    # add formatters
    fh.setFormatter(fmtr_precise)
    ch.setFormatter(fmtr_brief)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)


QuantRoundMode = ["STD_ROUND", "DPU_ROUND", "PY3_ROUND"]


class XConverter(object):
    @classmethod
    def run(
        cls,
        model_files: List[Path],
        model_type: str,
        layout: str = "NHWC",
        in_shapes: Optional[List[List[int]]] = None,
        batchsize: int = 1,
        dump_to: Optional[Path] = None,
        target: TargetType = TargetType.XIR,
    ) -> NoReturn:
        """
        Generate an xir Graph object from specified model files.

        Parameters
        ----------
        model_files : List[Path]
            list of paths to neural network models, which are prototxt and caffemodel files for Caffe, frozen pb file for TensorFlow, or pth file for PyTorch.
        model_type : str
            type of the original model: Caffe, TensorFlow, PyTorch, TensorFlow2 and NNDCT_PyTorch.
        layout : str, optional
            data format used by original model: "NCHW" or "NHWC", by default "NHWC".
        in_shapes : Optional[List[List[int]]], optional
            shape of the model inputs, by default None
        batchsize : int
            target batch size, by default 1.
        dump_to : Optional[Path], optional
            path for serializing model, by default None
        """
        assert model_type is not None, "'model_type' should not be None."
        model_t: str = model_type.lower()
        if model_t not in [
            "caffe",
            "tensorflow",
            "tensorflow2",
            "pytorch",
            "onnx",
            "nndct",
        ]:
            error = f"[ERROR] 'model_type' shoud be one of 'caffe', 'tensorflow', 'tensorflow2', 'pytorch', 'nndct_pytorch': actual: {model_t}."
            print(error)
            sys.exit(1)
        _layout = layout.lower()
        assert _layout in [
            "nhwc",
            "nchw",
        ], "'layout' should be one of 'NHWC' or 'NCHW'."
        _layout = Layout.NCHW if _layout == "nchw" else Layout.NHWC

        # convert raw model into XModel object
        for fname in model_files:
            print(f"[INFO] {model_type} model: {fname}")
        logger.info(f"start: translation of {model_type} model into xmodel")
        xmodel = CORE.make_xmodel(
            model_files, model_type, _layout, in_shapes, batchsize
        )
        assert xmodel is not None, "'xmodel' should not be None."
        logger.info(f"end: translation of {model_type} model into xmodel")

        # optimize for xir
        logger.info("start: optimize xnnc graph for xir protocol")
        OptManager.dispatch(xmodel, "xir")
        logger.info("end: optimize xnnc graph for xir protocol")

        if target is TargetType.OPENIR:
            # ! experimental
            xmodel.serialize(dump_to.resolve(), target)

        else:
            if use_xir_proto:
                # convert xnnc graph into xir graph
                logger.info("start: convert xnnc graph into xir graph")
                graph = cls.make_xir_graph_v2(xmodel, _layout)
                logger.info("end: convert xnnc graph into xir graph")

                # md5sum
                attr_value = xir_proto.MapString2String()
                for fname in model_files:
                    md5 = cls.__md5sum(fname)
                    attr_value.value[str(fname.resolve())] = str(md5)
                attr = xir_proto.AttrValue()
                attr.map_string_2_string_value.CopyFrom(attr_value)
                graph.graph_attr["files_md5sum"].CopyFrom(attr)

                print(f"[INFO] dump xmodel ...", end="\r")
                with open(str(dump_to), "wb") as f:
                    f.write(graph.SerializeToString())
                assert dump_to.exists() == True
                print(f"[INFO] dump xmodel: {str(dump_to)}")

            else:
                # convert xnnc graph into xir graph
                logger.info("start: convert xnnc graph into xir graph")
                graph = cls.make_xir_graph(xmodel, _layout)
                logger.info("end: convert xnnc graph into xir graph")

                # set md5sum for xir graph
                md5_dict = {}
                for fname in model_files:
                    md5 = cls.__md5sum(fname)
                    md5_dict[str(fname.resolve())] = str(md5)
                graph.set_attr("files_md5sum", md5_dict)

                print(f"[INFO] dump xmodel ...", end="\r")
                graph.serialize(str(dump_to))
                assert dump_to.exists() == True
                print(f"[INFO] dump xmodel: {dump_to.absolute()}")

    @classmethod
    def make_xir_graph(cls, xmodel: XModel, layout: Layout) -> "xir.Graph":
        """
        Convert an xnnc model into an xir graph

        Parameters
        ----------
        xmodel : XModel
            xnnc XModel object
        layout : Layout
            layout of the input XModel object

        Returns
        -------
        Graph
            xir Graph object
        """

        # TODO: solve the incompatibility: xnnc conv2dtranspose has an input for specifying output_shape, while xir transposed-conv2d does not support this.
        cls.__remove_conv2dtranspose_output_shape(xmodel)

        # create Graph object
        graph = xir.Graph(name=xmodel.name)
        graph.set_attr("origin", xmodel.origin)

        # save op for future look-up.
        # key: name of "xir.Op" object, value: "xir.Op" object
        op_dict: Dict[str, "xir.Op"] = {}
        # create "xir.Op" object one-by-one
        pbar = tqdm(
            xmodel.xnodes,
            desc="[INFO] generate xmodel",
            bar_format="{desc:27}:{percentage:3.0f}%|{bar}{r_bar:50}",
        )
        for xnode in pbar:
            logger.debug(f"* invoke to_{xnode.op_type}")
            if hasattr(XConverter, "to_" + xnode.op_type):

                func = getattr(XConverter, "to_" + xnode.op_type)
                assert func is not None, f"Not found function: {'_to_' + xnode.op_type}"
                op: "xir.Op" = func(xnode, graph, op_dict, layout, xmodel.origin)
                assert (
                    op is not None
                ), f"Failed to create op: op_name={xnode.op_name}, op_type={xnode.op_type}."
                # set quantization info
                for k, v in xnode.quant_in.items():
                    if v is not None:
                        op.set_attr("quant_in_" + k, v)
                for k, v in xnode.quant_out.items():
                    if v is not None:
                        op.set_attr("quant_out_" + k, v)
                # put into op_dict
                assert (
                    xnode.op_name not in op_dict
                ), f"Duplicated op: type: {xnode.op_type}, name: {xnode.op_name}."
                op_dict[xnode.op_name] = op
            else:
                msg = f"Not found function: XConverter._to_{xnode.op_type}"
                logger.error(msg)
                raise AttributeError(msg)
        logger.debug(f"numb. of XIR ops: {len(op_dict)}")

        return graph

    @classmethod
    def make_xir_graph_v2(cls, xmodel: XModel, layout: Layout) -> "xir.Graph":
        """
        Convert an xnnc model into an xir graph

        Parameters
        ----------
        xmodel : XModel
            xnnc XModel object
        layout : Layout
            layout of the input XModel object

        Returns
        -------
        Graph
            xir Graph object
        """

        # TODO: solve the incompatibility: xnnc conv2dtranspose has an input for specifying output_shape, while xir transposed-conv2d does not support this.
        cls.__remove_conv2dtranspose_output_shape(xmodel)

        # save op for future look-up.
        # key: name of "xir.Op" object, value: "xir.Op" object
        op_dict: Dict[str, "xir.Op"] = {}
        # create "xir.Op" object one-by-one
        pbar = tqdm(
            xmodel.xnodes,
            desc="[INFO] generate xmodel",
            bar_format="{desc:27}:{percentage:3.0f}%|{bar}{r_bar:50}",
        )
        for xnode in pbar:
            logger.debug(f"* invoke to_{xnode.op_type}")
            func_name = "to_" + xnode.op_type + "_v2"

            # ! debug
            # print(func_name)

            if hasattr(XConverter, func_name):
                func = getattr(XConverter, func_name)
                assert func is not None, f"Not found function: {func_name}"
                func(xnode, op_dict)

            else:
                msg = f"Not found function: XConverter.{func_name}"
                logger.error(msg)
                raise AttributeError(msg)
        logger.debug(f"numb. of XIR ops: {len(op_dict)}")

        # create xir Graph object
        graph = xir_proto.Graph()
        graph.graph_name = xmodel.name
        # origin
        attr = xir_proto.AttrValue()
        attr.string_value = xmodel.origin
        graph.graph_attr["origin"].CopyFrom(attr)

        # nodes
        graph.op_node.extend(op_dict.values())

        # subgraph
        subgraph = xir_proto.SubGraph()
        subgraph.subgraph_name = "root"
        subgraph.op_name.extend(op_dict.keys())
        graph.subg_root.CopyFrom(subgraph)

        return graph

    @classmethod
    def to_input(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: data, op_name: {xnode.op_name}")

        # attrs
        attrs: Dict[str, Any] = {}
        attrs["shape"] = xnode.outputs_tensor[0].shape
        attrs["data_type"] = xnode.outputs_tensor[0].dtype.name

        # create op
        return graph.create_op(name=xnode.op_name, kind="data", attrs=attrs)

    @classmethod
    def to_input_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: data, op_name: {xnode.op_name}")

        # create xir node
        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "data"

        # shape
        val_shape = xir_proto.Int32Vec()
        val_shape.value.extend(xnode.shape)
        attr_value = xir_proto.AttrValue()
        attr_value.int32_vec_value.CopyFrom(val_shape)
        node.op_attr["shape"].CopyFrom(attr_value)

        # data type
        attr_value = xir_proto.AttrValue()
        attr_value.string_value = "float32"
        node.op_attr["data_type"].CopyFrom(attr_value)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_conv2d(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")
        # set out_tensor: (H, W, in_ch, out_ch) => (out_ch, H, W, in_ch)
        out_tensor_w = tc.transpose(xnode.weights, (3, 0, 1, 2))
        logger.debug(
            f"attribute: weights: shape (out_channels, out_height, out_width, in_channels): {out_tensor_w.shape}, dtype: {out_tensor_w.dtype}"
        )
        # create const op for weights
        const_op_name = xnode.op_name + "_weights"
        weights = graph.create_const_op(const_op_name, out_tensor_w.to_numpy())
        # put into op_dict
        assert const_op_name not in op_dict, f"Duplicated op: op_name={const_op_name}."
        op_dict[const_op_name] = weights

        if (
            xnode.quant_weights["quantize_pos"] is not None
            and xnode.quant_weights["bit_width"] is not None
        ):
            # create FixedNeuron op for weights
            fix_op_name = const_op_name + "_fixneuron"
            # dtype
            dtype = out_tensor_w.dtype
            # shape
            shape = out_tensor_w.shape
            out_tensor = np.zeros(shape, dtype=dtype)
            fix_op_weights = cls.__create_fixneuron(
                fix_op_name,
                [const_op_name],
                xnode.quant_weights,
                out_tensor,
                graph,
                op_dict,
            )

        # create const op for bias
        bias, fix_op_bias = None, None
        if hasattr(xnode, "bias_term"):
            if xnode.bias_term:
                # set out_tensor
                out_tensor_b = xnode.bias
                if out_tensor_b.ndims!=1:
                    raise ValueError(f"type: {xnode.op_type}, name: {xnode.op_name}. bias tensor shape's dim num is {out_tensor_b.ndims}.")
                logger.debug(
                    f"attribute: bias: shape: {out_tensor_b.shape}, dtype: {out_tensor_b.dtype}"
                )
                const_op_name = xnode.op_name + "_bias"
                bias = graph.create_const_op(const_op_name, out_tensor_b.to_numpy())
                # put into op_dict
                assert (
                    const_op_name not in op_dict
                ), f"Duplicated op: op_name={const_op_name}."
                op_dict[const_op_name] = bias

                if (
                    xnode.quant_bias["quantize_pos"] is not None
                    and xnode.quant_bias["bit_width"] is not None
                ):
                    if xnode.quant_bias["round_mode"] is None:
                        print("here")
                    # create FixedNeuron op for bias
                    fix_op_name = const_op_name + "_fixneuron"
                    # dtype
                    dtype = out_tensor_b.dtype
                    # shape
                    shape = out_tensor_b.shape
                    out_tensor = np.zeros(shape, dtype=dtype)
                    fix_op_bias = cls.__create_fixneuron(
                        fix_op_name,
                        [const_op_name],
                        xnode.quant_bias,
                        out_tensor,
                        graph,
                        op_dict,
                    )

        # create input ops for conv2d
        input_ops = {}
        input_ops["weights"] = (
            [fix_op_weights]
            if xnode.quant_weights["quantize_pos"] is not None
            and xnode.quant_weights["bit_width"] is not None
            else [weights]
        )
        if xnode.bias_term:
            input_ops["bias"] = (
                [fix_op_bias]
                if xnode.quant_bias["quantize_pos"] is not None
                and xnode.quant_bias["bit_width"] is not None
                else [bias]
            )
        input_ops["input"] = []
        if xnode.bottom is not None and len(xnode.bottom) > 0:
            for in_op_name in xnode.bottom:
                input_op = op_dict.get(in_op_name)
                assert (
                    input_op is not None
                ), f"[ERROR] input op is None: op name: {xnode.op_name}"
                input_ops["input"].append(input_op)

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["kernel"] = xnode.kernel_size[::-1]
        logger.debug(f"attribute: (kernel_w, kernel_h): {attrs['kernel']}")
        attrs["dilation"] = xnode.dilation[-2:][::-1]
        logger.debug(f"attribute: (dilation_w, dilation_h): {attrs['dilation']}")
        attrs["stride"] = xnode.strides[::-1]
        logger.debug(f"attribute: (stride_w, stride_h): {attrs['stride']}")
        if xnode.pad_mode == PadMode.EXPLICIT:
            attrs["pad_mode"] = xnode.round_mode.name
            attrs["pad"] = xnode.padding[2:] + xnode.padding[:2]
        elif xnode.pad_mode == PadMode.SAME:
            attrs["pad_mode"] = PadMode.SAME.name
        elif xnode.pad_mode == PadMode.VALID:
            attrs["pad_mode"] = PadMode.VALID.name
        else:
            raise ValueError(f"Unsupported pad mode: {xnode.pad_mode}.")
        logger.debug(f"attribute: pad_mode: {attrs['pad_mode']}")
        if "pad" in attrs:
            logger.debug(f"attribute: pad (left, right, top, bottom): {attrs['pad']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "conv2d",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_conv2d_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        conv_node = xir_proto.OPNode()
        conv_node.op_name = xnode.op_name
        conv_node.op_type = "conv2d"

        # attributes of op
        # (kernel_w, kernel_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.kernel_size[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        conv_node.op_attr["kernel"].CopyFrom(attr)
        # (stride_w, stride_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.strides[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        conv_node.op_attr["stride"].CopyFrom(attr)
        # (dilation_w, dilation_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.dilation[-2:][::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        conv_node.op_attr["dilation"].CopyFrom(attr)
        # pad_mode
        attr_value = xir_proto.AttrValue()
        attr_value.string_value = (
            xnode.round_mode.name
            if xnode.pad_mode == PadMode.EXPLICIT
            else xnode.pad_mode.name
        )
        conv_node.op_attr["pad_mode"].CopyFrom(attr_value)
        # pad (pad_w_before, pad_w_after, pad_h_before, pad_h_after)
        if xnode.pad_mode == PadMode.EXPLICIT:
            attr_value = xir_proto.Int32Vec()
            attr_value.value.extend(xnode.padding[2:] + xnode.padding[:2])
            attr = xir_proto.AttrValue()
            attr.int32_vec_value.CopyFrom(attr_value)
            conv_node.op_attr["pad"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, conv_node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)

        # weights: (H, W, in_ch, out_ch) => (out_ch, H, W, in_ch)
        weights = tc.transpose(xnode.weights, (3, 0, 1, 2))
        const_node_w, fix_node_w = cls.__create_xir_const_node_from_xtensor(
            name=xnode.op_name + "_weights",
            xtensor=weights,
            quant_info=xnode.quant_weights,
        )
        assert (
            const_node_w.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={const_node_w.op_name}."
        op_dict[const_node_w.op_name] = const_node_w
        if fix_node_w:
            assert (
                fix_node_w.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={fix_node_w.op_name}."
            op_dict[fix_node_w.op_name] = fix_node_w

        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "weights"
        if fix_node_w:
            op_arg.arg_ops.extend([fix_node_w.op_name])
        else:
            op_arg.arg_ops.extend([const_node_w.op_name])
        op_args.append(op_arg)

        # bias
        if xnode.bias_term:
            const_node_b, fix_node_b = cls.__create_xir_const_node_from_xtensor(
                name=xnode.op_name + "_bias",
                xtensor=xnode.bias,
                quant_info=xnode.quant_bias,
            )
            assert (
                const_node_b.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={const_node_b.op_name}."
            op_dict[const_node_b.op_name] = const_node_b
            if fix_node_b:
                assert (
                    fix_node_b.op_name not in op_dict
                ), f"[ERROR] Found duplicate xir op: op_name={fix_node_b.op_name}."
                op_dict[fix_node_b.op_name] = fix_node_b

            op_arg = xir_proto.OpArg()
            op_arg.arg_name = "bias"
            if fix_node_b:
                op_arg.arg_ops.extend([fix_node_b.op_name])
            else:
                op_arg.arg_ops.extend([const_node_b.op_name])
            op_args.append(op_arg)

        conv_node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            conv_node.op_name, xnode.outputs_tensor[0]
        )
        conv_node.output_tensor.CopyFrom(output_tensor)

        assert (
            conv_node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={conv_node.op_name}."
        op_dict[conv_node.op_name] = conv_node

    @classmethod
    def to_depthwise_conv2d(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: depthwise-conv2d, op_name: {xnode.op_name}")
        # set out_tensor
        if xnode.host.origin == "caffe":
            # xir special case
            # (h,w,ic,cm) => (h,w,cm,ic)
            out_tensor_w = tc.transpose(xnode.weights, (0, 1, 3, 2))
            # (h,w,cm,ic) => (ic,h,w,cm)
            out_tensor_w = tc.transpose(out_tensor_w, (3, 0, 1, 2))
        else:
            # (h,w,ic,cm) => (cm,h,w,ic)
            out_tensor_w = tc.transpose(xnode.weights, (3, 0, 1, 2))
        logger.debug(
            f"attribute: weights: shape (out_channels, out_height, out_width, in_channels): {out_tensor_w.shape}, dtype: {out_tensor_w.dtype}"
        )
        # create const op for weights
        const_op_name = xnode.op_name + "_weights"
        weights = graph.create_const_op(const_op_name, out_tensor_w.to_numpy())
        # put into op_dict
        assert const_op_name not in op_dict, f"Duplicated op: op_name={const_op_name}."
        op_dict[const_op_name] = weights

        # create FixedNeuron op for weights
        if (
            xnode.quant_weights["quantize_pos"] is not None
            and xnode.quant_weights["bit_width"] is not None
        ):
            # create FixedNeuron op for weights
            fix_op_name = const_op_name + "_fixneuron"
            # dtype
            dtype = out_tensor_w.dtype
            # shape
            shape = out_tensor_w.shape
            out_tensor = np.zeros(shape, dtype=dtype)
            fix_op_weights = cls.__create_fixneuron(
                fix_op_name,
                [const_op_name],
                xnode.quant_weights,
                out_tensor,
                graph,
                op_dict,
            )

        # create const op for bias
        bias, fix_op_bias = None, None
        if hasattr(xnode, "bias_term"):
            if xnode.bias_term:
                # set out_tensor
                out_tensor_b = xnode.bias
                logger.debug(
                    f"attribute: bias: shape: {out_tensor_b.shape}, dtype: {out_tensor_b.dtype}"
                )
                const_op_name = xnode.op_name + "_bias"
                bias = graph.create_const_op(const_op_name, out_tensor_b.to_numpy())
                # put into op_dict
                assert (
                    const_op_name not in op_dict
                ), f"Duplicated op: op_name={const_op_name}."
                op_dict[const_op_name] = bias

                if (
                    xnode.quant_bias["quantize_pos"] is not None
                    and xnode.quant_bias["bit_width"] is not None
                ):
                    if xnode.quant_bias["round_mode"] is None:
                        print("here")
                    # create FixedNeuron op for bias
                    fix_op_name = const_op_name + "_fixneuron"
                    # dtype
                    dtype = out_tensor_b.dtype
                    # shape
                    shape = out_tensor_b.shape
                    out_tensor = np.zeros(shape, dtype=dtype)
                    fix_op_bias = cls.__create_fixneuron(
                        fix_op_name,
                        [const_op_name],
                        xnode.quant_bias,
                        out_tensor,
                        graph,
                        op_dict,
                    )

        # create input ops for conv2d
        input_ops = {}
        input_ops["weights"] = [fix_op_weights]
        if xnode.bias_term:
            input_ops["bias"] = [fix_op_bias]
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["kernel"] = xnode.kernel_size[::-1]
        logger.debug(f"attribute: (kernel_w, kernel_h): {attrs['kernel']}")
        attrs["dilation"] = xnode.dilation[-2:][::-1]
        logger.debug(f"attribute: (dilation_w, dilation_h): {attrs['dilation']}")
        attrs["stride"] = xnode.strides[::-1]
        logger.debug(f"attribute: (stride_w, stride_h): {attrs['stride']}")
        if xnode.pad_mode == PadMode.EXPLICIT:
            assert xnode.round_mode in [
                RoundMode.FLOOR,
                RoundMode.CEIL,
                RoundMode.ROUND_DOWN,
                RoundMode.ROUND_UP,
            ]
            attrs["pad_mode"] = (
                RoundMode.FLOOR.name
                if xnode.round_mode in [RoundMode.FLOOR, RoundMode.ROUND_DOWN]
                else RoundMode.CEIL.name
            )
            attrs["pad"] = xnode.padding[2:] + xnode.padding[:2]
        elif xnode.pad_mode == PadMode.SAME:
            attrs["pad_mode"] = PadMode.SAME.name
        elif xnode.pad_mode == PadMode.VALID:
            attrs["pad_mode"] = PadMode.VALID.name
        else:
            raise ValueError(f"Unsupported pad mode: {xnode.pad_mode}.")
        logger.debug(f"attribute: pad_mode: {attrs['pad_mode']}")
        if "pad" in attrs:
            logger.debug(f"attribute: pad (left, right, top, bottom): {attrs['pad']}")

        # create op
        return graph.create_op(
            xnode.op_name, "depthwise-conv2d", attrs=attrs, input_ops=input_ops
        )

    @classmethod
    def to_depthwise_conv2d_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        conv_node = xir_proto.OPNode()
        conv_node.op_name = xnode.op_name
        conv_node.op_type = "depthwise-conv2d"

        # attributes of op
        # (kernel_w, kernel_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.kernel_size[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        conv_node.op_attr["kernel"].CopyFrom(attr)
        # (stride_w, stride_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.strides[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        conv_node.op_attr["stride"].CopyFrom(attr)
        # (dilation_w, dilation_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.dilation[-2:][::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        conv_node.op_attr["dilation"].CopyFrom(attr)
        # pad_mode
        attr_value = xir_proto.AttrValue()
        attr_value.string_value = (
            xnode.round_mode.name
            if xnode.pad_mode == PadMode.EXPLICIT
            else xnode.pad_mode.name
        )
        conv_node.op_attr["pad_mode"].CopyFrom(attr_value)
        # pad (pad_w_before, pad_w_after, pad_h_before, pad_h_after)
        if xnode.pad_mode == PadMode.EXPLICIT:
            attr_value = xir_proto.Int32Vec()
            attr_value.value.extend(xnode.padding[2:] + xnode.padding[:2])
            attr = xir_proto.AttrValue()
            attr.int32_vec_value.CopyFrom(attr_value)
            conv_node.op_attr["pad"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, conv_node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)

        # weights
        if xnode.host.origin == "caffe":
            # xir special case
            # (h,w,ic,cm) => (h,w,cm,ic)
            weights = tc.transpose(xnode.weights, (0, 1, 3, 2))
            # (h,w,cm,ic) => (ic,h,w,cm)
            weights = tc.transpose(weights, (3, 0, 1, 2))
        else:
            # (h,w,ic,cm) => (cm,h,w,ic)
            weights = tc.transpose(xnode.weights, (3, 0, 1, 2))
        const_node_w, fix_node_w = cls.__create_xir_const_node_from_xtensor(
            name=xnode.op_name + "_weights",
            xtensor=weights,
            quant_info=xnode.quant_weights,
        )
        assert (
            const_node_w.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={const_node_w.op_name}."
        op_dict[const_node_w.op_name] = const_node_w
        if fix_node_w:
            assert (
                fix_node_w.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={fix_node_w.op_name}."
            op_dict[fix_node_w.op_name] = fix_node_w

        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "weights"
        if fix_node_w:
            op_arg.arg_ops.extend([fix_node_w.op_name])
        else:
            op_arg.arg_ops.extend([const_node_w.op_name])
        op_args.append(op_arg)

        # bias
        if xnode.bias_term:
            const_node_b, fix_node_b = cls.__create_xir_const_node_from_xtensor(
                name=xnode.op_name + "_bias",
                xtensor=xnode.bias,
                quant_info=xnode.quant_bias,
            )
            assert (
                const_node_b.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={const_node_b.op_name}."
            op_dict[const_node_b.op_name] = const_node_b
            if fix_node_b:
                assert (
                    fix_node_b.op_name not in op_dict
                ), f"[ERROR] Found duplicate xir op: op_name={fix_node_b.op_name}."
                op_dict[fix_node_b.op_name] = fix_node_b

            op_arg = xir_proto.OpArg()
            op_arg.arg_name = "bias"
            if fix_node_b:
                op_arg.arg_ops.extend([fix_node_b.op_name])
            else:
                op_arg.arg_ops.extend([const_node_b.op_name])
            op_args.append(op_arg)

        conv_node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            conv_node.op_name, xnode.outputs_tensor[0]
        )
        conv_node.output_tensor.CopyFrom(output_tensor)

        assert (
            conv_node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={conv_node.op_name}."
        op_dict[conv_node.op_name] = conv_node

    @classmethod
    def to_conv2d_transpose(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # h,w,oc,ic => h',w',oc,ic => oc,h',w',ic
        out_tensor_w = tc.transpose(tc.flip(xnode.weights, [0, 1]), (2, 0, 1, 3))
        logger.debug(
            f"attribute: weights: shape (out_channels, out_height, out_width, in_channels): {out_tensor_w.shape}, dtype: {out_tensor_w.dtype}"
        )

        # create const op for weights
        const_op_name = xnode.op_name + "_weights"
        weights = graph.create_const_op(const_op_name, out_tensor_w.to_numpy())
        # put into op_dict
        assert const_op_name not in op_dict, f"Duplicated op: op_name={const_op_name}."
        op_dict[const_op_name] = weights

        if (
            xnode.quant_weights["quantize_pos"] is not None
            and xnode.quant_weights["bit_width"] is not None
        ):
            # create FixedNeuron op for weights
            fix_op_name = const_op_name + "_fixneuron"
            # dtype
            dtype = out_tensor_w.dtype
            # shape
            shape = out_tensor_w.shape
            out_tensor = np.zeros(shape, dtype=dtype)
            fix_op_weights = cls.__create_fixneuron(
                fix_op_name,
                [const_op_name],
                xnode.quant_weights,
                out_tensor,
                graph,
                op_dict,
            )

        # create const op for bias
        bias, fix_op_bias = None, None
        if hasattr(xnode, "bias_term"):
            if xnode.bias_term:
                # set out_tensor
                out_tensor_b = xnode.bias
                logger.debug(
                    f"attribute: bias: shape: {out_tensor_b.shape}, dtype: {out_tensor_b.dtype}"
                )
                const_op_name = xnode.op_name + "_bias"
                bias = graph.create_const_op(const_op_name, out_tensor_b.to_numpy())
                # put into op_dict
                assert (
                    const_op_name not in op_dict
                ), f"Duplicated op: op_name={const_op_name}."
                op_dict[const_op_name] = bias

                if (
                    xnode.quant_bias["quantize_pos"] is not None
                    and xnode.quant_bias["bit_width"] is not None
                ):
                    # create FixedNeuron op for bias
                    fix_op_name = const_op_name + "_fixneuron"
                    # dtype
                    dtype = out_tensor_b.dtype
                    # shape
                    shape = out_tensor_b.shape
                    out_tensor = np.zeros(shape, dtype=dtype)
                    fix_op_bias = cls.__create_fixneuron(
                        fix_op_name,
                        [const_op_name],
                        xnode.quant_bias,
                        out_tensor,
                        graph,
                        op_dict,
                    )

        # create input ops for conv2d
        input_ops = {}
        input_ops["weights"] = (
            [fix_op_weights]
            if xnode.quant_weights["quantize_pos"] is not None
            and xnode.quant_weights["bit_width"] is not None
            else [weights]
        )
        if xnode.bias_term:
            input_ops["bias"] = (
                [fix_op_bias]
                if xnode.quant_bias["quantize_pos"] is not None
                and xnode.quant_bias["bit_width"] is not None
                else [bias]
            )
        input_ops["input"] = []
        if xnode.bottom is not None and len(xnode.bottom) > 0:
            # todo: need to refactor xir transposed-conv2d op to support two inputs
            # for in_op_name in xnode.bottom:
            #     input_op = op_dict.get(in_op_name)
            #     assert (
            #         input_op is not None
            #     ), f"[ERROR] input op is None: op name: {xnode.op_name}"
            #     input_ops["input"].append(input_op)

            input_op = op_dict.get(xnode.bottom[0])
            assert (
                input_op is not None
            ), f"[ERROR] input op is None: op name: {xnode.op_name}"
            input_ops["input"].append(input_op)

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["kernel"] = xnode.kernel_size[::-1]
        logger.debug(f"attribute: (kernel_w, kernel_h): {attrs['kernel']}")
        attrs["dilation"] = xnode.dilation[-2:][::-1]
        logger.debug(f"attribute: (dilation_w, dilation_h): {attrs['dilation']}")
        attrs["stride"] = xnode.strides[::-1]
        logger.debug(f"attribute: (stride_w, stride_h): {attrs['stride']}")
        if xnode.pad_mode == PadMode.EXPLICIT:
            assert xnode.round_mode in [
                RoundMode.FLOOR,
                RoundMode.CEIL,
                RoundMode.ROUND_DOWN,
                RoundMode.ROUND_UP,
            ]
            attrs["pad_mode"] = (
                RoundMode.FLOOR.name
                if xnode.round_mode in [RoundMode.FLOOR, RoundMode.ROUND_DOWN]
                else RoundMode.CEIL.name
            )
            attrs["pad"] = xnode.padding[2:] + xnode.padding[:2]
        elif xnode.pad_mode == PadMode.SAME:
            attrs["pad_mode"] = PadMode.SAME.name
        elif xnode.pad_mode == PadMode.VALID:
            attrs["pad_mode"] = PadMode.VALID.name
        else:
            raise ValueError(f"Unsupported pad mode: {xnode.pad_mode}.")
        logger.debug(f"attribute: pad_mode: {attrs['pad_mode']}")
        if "pad" in attrs:
            logger.debug(f"attribute: pad (left, right, top, bottom): {attrs['pad']}")

        # create op
        return graph.create_op(
            xnode.op_name, "transposed-conv2d", attrs=attrs, input_ops=input_ops
        )

    @classmethod
    def to_conv2d_transpose_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        conv_node = xir_proto.OPNode()
        conv_node.op_name = xnode.op_name
        conv_node.op_type = "transposed-conv2d"

        # attributes of op
        # (kernel_w, kernel_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.kernel_size[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        conv_node.op_attr["kernel"].CopyFrom(attr)
        # (stride_w, stride_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.strides[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        conv_node.op_attr["stride"].CopyFrom(attr)
        # (dilation_w, dilation_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.dilation[-2:][::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        conv_node.op_attr["dilation"].CopyFrom(attr)
        # pad_mode
        attr_value = xir_proto.AttrValue()
        attr_value.string_value = (
            xnode.round_mode.name
            if xnode.pad_mode == PadMode.EXPLICIT
            else xnode.pad_mode.name
        )
        conv_node.op_attr["pad_mode"].CopyFrom(attr_value)
        # pad (pad_w_before, pad_w_after, pad_h_before, pad_h_after)
        if xnode.pad_mode == PadMode.EXPLICIT:
            attr_value = xir_proto.Int32Vec()
            attr_value.value.extend(xnode.padding[2:] + xnode.padding[:2])
            attr = xir_proto.AttrValue()
            attr.int32_vec_value.CopyFrom(attr_value)
            conv_node.op_attr["pad"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, conv_node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)

        # h,w,oc,ic => h',w',oc,ic => oc,h',w',ic
        weights = tc.transpose(tc.flip(xnode.weights, [0, 1]), (2, 0, 1, 3))
        const_node_w, fix_node_w = cls.__create_xir_const_node_from_xtensor(
            name=xnode.op_name + "_weights",
            xtensor=weights,
            quant_info=xnode.quant_weights,
        )
        assert (
            const_node_w.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={const_node_w.op_name}."
        op_dict[const_node_w.op_name] = const_node_w
        if fix_node_w:
            assert (
                fix_node_w.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={fix_node_w.op_name}."
            op_dict[fix_node_w.op_name] = fix_node_w

        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "weights"
        if fix_node_w:
            op_arg.arg_ops.extend([fix_node_w.op_name])
        else:
            op_arg.arg_ops.extend([const_node_w.op_name])
        op_args.append(op_arg)

        # bias
        if xnode.bias_term:
            const_node_b, fix_node_b = cls.__create_xir_const_node_from_xtensor(
                name=xnode.op_name + "_bias",
                xtensor=xnode.bias,
                quant_info=xnode.quant_bias,
            )
            assert (
                const_node_b.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={const_node_b.op_name}."
            op_dict[const_node_b.op_name] = const_node_b
            if fix_node_b:
                assert (
                    fix_node_b.op_name not in op_dict
                ), f"[ERROR] Found duplicate xir op: op_name={fix_node_b.op_name}."
                op_dict[fix_node_b.op_name] = fix_node_b

            op_arg = xir_proto.OpArg()
            op_arg.arg_name = "bias"
            if fix_node_b:
                op_arg.arg_ops.extend([fix_node_b.op_name])
            else:
                op_arg.arg_ops.extend([const_node_b.op_name])
            op_args.append(op_arg)

        conv_node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            conv_node.op_name, xnode.outputs_tensor[0]
        )
        conv_node.output_tensor.CopyFrom(output_tensor)

        assert (
            conv_node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={conv_node.op_name}."
        op_dict[conv_node.op_name] = conv_node

    @classmethod
    def to_relu(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")
        # create input ops for conv2d
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        if xnode.alpha != 0.0:  # leaky relu
            attrs["alpha"] = xnode.alpha
            logger.debug(f"attribute: alpha: {attrs['alpha']}")

        if "alpha" not in attrs:
            # create Relu op
            return graph.create_op(xnode.op_name, "relu", input_ops=input_ops)
        else:
            # create LeakyRelu op
            return graph.create_op(
                xnode.op_name,
                "leaky-relu",
                attrs=attrs,
                input_ops=input_ops,
            )

    @classmethod
    def to_relu_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name

        # attributes of op
        cls.__set_xir_node_quant_info(xnode, node)

        if xnode.alpha != 0.0:  # leaky relu
            node.op_type = "leaky-relu"

            attr = xir_proto.AttrValue()
            attr.float_value = xnode.alpha
            node.op_attr["alpha"].CopyFrom(attr)

        else:
            node.op_type = "relu"

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_relu6(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: relu6, op_name: {xnode.op_name}")
        # create input ops for Mean
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create op
        return graph.create_op(xnode.op_name, "relu6", input_ops=input_ops)

    @classmethod
    def to_relu6_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "relu6"

        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_prelu(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")
        # create input ops for prelu
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        out_tensor_w = xnode.alpha
        const_op_name = xnode.op_name + "_weight"
        weights = graph.create_const_op(const_op_name, xnode.alpha.to_numpy())
        # put into op_dict
        assert const_op_name not in op_dict, f"Duplicated op: op_name={const_op_name}."
        op_dict[const_op_name] = weights

        if (
            xnode.quant_alpha["quantize_pos"] is not None
            and xnode.quant_alpha["bit_width"] is not None
        ):
            # create FixedNeuron op for weights
            fix_op_name = const_op_name + "_fixneuron"
            # dtype
            dtype = out_tensor_w.dtype
            # shape
            shape = out_tensor_w.shape
            out_tensor = np.zeros(shape, dtype=dtype)
            fix_op_weights = cls.__create_fixneuron(
                fix_op_name,
                [const_op_name],
                xnode.quant_alpha,
                out_tensor,
                graph,
                op_dict,
            )

        input_ops['weight'] = (
            [fix_op_weights]
            if xnode.quant_alpha["quantize_pos"] is not None
            and xnode.quant_alpha["bit_width"] is not None
            else [weights]
        )

        # set attrs for op
        attrs: Dict[str, Any] = {}

        # create PRelu op
        return graph.create_op(name=xnode.op_name,
                               kind="prelu",
                               attrs=attrs,
                               input_ops=input_ops)

    @classmethod
    def to_maxpool(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops for conv2d
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["kernel"] = xnode.kernel_size[::-1]
        logger.debug(f"attribute: (kernel_w, kernel_h): {attrs['kernel']}")
        attrs["stride"] = xnode.strides[::-1]
        logger.debug(f"attribute: (stride_w, kernel_h): {attrs['stride']}")
        if xnode.pad_mode == PadMode.EXPLICIT:
            assert xnode.round_mode in [
                RoundMode.FLOOR,
                RoundMode.CEIL,
                RoundMode.ROUND_DOWN,
                RoundMode.ROUND_UP,
            ]
            attrs["pad_mode"] = (
                RoundMode.FLOOR.name
                if xnode.round_mode in [RoundMode.FLOOR, RoundMode.ROUND_DOWN]
                else RoundMode.CEIL.name
            )
            attrs["pad"] = xnode.padding[2:] + xnode.padding[:2]
        elif xnode.pad_mode == PadMode.SAME:
            attrs["pad_mode"] = PadMode.SAME.name
        elif xnode.pad_mode == PadMode.VALID:
            attrs["pad_mode"] = PadMode.VALID.name
        else:
            raise ValueError(f"Unsupported pad mode: {xnode.pad_mode}.")
        logger.debug(f"attribute: pad_mode: {attrs['pad_mode']}")
        if "pad" in attrs:
            logger.debug(f"attribute: pad (left, right, top, bottom): {attrs['pad']}")
        attrs["global"] = xnode.is_global
        logger.debug(f"attribute: global: {attrs['global']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "maxpool2d",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_maxpool_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "maxpool2d"

        # attributes of op
        # (kernel_w, kernel_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.kernel_size[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["kernel"].CopyFrom(attr)

        # (stride_w, stride_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.strides[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["stride"].CopyFrom(attr)

        # pad_mode
        attr = xir_proto.AttrValue()
        attr.string_value = (
            xnode.round_mode.name
            if xnode.pad_mode == PadMode.EXPLICIT
            else xnode.pad_mode.name
        )
        node.op_attr["pad_mode"].CopyFrom(attr)

        # padding (pad_w_before, pad_w_after, pad_h_before, pad_h_after)
        if xnode.pad_mode == PadMode.EXPLICIT:
            attr_value = xir_proto.Int32Vec()
            attr_value.value.extend(xnode.padding[2:] + xnode.padding[:2])
            attr = xir_proto.AttrValue()
            attr.int32_vec_value.CopyFrom(attr_value)
            node.op_attr["pad"].CopyFrom(attr)

        # global
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.is_global
        node.op_attr["global"].CopyFrom(attr)

        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_elemadd(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: eltwise, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create op
        return graph.create_op(
            xnode.op_name,
            "add",
            input_ops=input_ops,
        )

    @classmethod
    def to_elemadd_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "add"

        # attributes of op
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_elemnegative(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: neg, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create op
        return graph.create_op(xnode.op_name, "neg", input_ops=input_ops)

    @classmethod
    def to_elemnegative_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "neg"

        # attributes of op
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_elemmul(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: mul, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create op
        return graph.create_op(xnode.op_name, "mul", input_ops=input_ops)

    @classmethod
    def to_elemmul_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "mul"

        # attributes of op
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_elemsub(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: sub, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create op
        return graph.create_op(xnode.op_name, "sub", input_ops=input_ops)

    @classmethod
    def to_elemsub_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "sub"

        # attributes of op
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_elemrealdiv(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: elemrealdiv, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create op
        return graph.create_op(xnode.op_name, "div", input_ops=input_ops)

    @classmethod
    def to_elemrealdiv_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "div"

        # attributes of op
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_elemexp(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: elemexp, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create op
        return graph.create_op(xnode.op_name, "exp", input_ops=input_ops)

    @classmethod
    def to_elemexp_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "exp"

        # attributes of op
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_stridedslice(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: strided_slice, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["begin"] = xnode.begin
        logger.debug(f"attribute: begin: {attrs['begin']}")
        attrs["end"] = xnode.end
        logger.debug(f"attribute: end: {attrs['end']}")
        attrs["strides"] = xnode.strides
        logger.debug(f"attribute: strides: {attrs['strides']}")
        # ! do not remove the follow comment-out code
        # attrs["begin_mask"] = xnode.begin_mask
        # logger.debug(f"attribute: begin_mask: {attrs['begin_mask']}")
        # attrs["end_mask"] = xnode.end_mask
        # logger.debug(f"attribute: end_mask: {attrs['end_mask']}")
        # attrs["ellipsis_mask"] = xnode.ellipsis_mask
        # logger.debug(f"attribute: ellipsis_mask: {attrs['ellipsis_mask']}")
        # attrs["new_axis_mask"] = xnode.new_axis_mask
        # logger.debug(f"attribute: new_axis_mask: {attrs['new_axis_mask']}")
        # attrs["shrink_axis_mask"] = xnode.shrink_axis_mask
        # logger.debug(f"attribute: shrink_axis_mask: {attrs['shrink_axis_mask']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "strided_slice",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_stridedslice_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "strided_slice"

        # attributes of op
        # begin
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.begin)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["begin"].CopyFrom(attr)
        # end
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.begin)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["end"].CopyFrom(attr)
        # strides
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.strides)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["strides"].CopyFrom(attr)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_avgpool(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")
        # create input ops for conv2d
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["kernel"] = xnode.kernel_size[::-1]
        logger.debug(f"attribute: (kernel_w, kernel_h): {attrs['kernel']}")
        attrs["stride"] = xnode.strides[::-1]
        logger.debug(f"attribute: (stride_w, stride_h): {attrs['stride']}")
        if xnode.pad_mode == PadMode.EXPLICIT:
            assert xnode.round_mode in [
                RoundMode.FLOOR,
                RoundMode.CEIL,
                RoundMode.ROUND_DOWN,
                RoundMode.ROUND_UP,
            ]
            attrs["pad_mode"] = (
                RoundMode.FLOOR.name
                if xnode.round_mode in [RoundMode.FLOOR, RoundMode.ROUND_DOWN]
                else RoundMode.CEIL.name
            )
            attrs["pad"] = xnode.padding[2:] + xnode.padding[:2]
        elif xnode.pad_mode == PadMode.SAME:
            attrs["pad_mode"] = PadMode.SAME.name
        elif xnode.pad_mode == PadMode.VALID:
            attrs["pad_mode"] = PadMode.VALID.name
        else:
            raise ValueError(f"Unsupported pad mode: {xnode.pad_mode}.")
        logger.debug(f"attribute: pad_mode: {attrs['pad_mode']}")
        if "pad" in attrs:
            logger.debug(f"attribute: pad (left, right, top, bottom): {attrs['pad']}")
        attrs["global"] = xnode.is_global
        logger.debug(f"attribute: global: {attrs['global']}")

        # count_include_pad
        if xnode.host.origin == "nndct":
            attrs["count_include_pad"] = xnode.zero_padding_included
            logger.debug(f"attribute: count_include_pad: {attrs['count_include_pad']}")

            attrs["count_include_invalid"] = xnode.count_include_invalid
            logger.debug(
                f"attribute: count_include_invalid: {attrs['count_include_invalid']}"
            )

        else:
            attrs["count_include_pad"] = xnode.count_include_pad
            logger.debug(f"attribute: count_include_pad: {attrs['count_include_pad']}")

            # count_include_invalid
            if xnode.host.origin in ["caffe", "nndct_pytorch"]:
                attrs["count_include_invalid"] = False
            elif xnode.host.origin in [
                "tensorflow",
                "pytorch",
                "tensorflow2",
            ]:
                attrs["count_include_invalid"] = True
            logger.debug(
                f"attribute: count_include_invalid: {attrs['count_include_invalid']}"
            )

        # create op
        return graph.create_op(
            xnode.op_name,
            "avgpool2d",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_avgpool_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "avgpool2d"

        # attributes of op
        # (kernel_w, kernel_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.kernel_size[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["kernel"].CopyFrom(attr)

        # (stride_w, stride_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.strides[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["stride"].CopyFrom(attr)

        # pad_mode
        attr = xir_proto.AttrValue()
        attr.string_value = (
            xnode.round_mode.name
            if xnode.pad_mode == PadMode.EXPLICIT
            else xnode.pad_mode.name
        )
        node.op_attr["pad_mode"].CopyFrom(attr)

        # padding (pad_w_before, pad_w_after, pad_h_before, pad_h_after)
        if xnode.pad_mode == PadMode.EXPLICIT:
            attr_value = xir_proto.Int32Vec()
            attr_value.value.extend(xnode.padding[2:] + xnode.padding[:2])
            attr = xir_proto.AttrValue()
            attr.int32_vec_value.CopyFrom(attr_value)
            node.op_attr["pad"].CopyFrom(attr)

        # global
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.is_global
        node.op_attr["global"].CopyFrom(attr)

        if xnode.host.origin == "nndct":
            # count_include_invalid
            attr = xir_proto.AttrValue()
            attr.bool_value = xnode.count_include_invalid
            node.op_attr["count_include_invalid"].CopyFrom(attr)

            # count_include_pad
            attr = xir_proto.AttrValue()
            attr.bool_value = xnode.zero_padding_included
            node.op_attr["count_include_pad"].CopyFrom(attr)

        else:
            # count_include_pad
            attr = xir_proto.AttrValue()
            attr.bool_value = xnode.count_include_pad
            node.op_attr["count_include_pad"].CopyFrom(attr)

            # count_include_invalid
            attr = xir_proto.AttrValue()
            if xnode.host.origin in ["caffe", "nndct_pytorch"]:
                attr.bool_value = False
            elif xnode.host.origin in [
                "tensorflow",
                "pytorch",
                "tensorflow2",
            ]:
                attr.bool_value = True
            node.op_attr["count_include_invalid"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_dot(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: inner-product, op_name: {xnode.op_name}")
        # set out_tensor
        in_shape = xnode.inputs_tensor[0].shape
        if len(in_shape) != 2 and len(in_shape) != 4:
            raise NotImplementedError

        # create const op for weights
        out_tensor_w = xnode.weights
        const_op_name = xnode.op_name + "_weights"
        weights = graph.create_const_op(const_op_name, out_tensor_w.to_numpy())
        # put into op_dict
        assert const_op_name not in op_dict, f"Duplicated op: op_name={const_op_name}."
        op_dict[const_op_name] = weights

        # create FixedNeuron op for weights
        fix_op_name = const_op_name + "_fixneuron"
        # dtype
        dtype = out_tensor_w.dtype
        # shape
        shape = out_tensor_w.shape
        out_tensor = np.zeros(shape, dtype=dtype)
        fix_op_weights = cls.__create_fixneuron(
            fix_op_name,
            [const_op_name],
            xnode.quant_weights,
            out_tensor,
            graph,
            op_dict,
        )

        # create const op for bias
        bias, fix_op_bias = None, None
        if hasattr(xnode, "bias_term"):
            if xnode.bias_term:
                # set out_tensor
                out_tensor_b = xnode.bias
                logger.debug(
                    f"attribute: bias: shape: {out_tensor_b.shape}, dtype: {out_tensor_b.dtype}"
                )
                const_op_name = xnode.op_name + "_bias"
                bias = graph.create_const_op(const_op_name, out_tensor_b.to_numpy())
                # put into op_dict
                assert (
                    const_op_name not in op_dict
                ), f"Duplicated op: op_name={const_op_name}."
                op_dict[const_op_name] = bias

                # create FixedNeuron op for bias
                fix_op_name = const_op_name + "_fixneuron"
                # dtype
                dtype = out_tensor_b.dtype
                # shape
                shape = out_tensor_b.shape
                out_tensor = np.zeros(shape, dtype=dtype)
                fix_op_bias = cls.__create_fixneuron(
                    fix_op_name,
                    [const_op_name],
                    xnode.quant_bias,
                    out_tensor,
                    graph,
                    op_dict,
                )

        # create input ops for dot
        input_ops = {}
        input_ops["weights"] = [fix_op_weights]
        if xnode.bias_term:
            input_ops["bias"] = [fix_op_bias]
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["axis"] = 1
        logger.debug(f"attribute: axis: {attrs['axis']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "inner-product",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_dot_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        in_shape = xnode.inputs_tensor[0].shape
        if len(in_shape) != 2 and len(in_shape) != 4:
            raise NotImplementedError

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "inner-product"

        # attributes of op
        # axis = 1 for both NCHW and NHWC layout
        attr = xir_proto.AttrValue()
        attr.int32_value = 1
        node.op_attr["axis"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)

        # weights: (H, W, in_ch, out_ch) => (out_ch, H, W, in_ch)
        weights = xnode.weights
        const_node_w, fix_node_w = cls.__create_xir_const_node_from_xtensor(
            name=xnode.op_name + "_weights",
            xtensor=weights,
            quant_info=xnode.quant_weights,
        )
        assert (
            const_node_w.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={const_node_w.op_name}."
        op_dict[const_node_w.op_name] = const_node_w
        if fix_node_w:
            assert (
                fix_node_w.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={fix_node_w.op_name}."
            op_dict[fix_node_w.op_name] = fix_node_w

        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "weights"
        if fix_node_w:
            op_arg.arg_ops.extend([fix_node_w.op_name])
        else:
            op_arg.arg_ops.extend([const_node_w.op_name])
        op_args.append(op_arg)

        # bias
        if xnode.bias_term:
            const_node_b, fix_node_b = cls.__create_xir_const_node_from_xtensor(
                name=xnode.op_name + "_bias",
                xtensor=xnode.bias,
                quant_info=xnode.quant_bias,
            )
            assert (
                const_node_b.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={const_node_b.op_name}."
            op_dict[const_node_b.op_name] = const_node_b
            if fix_node_b:
                assert (
                    fix_node_b.op_name not in op_dict
                ), f"[ERROR] Found duplicate xir op: op_name={fix_node_b.op_name}."
                op_dict[fix_node_b.op_name] = fix_node_b

            op_arg = xir_proto.OpArg()
            op_arg.arg_name = "bias"
            if fix_node_b:
                op_arg.arg_ops.extend([fix_node_b.op_name])
            else:
                op_arg.arg_ops.extend([const_node_b.op_name])
            op_args.append(op_arg)

        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_softmax(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: softmax, op_name: {xnode.op_name}")
        # create input ops for conv2d
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        # # convert axis to follow the layout from (N,C,H,W) to (N,H,W,C)
        # ! do not remove
        # if xnode.axis == 0:
        #     attrs["axis"] = xnode.axis
        # elif xnode.axis == 1:
        #     attrs["axis"] = 3
        # elif xnode.axis == 2:
        #     attrs["axis"] = 1
        # elif xnode.axis == 3:
        #     attrs["axis"] = 2
        # else:
        #     raise ValueError(f"[ERROR] Unsupported axis value: {xnode.axis}")
        # ! tmp solution
        attrs["axis"] = -1
        logger.debug(f"attribute: axis: {attrs['axis']}")

        # create op
        if "hard_softmax" in xnode.tmp_params and xnode.tmp_params["hard_softmax"]:
            attrs["type"] = xnode.tmp_params["type"]
            logger.debug(f"attribute: type: {attrs['type']}")
            return graph.create_op(xnode.op_name, "hard-softmax", attrs=attrs, input_ops=input_ops)
        else:
            return graph.create_op(
                xnode.op_name, "softmax", attrs=attrs, input_ops=input_ops
            )

    @classmethod
    def to_softmax_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "softmax"

        # attributes of op
        # axis
        attr = xir_proto.AttrValue()
        attr.int32_value = -1
        node.op_attr["axis"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_concat(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: concat, op_name: {xnode.op_name}")
        # create input ops for conv2d
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["axis"] = xnode.axis
        logger.debug(f"attribute: axis: {attrs['axis']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "concat",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_concat_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "concat"

        # attributes of op
        # axis
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.axis
        node.op_attr["axis"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_squeeze(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: squeeze, op_name: {xnode.op_name}")
        # create input ops for conv2d
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["axis"] = xnode.axis
        logger.debug(f"attribute: axis: {attrs['axis']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "squeeze",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_squeeze_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "squeeze"

        # attributes of op
        # axis
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.axis)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["axis"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_reshape(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: reshape, op_name: {xnode.op_name}")
        # create input ops for conv2d
        input_ops = {}
        in_op: "xir.Op" = op_dict.get(xnode.bottom[0])
        assert in_op is not None, f"Not found op: name: {xnode.bottom[0]}"

        input_ops["input"] = [in_op]
        if len(xnode.bottom) == 2:
            # shape info as one input
            shape_op: "xir.Op" = op_dict.get(xnode.bottom[1])
            assert (
                shape_op is not None
            ), f"Not found op: name: {xnode.bottom[1]}. Current op type: {xnode.op_type}, name: {xnode.op_name}."
            input_ops["shape"] = [shape_op]

            return graph.create_op(
                xnode.op_name,
                "reshape",
                input_ops=input_ops,
            )
        else:
            # set attrs for op
            attrs: Dict[str, Any] = {}
            attrs["shape"] = xnode.shape
            logger.debug(f"attribute: shape: {attrs['shape']}")

            # create reshape op
            return graph.create_op(
                xnode.op_name,
                "reshape",
                attrs=attrs,
                input_ops=input_ops,
            )

    @classmethod
    def to_reshape_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: reshape, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "reshape"

        # arguments of op
        op_args = []

        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend([xnode.bottom[0]])
        op_args.append(op_arg)
        if len(xnode.bottom) == 2:
            # shape as an input
            op_arg = xir_proto.OpArg()
            op_arg.arg_name = "shape"
            op_arg.arg_ops.extend([xnode.bottom[1]])
            op_args.append(op_arg)
        node.args.extend(op_args)

        # attributes of op
        if len(xnode.bottom) == 1:
            # shape
            attr_value = xir_proto.Int32Vec()
            attr_value.value.extend(xnode.shape)
            attr = xir_proto.AttrValue()
            attr.int32_vec_value.CopyFrom(attr_value)
            node.op_attr["shape"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_batch_norm(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: batchnorm, op_name: {xnode.op_name}")
        # create input ops for conv2d
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create op
        return graph.create_op(
            xnode.op_name,
            "batchnorm",
            input_ops=input_ops,
        )

    @classmethod
    def to_batch_norm_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."

        logger.info(f"*** op_type: reshape, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "batchnorm"

        # attributes of op
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_pad(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: pad, op_name: {xnode.op_name}")
        # create input ops for conv2d
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        # paddings
        padding = xnode.padding
        if xnode.layout == "NCHW":
            padding = padding[:2] + padding[4:] + padding[2:4]
        attrs["paddings"] = padding
        logger.debug(f"attribute: paddings: {attrs['paddings']}")
        # mode
        attrs["mode"] = xnode.pad_mode.upper()
        logger.debug(f"attribute: mode: {attrs['mode']} ({xnode.pad_mode.lower()})")
        if xnode.pad_mode.upper() == "CONSTANT":
            if xnode.layout == "NHWC":
                constant_values = xnode.constant_values[2:6]
            else:
                constant_values = xnode.constant_values[-4:]
            attrs["constant_values"] = constant_values[2:] + constant_values[:2]
            logger.debug(
                f"attribute: constant_values (w_before, w_after, h_before, h_after): {attrs['constant_values']}"
            )

        # create op
        return graph.create_op(xnode.op_name, "pad", attrs=attrs, input_ops=input_ops)

    @classmethod
    def to_pad_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "pad"

        # attributes of op
        # mode
        attr = xir_proto.AttrValue()
        attr.string_value = xnode.pad_mode.upper()
        node.op_attr["mode"].CopyFrom(attr)

        # paddings (pad_N_before, pad_N_after, pad_H_before, pad_H_after, pad_W_before, pad_W_after, pad_C_before, pad_C_after)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.padding)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["paddings"].CopyFrom(attr)

        # constant_values (w_before, w_after, h_before, h_after)
        if xnode.pad_mode == "constant":
            constant_values = xnode.constant_values[2:6]
            constant_values = constant_values[2:] + constant_values[:2]
            attr_value = xir_proto.FloatVec()
            attr_value.value.extend(constant_values)
            attr = xir_proto.AttrValue()
            attr.float_vec_value.CopyFrom(attr_value)
            node.op_attr["constant_values"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args of op
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_mean(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: mean, op_name: {xnode.op_name}")
        # create input ops for Mean
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["keep_dims"] = xnode.keep_dims
        logger.debug(f"attribute: keep_dims: {attrs['keep_dims']}")
        attrs["axis"] = xnode.axis
        logger.debug(f"attribute: axis: {attrs['axis']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "reduction_mean",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_mean_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "reduction_mean"

        # attributes of op
        # axis
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.axis)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["axis"].CopyFrom(attr)
        # keep_dims
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.keep_dims
        node.op_attr["keep_dims"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args of op
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_const(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        # create const op for tensor
        const_op_name = xnode.op_name + "_const"
        logger.info(f"*** op_type: const, op_name: {const_op_name}")
        const_op: "xir.Op" = graph.create_const_op(
            const_op_name, xnode.tensor.to_numpy()
        )
        logger.info(f"attribute: data: {const_op.get_attr('data')}")
        # put into op_dict
        assert const_op_name not in op_dict, f"Duplicated op: op_name={const_op_name}."
        op_dict[const_op_name] = const_op

        if (
            xnode.quant_out["quantize_pos"] is not None
            and xnode.quant_out["bit_width"] is not None
        ):
            fix_op_name = const_op_name + "_fixneuron"
            logger.info(f"*** op_type: fix, op_name: {fix_op_name}")
            # create input ops
            input_ops = {"input": [const_op]}
            # set attrs for op
            attrs: Dict[str, Any] = {}
            attrs["fix_point"] = xnode.quant_out["quantize_pos"]
            logger.debug(f"attribute: fix_point: {attrs['fix_point']}")
            attrs["bit_width"] = xnode.quant_out["bit_width"]
            logger.debug(f"attribute: bit_width: {attrs['bit_width']}")
            attrs["if_signed"] = xnode.quant_out.get("signed")
            logger.debug(f"attribute: if_signed: {attrs['if_signed']}")
            attrs["round_mode"] = QuantRoundMode[xnode.quant_out.get("round_mode")]
            logger.debug(f"attribute: round_mode: {attrs['round_mode']}")
            # create FixedNeuron op
            op: "xir.Op" = graph.create_op(
                fix_op_name, "fix", attrs=attrs, input_ops=input_ops
            )
            return op
        elif (
            xnode.quant_out["quantize_pos"] is None
            and xnode.quant_out["bit_width"] is None
        ):
            return const_op
        else:
            raise ValueError(
                f"[Error] Invalid quantization info: xnode name: {xnode.op_name}, bit_width: {xnode.quant_out['bit_width']}, quantize_pos: {xnode.quant_out['quantize_pos']}."
            )

    @classmethod
    def to_const_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        const_node, fix_node = cls.__create_xir_const_node_from_xtensor(
            name=xnode.op_name, xtensor=xnode.tensor, quant_info=xnode.quant_in
        )
        assert (
            const_node is not None
        ), f"[ERROR] Failed to create xir const node from: op name: {xnode.op_name}."
        op_dict[const_node.op_name] = const_node

        if fix_node:
            op_dict[fix_node.op_name] = fix_node

    @classmethod
    def to_shape(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: shape, op_name: {xnode.op_name}")
        # create input ops for Mean
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create op
        return graph.create_op(xnode.op_name, "shape", input_ops=input_ops)

    @classmethod
    def to_shape_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "shape"

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args of op
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_permute(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["order"] = xnode.order
        logger.debug(f"attribute: order: {attrs['order']}")

        # create op
        return graph.create_op(
            xnode.op_name, "transpose", attrs=attrs, input_ops=input_ops
        )

    @classmethod
    def to_permute_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "transpose"

        # attributes of op
        # order
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.order)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["order"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args of op
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_priorbox(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: priorbox, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        if xnode.min_sizes is not None and len(xnode.min_sizes) > 0:
            attrs["min_sizes"] = xnode.min_sizes
            logger.debug(f"attribute: min_sizes: {attrs['min_sizes']}")
        if xnode.max_sizes is not None and len(xnode.max_sizes) > 0:
            attrs["max_sizes"] = xnode.max_sizes
            logger.debug(f"attribute: max_sizes: {attrs['max_sizes']}")
        if xnode.aspect_ratio is not None and len(xnode.aspect_ratio) > 0:
            attrs["aspect_ratio"] = xnode.aspect_ratio
            logger.debug(f"attribute: aspect_ratio: {attrs['aspect_ratio']}")
        attrs["flip"] = xnode.flip
        logger.debug(f"attribute: flip: {attrs['flip']}")
        attrs["clip"] = xnode.clip
        logger.debug(f"attribute: clip: {attrs['clip']}")
        if xnode.variance is not None and len(xnode.variance) > 0:
            attrs["variance"] = xnode.variance
            logger.debug(f"attribute: variance: {attrs['variance']}")
        if xnode.step is not None and len(xnode.step) > 0:
            attrs["step"] = xnode.step
            logger.debug(f"attribute: step: {attrs['step']}")
        attrs["offset"] = xnode.offset
        logger.debug(f"attribute: offset: {attrs['offset']}")

        # create op
        return graph.create_op(
            xnode.op_name, "priorbox", attrs=attrs, input_ops=input_ops
        )

    @classmethod
    def to_priorbox_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "priorbox"

        # attributes of op
        # flip
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.flip
        node.op_attr["flip"].CopyFrom(attr)
        # clip
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.clip
        node.op_attr["clip"].CopyFrom(attr)
        # offset
        attr = xir_proto.AttrValue()
        attr.float_value = xnode.offset
        node.op_attr["offset"].CopyFrom(attr)
        # min_sizes
        if xnode.min_sizes is not None and len(xnode.min_sizes) > 0:
            attr_value = xir_proto.FloatVec()
            attr_value.value.extend(xnode.min_sizes)
            attr = xir_proto.AttrValue()
            attr.float_vec_value.CopyFrom(attr_value)
            node.op_attr["min_sizes"].CopyFrom(attr)
        # max_sizes
        if xnode.max_sizes is not None and len(xnode.max_sizes) > 0:
            attr_value = xir_proto.FloatVec()
            attr_value.value.extend(xnode.max_sizes)
            attr = xir_proto.AttrValue()
            attr.float_vec_value.CopyFrom(attr_value)
            node.op_attr["max_sizes"].CopyFrom(attr)
        # aspect_ratio
        if xnode.aspect_ratio is not None and len(xnode.aspect_ratio) > 0:
            attr_value = xir_proto.FloatVec()
            attr_value.value.extend(xnode.aspect_ratio)
            attr = xir_proto.AttrValue()
            attr.float_vec_value.CopyFrom(attr_value)
            node.op_attr["aspect_ratio"].CopyFrom(attr)
        # variance
        if xnode.variance is not None and len(xnode.variance) > 0:
            attr_value = xir_proto.FloatVec()
            attr_value.value.extend(xnode.variance)
            attr = xir_proto.AttrValue()
            attr.float_vec_value.CopyFrom(attr_value)
            node.op_attr["variance"].CopyFrom(attr)
        # step
        if xnode.step is not None and len(xnode.step) > 0:
            attr_value = xir_proto.FloatVec()
            attr_value.value.extend(xnode.step)
            attr = xir_proto.AttrValue()
            attr.float_vec_value.CopyFrom(attr_value)
            node.op_attr["step"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args of op
        assert len(xnode.bottom) > 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_flatten(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: flatten, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["start_axis"] = xnode.start_dim
        logger.debug(f"attribute: start_axis: {attrs['start_axis']}")
        attrs["end_axis"] = xnode.end_dim
        logger.debug(f"attribute: end_axis: {attrs['end_axis']}")

        # create op
        return graph.create_op(
            xnode.op_name, "flatten", attrs=attrs, input_ops=input_ops
        )

    @classmethod
    def to_flatten_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "flatten"

        # attributes of op
        # start_axis
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.start_dim
        node.op_attr["start_axis"].CopyFrom(attr)
        # end_axis
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.end_dim
        node.op_attr["end_axis"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args of op
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_stack(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: stack, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["axis"] = xnode.axis
        logger.debug(f"attribute: axis: {attrs['axis']}")

        # Due to xir's implementation of the 'stack' op, if the input op does not have a 'data' attribute, it is necessary to set the 'shape_info' information of the input tensor.
        for xir_op in input_ops['input']:
            if 'data' not in xir_op.get_attrs():
                str_input_shape_data = xir_op.get_output_tensor().__str__().split(', shape: [')[1]
                index = str_input_shape_data.index(']')
                input_shape_data = [int(i) for i in (str_input_shape_data[:index].split(','))]
                xir_op.get_output_tensor().set_attr('shape_info', input_shape_data)

        # create op
        return graph.create_op(xnode.op_name, "stack", attrs=attrs, input_ops=input_ops)

    @classmethod
    def to_stack_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "stack"

        # attributes of op
        # axis
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.axis
        node.op_attr["axis"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args of op
        assert len(xnode.bottom) > 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_matmul(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: matmul, op_name: {xnode.op_name}")
        # create input ops for matmul
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create const op for bias
        bias, fix_op_bias = None, None
        if xnode.bias is not None:
            # set out_tensor
            out_tensor_b = xnode.bias
            logger.debug(
                f"attribute: bias: shape: {out_tensor_b.shape}, dtype: {out_tensor_b.dtype}"
            )
            const_op_name = xnode.op_name + "_bias"
            bias = graph.create_const_op(const_op_name, out_tensor_b.to_numpy())
            # put into op_dict
            assert (
                const_op_name not in op_dict
            ), f"Duplicated op: op_name={const_op_name}."
            op_dict[const_op_name] = bias

            # create FixedNeuron op for bias
            fix_op_name = const_op_name + "_fixneuron"
            # dtype
            dtype = out_tensor_b.dtype
            # shape
            shape = out_tensor_b.shape
            out_tensor = np.zeros(shape, dtype=dtype)
            fix_op_bias = cls.__create_fixneuron(
                fix_op_name,
                [const_op_name],
                xnode.quant_bias,
                out_tensor,
                graph,
                op_dict,
            )
            input_ops["bias"] = [fix_op_bias]

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["transpose_a"] = xnode.transpose_a
        attrs["transpose_b"] = xnode.transpose_b
        logger.debug(
            f"attribute: transpose_a: {attrs['transpose_a']}, transpose_b: {attrs['transpose_b']}"
        )

        # create op
        return graph.create_op(
            xnode.op_name,
            "matmul",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_matmul_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "matmul"

        # attributes of op
        # transpose_a
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.transpose_a
        node.op_attr["transpose_a"].CopyFrom(attr)
        # transpose_b
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.transpose_b
        node.op_attr["transpose_b"].CopyFrom(attr)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)

        # bias
        if xnode.bias is not None:
            const_node_b, fix_node_b = cls.__create_xir_const_node_from_xtensor(
                name=xnode.op_name + "_bias",
                xtensor=xnode.bias,
                quant_info=xnode.quant_bias,
            )
            assert (
                const_node_b.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={const_node_b.op_name}."
            op_dict[const_node_b.op_name] = const_node_b
            if fix_node_b:
                assert (
                    fix_node_b.op_name not in op_dict
                ), f"[ERROR] Found duplicate xir op: op_name={fix_node_b.op_name}."
                op_dict[fix_node_b.op_name] = fix_node_b

            op_arg = xir_proto.OpArg()
            op_arg.arg_name = "bias"
            if fix_node_b:
                op_arg.arg_ops.extend([fix_node_b.op_name])
            else:
                op_arg.arg_ops.extend([const_node_b.op_name])
            op_args.append(op_arg)

        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_fixneuron(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: fixneuron, op_name: {xnode.op_name}")
        # create input ops for conv2d
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["fix_point"] = xnode.quant_in.get("quantize_pos")
        logger.debug(f"attribute: fix_point: {attrs['fix_point']}")
        attrs["bit_width"] = xnode.quant_in.get("bit_width")
        logger.debug(f"attribute: bit_width: {attrs['bit_width']}")
        attrs["if_signed"] = xnode.quant_in.get("signed")
        logger.debug(f"attribute: if_signed: {attrs['if_signed']}")
        attrs["round_mode"] = QuantRoundMode[xnode.quant_in.get("round_mode")]
        logger.debug(f"attribute: round_mode: {attrs['round_mode']}")

        # create FixNeuron op
        return graph.create_op(
            xnode.op_name,
            "fix",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_fixneuron_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        fix_node = xir_proto.OPNode()
        fix_node.op_name = xnode.op_name
        fix_node.op_type = "fix"

        # attributes of op
        # bit_width
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.quant_in.get("bit_width")
        fix_node.op_attr["bit_width"].CopyFrom(attr)
        fix_node.op_attr["quant_in_bit_width"].CopyFrom(attr)
        fix_node.op_attr["quant_out_bit_width"].CopyFrom(attr)

        # quantize_pos
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.quant_in.get("quantize_pos")
        fix_node.op_attr["fix_point"].CopyFrom(attr)
        fix_node.op_attr["quant_in_quantize_pos"].CopyFrom(attr)
        fix_node.op_attr["quant_out_quantize_pos"].CopyFrom(attr)

        # round_mode
        attr = xir_proto.AttrValue()
        attr.string_value = QuantRoundMode[xnode.quant_in.get("round_mode")]
        fix_node.op_attr["round_mode"].CopyFrom(attr)
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.quant_in.get("round_mode")
        fix_node.op_attr["quant_in_round_mode"].CopyFrom(attr)
        fix_node.op_attr["quant_out_round_mode"].CopyFrom(attr)

        # quant_in_signed
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.quant_in.get("signed")
        fix_node.op_attr["if_signed"].CopyFrom(attr)
        fix_node.op_attr["quant_in_signed"].CopyFrom(attr)
        fix_node.op_attr["quant_out_signed"].CopyFrom(attr)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        fix_node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            fix_node.op_name, xnode.outputs_tensor[0]
        )
        fix_node.output_tensor.CopyFrom(output_tensor)

        op_dict[fix_node.op_name] = fix_node

    @classmethod
    def to_identity(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: identity, op_name: {xnode.op_name}")
        # create input ops for identity
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create Identity op
        return graph.create_op(
            xnode.op_name,
            "identity",
            input_ops=input_ops,
        )

    @classmethod
    def to_identity_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "identity"

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args of op
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_deconvolution(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")
        # convert (H, W, in_ch, out_ch) into (out_ch, H, W, in_ch)
        out_tensor_w = tc.transpose(xnode.weights, (3, 0, 1, 2))
        logger.debug(
            f"attribute: weights: shape (out_channels, out_height, out_width, in_channels): {out_tensor_w.shape}, dtype: {out_tensor_w.dtype}"
        )
        # create const op for weights
        const_op_name = xnode.op_name + "_weights"
        weights = graph.create_const_op(const_op_name, out_tensor_w.to_numpy())
        # put into op_dict
        assert const_op_name not in op_dict, f"Duplicated op: op_name={const_op_name}."
        op_dict[const_op_name] = weights

        # create FixedNeuron op for weights
        fix_op_name = const_op_name + "_fixneuron"
        # dtype
        dtype = out_tensor_w.dtype
        # shape
        shape = out_tensor_w.shape
        out_tensor = np.zeros(shape, dtype=dtype)
        fix_op_weights = cls.__create_fixneuron(
            fix_op_name,
            [const_op_name],
            xnode.quant_weights,
            out_tensor,
            graph,
            op_dict,
        )

        # create const op for bias
        bias, fix_op_bias = None, None
        if hasattr(xnode, "bias_term"):
            if xnode.bias_term:
                # set out_tensor
                out_tensor_b = xnode.bias
                logger.debug(
                    f"attribute: bias: shape: {out_tensor_b.shape}, dtype: {out_tensor_b.dtype}"
                )
                const_op_name = xnode.op_name + "_bias"
                bias = graph.create_const_op(const_op_name, out_tensor_b.to_numpy())
                # put into op_dict
                assert (
                    const_op_name not in op_dict
                ), f"Duplicated op: op_name={const_op_name}."
                op_dict[const_op_name] = bias

                # create FixedNeuron op for bias
                fix_op_name = const_op_name + "_fixneuron"
                # dtype
                dtype = out_tensor_b.dtype
                # shape
                shape = out_tensor_b.shape
                out_tensor = np.zeros(shape, dtype=dtype)
                fix_op_bias = cls.__create_fixneuron(
                    fix_op_name,
                    [const_op_name],
                    xnode.quant_bias,
                    out_tensor,
                    graph,
                    op_dict,
                )

        # create input ops for conv2d
        input_ops = {}
        input_ops["weights"] = [fix_op_weights]
        if xnode.bias_term:
            input_ops["bias"] = [fix_op_bias]
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["kernel"] = xnode.kernel_size[::-1]
        logger.debug(f"attribute: (kernel_w, kernel_h): {attrs['kernel']}")
        attrs["dilation"] = xnode.dilation[-2:][::-1]
        logger.debug(f"attribute: (dilation_w, dilation_h): {attrs['dilation']}")
        attrs["stride"] = xnode.strides[::-1]
        logger.debug(f"attribute: (stride_w, stride_h): {attrs['stride']}")
        if xnode.pad_mode == PadMode.EXPLICIT:
            assert xnode.round_mode in [
                RoundMode.FLOOR,
                RoundMode.CEIL,
                RoundMode.ROUND_DOWN,
                RoundMode.ROUND_UP,
            ]
            attrs["pad_mode"] = (
                RoundMode.FLOOR.name
                if xnode.round_mode in [RoundMode.FLOOR, RoundMode.ROUND_DOWN]
                else RoundMode.CEIL.name
            )
            attrs["pad"] = xnode.padding[2:] + xnode.padding[:2]
        elif xnode.pad_mode == PadMode.SAME:
            attrs["pad_mode"] = PadMode.SAME.name
        elif xnode.pad_mode == PadMode.VALID:
            attrs["pad_mode"] = PadMode.VALID.name
        else:
            raise ValueError(f"Unsupported pad mode: {xnode.pad_mode}.")
        logger.debug(f"attribute: pad_mode: {attrs['pad_mode']}")
        if "pad" in attrs:
            logger.debug(f"attribute: pad (left, right, top, bottom): {attrs['pad']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "transposed-conv2d",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_deconvolution_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "transposed-conv2d"

        # attributes of op
        # (kernel_w, kernel_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.kernel_size[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["kernel"].CopyFrom(attr)
        # (stride_w, stride_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.strides[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["stride"].CopyFrom(attr)
        # (dilation_w, dilation_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.dilation[-2:][::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["dilation"].CopyFrom(attr)
        # pad_mode
        attr_value = xir_proto.AttrValue()
        attr_value.string_value = (
            xnode.round_mode.name
            if xnode.pad_mode == PadMode.EXPLICIT
            else xnode.pad_mode.name
        )
        node.op_attr["pad_mode"].CopyFrom(attr_value)
        # pad (pad_w_before, pad_w_after, pad_h_before, pad_h_after)
        if xnode.pad_mode == PadMode.EXPLICIT:
            attr_value = xir_proto.Int32Vec()
            attr_value.value.extend(xnode.padding[2:] + xnode.padding[:2])
            attr = xir_proto.AttrValue()
            attr.int32_vec_value.CopyFrom(attr_value)
            node.op_attr["pad"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)

        # convert (h,w,ic,oc) into (oc,h,w,ic)
        weights = tc.transpose(xnode.weights, (3, 0, 1, 2))
        const_node_w, fix_node_w = cls.__create_xir_const_node_from_xtensor(
            name=xnode.op_name + "_weights",
            xtensor=weights,
            quant_info=xnode.quant_weights,
        )
        assert (
            const_node_w.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={const_node_w.op_name}."
        op_dict[const_node_w.op_name] = const_node_w
        if fix_node_w:
            assert (
                fix_node_w.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={fix_node_w.op_name}."
            op_dict[fix_node_w.op_name] = fix_node_w

        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "weights"
        if fix_node_w:
            op_arg.arg_ops.extend([fix_node_w.op_name])
        else:
            op_arg.arg_ops.extend([const_node_w.op_name])
        op_args.append(op_arg)

        # bias
        if xnode.bias_term:
            const_node_b, fix_node_b = cls.__create_xir_const_node_from_xtensor(
                name=xnode.op_name + "_bias",
                xtensor=xnode.bias,
                quant_info=xnode.quant_bias,
            )
            assert (
                const_node_b.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={const_node_b.op_name}."
            op_dict[const_node_b.op_name] = const_node_b
            if fix_node_b:
                assert (
                    fix_node_b.op_name not in op_dict
                ), f"[ERROR] Found duplicate xir op: op_name={fix_node_b.op_name}."
                op_dict[fix_node_b.op_name] = fix_node_b

            op_arg = xir_proto.OpArg()
            op_arg.arg_name = "bias"
            if fix_node_b:
                op_arg.arg_ops.extend([fix_node_b.op_name])
            else:
                op_arg.arg_ops.extend([const_node_b.op_name])
            op_args.append(op_arg)

        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_depthwise_deconvolution(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")
        # convert (H, W, in_ch, out_ch) into (out_ch, H, W, in_ch)
        out_tensor_w = tc.transpose(xnode.weights, (3, 0, 1, 2))
        logger.debug(
            f"attribute: weights: shape (out_channels, out_height, out_width, in_channels): {out_tensor_w.shape}, dtype: {out_tensor_w.dtype}"
        )
        # create const op for weights
        const_op_name = xnode.op_name + "_weights"
        weights = graph.create_const_op(const_op_name, out_tensor_w.to_numpy())
        # put into op_dict
        assert const_op_name not in op_dict, f"Duplicated op: op_name={const_op_name}."
        op_dict[const_op_name] = weights

        # create FixedNeuron op for weights
        fix_op_name = const_op_name + "_fixneuron"
        # dtype
        dtype = out_tensor_w.dtype
        # shape
        shape = out_tensor_w.shape
        out_tensor = np.zeros(shape, dtype=dtype)
        fix_op_weights = cls.__create_fixneuron(
            fix_op_name,
            [const_op_name],
            xnode.quant_weights,
            out_tensor,
            graph,
            op_dict,
        )

        # create const op for bias
        bias, fix_op_bias = None, None
        if hasattr(xnode, "bias_term"):
            if xnode.bias_term:
                # set out_tensor
                out_tensor_b = xnode.bias
                logger.debug(
                    f"attribute: bias: shape: {out_tensor_b.shape}, dtype: {out_tensor_b.dtype}"
                )
                const_op_name = xnode.op_name + "_bias"
                bias = graph.create_const_op(const_op_name, out_tensor_b.to_numpy())
                # put into op_dict
                assert (
                    const_op_name not in op_dict
                ), f"Duplicated op: op_name={const_op_name}."
                op_dict[const_op_name] = bias

                # create FixedNeuron op for bias
                fix_op_name = const_op_name + "_fixneuron"
                # dtype
                dtype = out_tensor_b.dtype
                # shape
                shape = out_tensor_b.shape
                out_tensor = np.zeros(shape, dtype=dtype)
                fix_op_bias = cls.__create_fixneuron(
                    fix_op_name,
                    [const_op_name],
                    xnode.quant_bias,
                    out_tensor,
                    graph,
                    op_dict,
                )

        # create input ops for conv2d
        input_ops = {}
        input_ops["weights"] = [fix_op_weights]
        if xnode.bias_term:
            input_ops["bias"] = [fix_op_bias]
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["kernel"] = xnode.kernel_size[::-1]
        logger.debug(f"attribute: (kernel_w, kernel_h): {attrs['kernel']}")
        # attrs["dilation"] = xnode.dilation[-2:][::-1]
        # logger.debug(f"attribute: (dilation_w, dilation_h): {attrs['dilation']}")
        attrs["stride"] = xnode.strides[::-1]
        logger.debug(f"attribute: (stride_w, stride_h): {attrs['stride']}")
        if xnode.pad_mode == PadMode.EXPLICIT:
            assert xnode.round_mode in [
                RoundMode.FLOOR,
                RoundMode.CEIL,
                RoundMode.ROUND_DOWN,
                RoundMode.ROUND_UP,
            ]
            attrs["pad_mode"] = (
                RoundMode.FLOOR.name
                if xnode.round_mode in [RoundMode.FLOOR, RoundMode.ROUND_DOWN]
                else RoundMode.CEIL.name
            )
            attrs["pad"] = xnode.padding[2:] + xnode.padding[:2]
        elif xnode.pad_mode == PadMode.SAME:
            attrs["pad_mode"] = PadMode.SAME.name
        elif xnode.pad_mode == PadMode.VALID:
            attrs["pad_mode"] = PadMode.VALID.name
        else:
            raise ValueError(f"Unsupported pad mode: {xnode.pad_mode}.")
        logger.debug(f"attribute: pad_mode: {attrs['pad_mode']}")
        if "pad" in attrs:
            logger.debug(f"attribute: pad (left, right, top, bottom): {attrs['pad']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "transposed-depthwise-conv2d",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_depthwise_deconvolution_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "transposed-depthwise-conv2d"

        # attributes of op
        # (kernel_w, kernel_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.kernel_size[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["kernel"].CopyFrom(attr)
        # (stride_w, stride_h)
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.strides[::-1])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["stride"].CopyFrom(attr)

        # # (dilation_w, dilation_h)
        # attr_value = xir_proto.Int32Vec()
        # attr_value.value.extend(xnode.dilation[-2:][::-1])
        # attr = xir_proto.AttrValue()
        # attr.int32_vec_value.CopyFrom(attr_value)
        # node.op_attr["dilation"].CopyFrom(attr)

        # pad_mode
        attr_value = xir_proto.AttrValue()
        attr_value.string_value = (
            xnode.round_mode.name
            if xnode.pad_mode == PadMode.EXPLICIT
            else xnode.pad_mode.name
        )
        node.op_attr["pad_mode"].CopyFrom(attr_value)
        # pad (pad_w_before, pad_w_after, pad_h_before, pad_h_after)
        if xnode.pad_mode == PadMode.EXPLICIT:
            attr_value = xir_proto.Int32Vec()
            attr_value.value.extend(xnode.padding[2:] + xnode.padding[:2])
            attr = xir_proto.AttrValue()
            attr.int32_vec_value.CopyFrom(attr_value)
            node.op_attr["pad"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)

        # convert (h,w,ic,oc) into (oc,h,w,ic)
        weights = tc.transpose(xnode.weights, (3, 0, 1, 2))
        const_node_w, fix_node_w = cls.__create_xir_const_node_from_xtensor(
            name=xnode.op_name + "_weights",
            xtensor=weights,
            quant_info=xnode.quant_weights,
        )
        assert (
            const_node_w.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={const_node_w.op_name}."
        op_dict[const_node_w.op_name] = const_node_w
        if fix_node_w:
            assert (
                fix_node_w.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={fix_node_w.op_name}."
            op_dict[fix_node_w.op_name] = fix_node_w

        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "weights"
        if fix_node_w:
            op_arg.arg_ops.extend([fix_node_w.op_name])
        else:
            op_arg.arg_ops.extend([const_node_w.op_name])
        op_args.append(op_arg)

        # bias
        if xnode.bias_term:
            const_node_b, fix_node_b = cls.__create_xir_const_node_from_xtensor(
                name=xnode.op_name + "_bias",
                xtensor=xnode.bias,
                quant_info=xnode.quant_bias,
            )
            assert (
                const_node_b.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={const_node_b.op_name}."
            op_dict[const_node_b.op_name] = const_node_b
            if fix_node_b:
                assert (
                    fix_node_b.op_name not in op_dict
                ), f"[ERROR] Found duplicate xir op: op_name={fix_node_b.op_name}."
                op_dict[fix_node_b.op_name] = fix_node_b

            op_arg = xir_proto.OpArg()
            op_arg.arg_name = "bias"
            if fix_node_b:
                op_arg.arg_ops.extend([fix_node_b.op_name])
            else:
                op_arg.arg_ops.extend([const_node_b.op_name])
            op_args.append(op_arg)

        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        op_dict[node.op_name] = node

    @classmethod
    def to_gstiling(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops for gstiling
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["reverse"] = xnode.reverse
        logger.debug(f"attribute: reverse: {attrs['reverse']}")
        attrs["stride"] = xnode.stride
        logger.debug(f"attribute: stride: {attrs['stride']}")

        # create gstiling op
        return graph.create_op(
            xnode.op_name,
            "gstiling",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_gstiling_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "gstiling"

        # attributes of op
        # reverse
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.reverse
        node.op_attr["reverse"].CopyFrom(attr)
        # stride
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.stride
        node.op_attr["stride"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_reorg(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: reorg, op_name: {xnode.op_name}")
        # create input ops for reorg
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        assert (
            xnode.stride[0] == xnode.stride[1]
        ), f"[ERROR] Mismatched stride values: {xnode.stride}"
        attrs["scale"] = xnode.stride[0]
        attrs["reverse"] = xnode.reverse
        logger.debug(f"attribute: scale: {attrs['scale']}")
        logger.debug(f"attribute: reverse: {attrs['reverse']}")

        # create reorg op
        return graph.create_op(
            xnode.op_name,
            "reorg",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_reorg_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "reorg"

        # attributes of op
        # reverse
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.reverse
        node.op_attr["reverse"].CopyFrom(attr)
        # scale
        assert xnode.stride[0] == xnode.stride[1]
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.stride[0]
        node.op_attr["scale"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_deephiresize(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: deephiresize, op_name: {xnode.op_name}")
        # create input ops for deephiresize
        input_ops = {}
        input_ops["input"] = [op_dict.get(xnode.bottom[0])]

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["scale"] = xnode.scale
        logger.debug(f"attribute: scale: {attrs['scale']}")
        # 0: opencv-nearest, 1: opencv-bilinear
        attrs["mode"] = xnode.mode.upper()
        logger.debug(f"attribute: mode: {attrs['mode']}")
        if attrs["mode"] == "BILINEAR":
            attrs["half_pixel_centers"] = True

        # create deephiresize op
        return graph.create_op(
            xnode.op_name,
            "resize",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_deephiresize_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "resize"

        # attributes of op
        # scale
        attr_value = xir_proto.FloatVec()
        attr_value.value.extend(xnode.scale[::-1])
        attr = xir_proto.AttrValue()
        attr.float_vec_value.CopyFrom(attr_value)
        node.op_attr["scale"].CopyFrom(attr)
        # mode
        attr = xir_proto.AttrValue()
        attr.string_value = xnode.mode.upper()
        node.op_attr["mode"].CopyFrom(attr)
        # half_pixel_centers
        attr = xir_proto.AttrValue()
        attr.bool_value = False
        if xnode.mode.upper() == "BILINEAR":
            attr.bool_value = True
        node.op_attr["half_pixel_centers"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_resize(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: resize, op_name: {xnode.op_name}")
        # create input ops for ResizeNearestNeighbor
        input_ops = {}
        input_ops["input"] = [op_dict.get(xnode.bottom[0])]
        input_ops["size"] = [op_dict.get(xnode.bottom[1])]

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["mode"] = xnode.mode.upper()
        logger.debug(f"attribute: mode: {attrs['mode']}")
        attrs["align_corners"] = xnode.align_corners
        logger.debug(f"attribute: align_corners: {attrs['align_corners']}")
        attrs["half_pixel_centers"] = xnode.half_pixel_centers
        logger.debug(f"attribute: half_pixel_centers: {attrs['half_pixel_centers']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "resize",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_resize_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "resize"

        # attributes of op
        # mode
        attr = xir_proto.AttrValue()
        attr.string_value = xnode.mode.upper()
        node.op_attr["mode"].CopyFrom(attr)
        # align_corners
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.align_corners
        node.op_attr["align_corners"].CopyFrom(attr)
        # half_pixel_centers
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.half_pixel_centers
        node.op_attr["half_pixel_centers"].CopyFrom(attr)
        # scale
        attr_value = xir_proto.FloatVec()
        attr_value.value.extend([1.0, 1.0])
        attr = xir_proto.AttrValue()
        attr.float_vec_value.CopyFrom(attr_value)
        node.op_attr["scale"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args of op
        assert len(xnode.bottom) == 2
        op_arg_input = xir_proto.OpArg()
        op_arg_input.arg_name = "input"
        op_arg_input.arg_ops.extend([xnode.bottom[0]])
        op_arg_size = xir_proto.OpArg()
        op_arg_size.arg_name = "size"
        op_arg_size.arg_ops.extend([xnode.bottom[1]])
        node.args.extend([op_arg_input, op_arg_size])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_linear(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")
        # set out_tensor: (out_features, in_features)
        out_tensor_w = xnode.weights
        logger.debug(
            f"attribute: weights: shape (out_channels, out_height, out_width, in_channels): {out_tensor_w.shape}, dtype: {out_tensor_w.dtype}"
        )
        # create const op for weights
        const_op_name = xnode.op_name + "_weights"
        weights = graph.create_const_op(const_op_name, out_tensor_w.to_numpy())
        # put into op_dict
        assert const_op_name not in op_dict, f"Duplicated op: op_name={const_op_name}."
        op_dict[const_op_name] = weights

        if (
            xnode.quant_weights["quantize_pos"] is not None
            and xnode.quant_weights["bit_width"] is not None
        ):
            # create FixedNeuron op for weights
            fix_op_name = const_op_name + "_fixneuron"
            fix_op_weights = cls.__create_fixneuron(
                fix_op_name,
                [const_op_name],
                xnode.quant_weights,
                graph,
                op_dict,
            )

        # create const op for bias
        bias, fix_op_bias = None, None
        if xnode.bias is not None:
            # set out_tensor (out_features)
            out_tensor_b = xnode.bias
            logger.debug(
                f"attribute: bias: shape: {out_tensor_b.shape}, dtype: {out_tensor_b.dtype}"
            )
            const_op_name = xnode.op_name + "_bias"
            bias = graph.create_const_op(const_op_name, out_tensor_b.to_numpy())
            # put into op_dict
            assert (
                const_op_name not in op_dict
            ), f"Duplicated op: op_name={const_op_name}."
            op_dict[const_op_name] = bias

            if (
                xnode.quant_bias["quantize_pos"] is not None
                and xnode.quant_bias["bit_width"] is not None
            ):
                # create FixedNeuron op for bias
                fix_op_name = const_op_name + "_fixneuron"
                fix_op_bias = cls.__create_fixneuron(
                    fix_op_name,
                    [const_op_name],
                    xnode.quant_bias,
                    graph,
                    op_dict,
                )

        # create input ops for conv2d
        input_ops = {}
        input_ops["weights"] = (
            [fix_op_weights]
            if xnode.quant_weights["quantize_pos"] is not None
            and xnode.quant_weights["bit_width"] is not None
            else [weights]
        )
        if xnode.bias is not None:
            input_ops["bias"] = (
                [fix_op_bias]
                if xnode.quant_bias["quantize_pos"] is not None
                and xnode.quant_bias["bit_width"] is not None
                else [bias]
            )
        input_ops["input"] = []
        if xnode.bottom is not None and len(xnode.bottom) > 0:
            for in_op_name in xnode.bottom:
                input_op = op_dict.get(in_op_name)
                assert (
                    input_op is not None
                ), f"[ERROR] input op is None: op name: {xnode.op_name}"
                input_ops["input"].append(input_op)

        # create op
        return graph.create_op(
            xnode.op_name,
            "linear",
            input_ops=input_ops,
        )

    @classmethod
    def to_linear_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "linear"

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # arguments of op
        op_args = []
        # input
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        op_args.append(op_arg)

        # weights: (out_features, in_features)
        assert len(xnode.weights.shape) == 2
        weights = xnode.weights
        const_node_w, fix_node_w = cls.__create_xir_const_node_from_xtensor(
            name=xnode.op_name + "_weights",
            xtensor=weights,
            quant_info=xnode.quant_weights,
        )
        assert (
            const_node_w.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={const_node_w.op_name}."
        op_dict[const_node_w.op_name] = const_node_w
        if fix_node_w:
            assert (
                fix_node_w.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={fix_node_w.op_name}."
            op_dict[fix_node_w.op_name] = fix_node_w

        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "weights"
        if fix_node_w:
            op_arg.arg_ops.extend([fix_node_w.op_name])
        else:
            op_arg.arg_ops.extend([const_node_w.op_name])
        op_args.append(op_arg)

        # bias
        if xnode.bias_term:
            const_node_b, fix_node_b = cls.__create_xir_const_node_from_xtensor(
                name=xnode.op_name + "_bias",
                xtensor=xnode.bias,
                quant_info=xnode.quant_bias,
            )
            assert (
                const_node_b.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={const_node_b.op_name}."
            op_dict[const_node_b.op_name] = const_node_b
            if fix_node_b:
                assert (
                    fix_node_b.op_name not in op_dict
                ), f"[ERROR] Found duplicate xir op: op_name={fix_node_b.op_name}."
                op_dict[fix_node_b.op_name] = fix_node_b

            op_arg = xir_proto.OpArg()
            op_arg.arg_name = "bias"
            if fix_node_b:
                op_arg.arg_ops.extend([fix_node_b.op_name])
            else:
                op_arg.arg_ops.extend([const_node_b.op_name])
            op_args.append(op_arg)

        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_batchnorm2d(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops for conv2d
        input_ops = {}
        input_ops["input"] = []
        if xnode.bottom is not None and len(xnode.bottom) > 0:
            for in_op_name in xnode.bottom:
                input_op = op_dict.get(in_op_name)
                assert (
                    input_op is not None
                ), f"[ERROR] input op is None: op name: {xnode.op_name}"
                input_ops["input"].append(input_op)

        # set attrs for op
        attrs: Dict[str, Any] = {}
        # ! hardcode
        attrs["size"] = xnode.outputs_tensor_shape[0]

        # create op
        return graph.create_op(
            xnode.op_name,
            "batchnorm",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_batchnorm2d_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "resize"

        # attributes of op
        # size
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.outputs_tensor_shape[0])
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["size"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_adaptiveavgpoolnd(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops for deephiresize
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["output_size"] = xnode.out_size
        logger.debug(f"attribute: output_size: {attrs['output_size']}")

        # create adaptiveavgpool op
        return graph.create_op(
            xnode.op_name,
            "adaptive_avgpool_nd",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_adaptiveavgpoolnd_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "adaptive_avgpool_nd"

        # attributes of op
        # size
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.out_size)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["output_size"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_size(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops for size
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["dims"] = xnode.dims
        logger.debug(f"attribute: dims: {attrs['dims']}")

        # create size op
        return graph.create_op(
            xnode.op_name,
            "size",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_size_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "size"

        # attributes of op
        # dims
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.dims)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["dims"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_upsample(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        """upsample will be converted into xir resize"""

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops for upsample
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        # scale: (scale_w, scale_h)
        assert xnode.scale[:2] == [
            1,
            1,
        ], f"[ERROR] The scale values in the axes of N and C must be 1: actual: {xnode.scale[:2]}"
        if xnode.host.origin == "pytorch":
            # compute reciprocal of each element
            attrs["scale"] = [1 / x for x in xnode.scale[-2:][::-1]]
        elif xnode.host.origin == "tensorflow2":
            attrs["scale"] = xnode.scale[-2:][::-1]
        else:
            raise ValueError(f"[ERROR] unsupported model type: {xnode.host.origin}")
        logger.debug(f"attribute: scale: {attrs['scale']}")
        # mode: only support "nearest" and "bilinear"
        attrs["mode"] = xnode.mode.upper()
        logger.debug(f"attribute: mode: {attrs['mode']}")
        attrs["align_corners"] = xnode.align_corners
        logger.debug(f"attribute: align_corners: {attrs['align_corners']}")
        attrs["half_pixel_centers"]=True

        # create resize op
        return graph.create_op(
            xnode.op_name,
            "resize",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_upsample_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "resize"

        # attributes of op
        # scale: (scale_w, scale_h)
        assert xnode.scale[:2] == [
            1,
            1,
        ], f"[ERROR] The scale values in the axes of N and C must be 1: actual: {xnode.scale[:2]}"
        attr_value = xir_proto.FloatVec()
        if xnode.host.origin == "pytorch":
            # compute reciprocal of each element
            attr_value.value.extend([1 / x for x in xnode.scale[-2:][::-1]])
        elif xnode.host.origin == "tensorflow2":
            attr_value.value.extend(xnode.scale[-2:][::-1])
        else:
            raise ValueError(f"[ERROR] unsupported model type: {xnode.host.origin}")
        attr = xir_proto.AttrValue()
        attr.float_vec_value.CopyFrom(attr_value)
        node.op_attr["scale"].CopyFrom(attr)
        # mode
        attr = xir_proto.AttrValue()
        attr.string_value = xnode.mode.upper()
        node.op_attr["mode"].CopyFrom(attr)
        # align_corners
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.align_corners
        node.op_attr["align_corners"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_reduceprod(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: reduceprod, op_name: {xnode.op_name}")

        # create input ops for Prod
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["keep_dims"] = xnode.keep_dims
        logger.debug(f"attribute: keep_dims: {attrs['keep_dims']}")
        attrs["dims"] = xnode.axis
        logger.debug(f"attribute: dims: {attrs['dims']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "reduction_product",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_reduceprod_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "reduction_product"

        # attributes of op
        # dims
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.axis)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["dims"].CopyFrom(attr)
        # keep_dims
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.keep_dims
        node.op_attr["keep_dims"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_reducesum(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: reducesum, op_name: {xnode.op_name}")

        # create input ops for Prod
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["keep_dims"] = xnode.keep_dims
        logger.debug(f"attribute: keep_dims: {attrs['keep_dims']}")
        attrs["axis"] = xnode.axis
        logger.debug(f"attribute: axis: {attrs['axis']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "reduction_sum",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_reducesum_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "reduction_sum"

        # attributes of op
        # dims
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.axis)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["axis"].CopyFrom(attr)
        # keep_dims
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.keep_dims
        node.op_attr["keep_dims"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_reducemax(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: reducemax, op_name: {xnode.op_name}")

        # create input ops for Prod
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["keep_dims"] = xnode.keep_dims
        logger.debug(f"attribute: keep_dims: {attrs['keep_dims']}")
        attrs["axis"] = xnode.axis
        logger.debug(f"attribute: axis: {attrs['axis']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "reduction_max",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_reducemax_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "reduction_max"

        # attributes of op
        # axis
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.axis)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["axis"].CopyFrom(attr)
        # keep_dims
        attr = xir_proto.AttrValue()
        attr.bool_value = xnode.keep_dims
        node.op_attr["keep_dims"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_scale(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: scale, op_name: {xnode.op_name}")

        # create const op for bias
        bias, fix_op_bias = None, None
        if hasattr(xnode, "bias_term"):
            if xnode.bias_term:
                # set out_tensor
                out_tensor_b = xnode.bias
                logger.debug(
                    f"attribute: bias: shape: {out_tensor_b.shape}, dtype: {out_tensor_b.dtype}"
                )
                const_op_name = xnode.op_name + "_bias"
                bias = graph.create_const_op(const_op_name, out_tensor_b.to_numpy())
                # put into op_dict
                assert (
                    const_op_name not in op_dict
                ), f"Duplicated op: op_name={const_op_name}."
                op_dict[const_op_name] = bias

                if (
                    xnode.quant_bias["quantize_pos"] is not None
                    and xnode.quant_bias["bit_width"] is not None
                ):
                    # create FixedNeuron op for bias
                    fix_op_name = const_op_name + "_fixneuron"
                    # dtype
                    dtype = out_tensor_b.dtype
                    # shape
                    shape = out_tensor_b.shape
                    out_tensor = np.zeros(shape, dtype=dtype)
                    fix_op_bias = cls.__create_fixneuron(
                        fix_op_name,
                        [const_op_name],
                        xnode.quant_bias,
                        out_tensor,
                        graph,
                        op_dict,
                    )

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        input_op = op_dict.get(xnode.bottom[0])
        assert (
            input_op is not None
        ), f"[ERROR] input op is None: op name: {xnode.op_name}"
        input_ops["input"].append(input_op)
        # scale
        scale_op = op_dict.get(xnode.bottom[1])
        assert (
            scale_op is not None
        ), f"[ERROR] scale op is None: op name: {xnode.op_name}"
        input_ops["scale"] = [scale_op]
        # bias
        if xnode.bias_term:
            input_ops["bias"] = (
                [fix_op_bias]
                if xnode.quant_bias["quantize_pos"] is not None
                and xnode.quant_bias["bit_width"] is not None
                else [bias]
            )

        # create op
        return graph.create_op(xnode.op_name, "scale", input_ops=input_ops)

    @classmethod
    def to_scale_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "scale"

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args of op
        op_args = []
        assert len(xnode.bottom) == 2
        # input
        op_arg_input = xir_proto.OpArg()
        op_arg_input.arg_name = "input"
        op_arg_input.arg_ops.extend([xnode.bottom[0]])
        op_args.append(op_arg_input)
        # scale
        op_arg_scale = xir_proto.OpArg()
        op_arg_scale.arg_name = "scale"
        op_arg_scale.arg_ops.extend([xnode.bottom[1]])
        op_args.append(op_arg_scale)
        # bias
        if xnode.bias_term:
            const_node_b, fix_node_b = cls.__create_xir_const_node_from_xtensor(
                name=xnode.op_name + "_bias",
                xtensor=xnode.bias,
                quant_info=xnode.quant_bias,
            )
            assert (
                const_node_b.op_name not in op_dict
            ), f"[ERROR] Found duplicate xir op: op_name={const_node_b.op_name}."
            op_dict[const_node_b.op_name] = const_node_b
            if fix_node_b:
                assert (
                    fix_node_b.op_name not in op_dict
                ), f"[ERROR] Found duplicate xir op: op_name={fix_node_b.op_name}."
                op_dict[fix_node_b.op_name] = fix_node_b

            op_arg = xir_proto.OpArg()
            op_arg.arg_name = "bias"
            if fix_node_b:
                op_arg.arg_ops.extend([fix_node_b.op_name])
            else:
                op_arg.arg_ops.extend([const_node_b.op_name])
            op_args.append(op_arg)
        node.args.extend(op_args)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        op_dict[node.op_name] = node

    @classmethod
    def to_sigmoid(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: sigmoid, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        input_op = op_dict.get(xnode.bottom[0])
        assert (
            input_op is not None
        ), f"[ERROR] input op is None: op name: {xnode.op_name}"
        input_ops["input"].append(input_op)

        # create op
        if "hard_sigmoid" in xnode.tmp_params and xnode.tmp_params["hard_sigmoid"]:
            return graph.create_op(xnode.op_name, "hard-sigmoid", input_ops=input_ops)
        else:
            return graph.create_op(xnode.op_name, "sigmoid", input_ops=input_ops)

    @classmethod
    def to_sigmoid_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name

        if "hard_sigmoid" in xnode.tmp_params and xnode.tmp_params["hard_sigmoid"]:
            node.op_type = "hard-sigmoid"
        else:
            node.op_type = "sigmoid"

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args of op
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        op_dict[node.op_name] = node

    @classmethod
    def to_elemsquare(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: elemsqaure, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        input_op = op_dict.get(xnode.bottom[0])
        assert (
            input_op is not None
        ), f"[ERROR] input op is None: op name: {xnode.op_name}"
        input_ops["input"].append(input_op)

        # create op
        return graph.create_op(xnode.op_name, "elemsquare", input_ops=input_ops)

    @classmethod
    def to_elemsquare_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "elemsquare"

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_elemrsqrt(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: elemrsqrt, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        input_op = op_dict.get(xnode.bottom[0])
        assert (
            input_op is not None
        ), f"[ERROR] input op is None: op name: {xnode.op_name}"
        input_ops["input"].append(input_op)

        # create op
        return graph.create_op(xnode.op_name, "elemrsqrt", input_ops=input_ops)

    @classmethod
    def to_elemrsqrt_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "elemrsqrt"

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_elemmax(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create op
        return graph.create_op(xnode.op_name, "max", input_ops=input_ops)

    @classmethod
    def to_elemmax_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "max"

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 2
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_l2_normalize(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["axis"] = xnode.axis
        logger.debug(f"attribute: axis: {attrs['axis']}")
        attrs["epsilon"] = xnode.epsilon
        logger.debug(f"attribute: epsilon: {attrs['epsilon']}")

        # create op
        return graph.create_op(
            xnode.op_name, "l2_normalize", attrs=attrs, input_ops=input_ops
        )

    @classmethod
    def to_l2_normalize_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "max"

        # attributes of op
        # axis
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.axis)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["axis"].CopyFrom(attr)
        # epsilon
        attr = xir_proto.AttrValue()
        attr.float_value = xnode.epsilon
        node.op_attr["epsilon"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_depth_to_space(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops for gstiling
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["reverse"] = True
        logger.debug(f"attribute: reverse: {attrs['reverse']}")
        attrs["stride"] = xnode.block_size
        logger.debug(f"attribute: stride: {attrs['stride']}")

        # create gstiling op
        return graph.create_op(
            xnode.op_name,
            "gstiling",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_depth_to_space_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "gstiling"

        # attributes of op
        # stride
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.block_size
        node.op_attr["stride"].CopyFrom(attr)
        # reverse
        attr = xir_proto.AttrValue()
        attr.bool_value = True
        node.op_attr["reverse"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_elemmin(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # create op
        return graph.create_op(xnode.op_name, "min", input_ops=input_ops)

    @classmethod
    def to_elemmin_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "min"

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 2
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_round_typecast(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        if (
            xnode.tmp_params["src_dtype"] == np.float32
            and xnode.tmp_params["dst_dtype"] == np.uint8
        ):
            # set attrs for op
            attrs: Dict[str, Any] = {}
            attrs["fix_point"] = 0
            logger.debug(f"attribute: fix_point: {attrs['fix_point']}")
            attrs["bit_width"] = 8
            logger.debug(f"attribute: bit_width: {attrs['bit_width']}")
            attrs["if_signed"] = False
            logger.debug(f"attribute: if_signed: {attrs['if_signed']}")
            attrs["round_mode"] = xnode.tmp_params["round_mode"].upper()
            logger.debug(f"attribute: round_mode: {attrs['round_mode']}")

            # create op
            return graph.create_op(
                xnode.op_name, "float2fix", attrs=attrs, input_ops=input_ops
            )
        else:
            raise TypeError(f"[ERROR] Failed to map {xnode.op_type} to an xir op.")

    @classmethod
    def to_round_typecast_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "float2fix"

        assert (
            xnode.tmp_params["src_dtype"] == np.float32
            and xnode.tmp_params["dst_dtype"] == np.uint8
        ), f"[ERROR] Failed to map {xnode.op_type} to an xir op."

        # attributes of op
        # fix_point
        attr = xir_proto.AttrValue()
        attr.int32_value = 0
        node.op_attr["fix_point"].CopyFrom(attr)
        # bit_width
        attr = xir_proto.AttrValue()
        attr.int32_value = 8
        node.op_attr["bit_width"].CopyFrom(attr)
        # round_mode
        attr = xir_proto.AttrValue()
        attr.string_value = xnode.tmp_params["round_mode"].upper()
        node.op_attr["round_mode"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_random_standard_normal(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        # dtype
        attrs["data_type"] = xnode.dtype.__name__
        logger.debug(f"attribute: data_type: {attrs['data_type']}")
        # seed
        attrs["seed"] = xnode.seed
        logger.debug(f"attribute: seed: {attrs['seed']}")
        # seed2
        attrs["seed2"] = xnode.seed2
        logger.debug(f"attribute: seed2: {attrs['seed2']}")
        # shape
        attrs["shape"] = xnode.outputs_tensor[0].shape
        logger.debug(f"attribute: shape: {attrs['shape']}")

        # create op
        return graph.create_op(
            xnode.op_name,
            "random_standard_normal",
            attrs=attrs,  # input_ops=input_ops
        )

    @classmethod
    def to_random_standard_normal_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "random_standard_normal"

        # attributes of op
        # data_type
        attr = xir_proto.AttrValue()
        attr.string_value = xnode.dtype.__name__
        node.op_attr["data_type"].CopyFrom(attr)
        # seed
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.seed
        node.op_attr["seed"].CopyFrom(attr)
        # seed2
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.seed2
        node.op_attr["seed2"].CopyFrom(attr)
        # shape
        attr_value = xir_proto.Int32Vec()
        attr_value.value.extend(xnode.outputs_tensor[0].shape)
        attr = xir_proto.AttrValue()
        attr.int32_vec_value.CopyFrom(attr_value)
        node.op_attr["shape"].CopyFrom(attr)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_elemtanh(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: elemtanh, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        input_op = op_dict.get(xnode.bottom[0])
        assert (
            input_op is not None
        ), f"[ERROR] input op is None: op name: {xnode.op_name}"
        input_ops["input"].append(input_op)

        # create op
        return graph.create_op(xnode.op_name, "tanh", input_ops=input_ops)

    @classmethod
    def to_elemtanh_v2(
        cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "tanh"

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_argmax(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: argmax, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        input_op = op_dict.get(xnode.bottom[0])
        assert (
            input_op is not None
        ), f"[ERROR] input op is None: op name: {xnode.op_name}"
        input_ops["input"].append(input_op)

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["axis"] = xnode.axis
        logger.debug(f"attribute: axis: {attrs['axis']}")

        attrs['data_type']=xnode.output_type.__name__
        attrs['shape']=xnode.outputs_tensor[0].shape

        # create op
        return graph.create_op(
            xnode.op_name,
            "argmax",
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_argmax_v2(cls, xnode: XModelNode, op_dict: Dict[str, "xir.Op"]) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' shoulde be an XModelNode instance."

        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        node = xir_proto.OPNode()
        node.op_name = xnode.op_name
        node.op_type = "argmax"

        # attributes of op
        # axis
        attr = xir_proto.AttrValue()
        attr.int32_value = xnode.axis
        node.op_attr["axis"].CopyFrom(attr)

        # quantization info
        cls.__set_xir_node_quant_info(xnode, node)

        # args
        assert len(xnode.bottom) == 1
        op_arg = xir_proto.OpArg()
        op_arg.arg_name = "input"
        op_arg.arg_ops.extend(xnode.bottom)
        node.args.extend([op_arg])

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(
            node.op_name, xnode.outputs_tensor[0]
        )
        node.output_tensor.CopyFrom(output_tensor)

        assert (
            node.op_name not in op_dict
        ), f"[ERROR] Found duplicate xir op: op_name={node.op_name}."
        op_dict[node.op_name] = node

    @classmethod
    def to_clip_by_value(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        input_op = op_dict.get(xnode.bottom[0])
        assert (
            input_op is not None
        ), f"[ERROR] input op is None: op name: {xnode.op_name}"
        input_ops["input"].append(input_op)

        # set attrs for op
        attrs: Dict[str, Any] = {}
        for key, val in xnode.tmp_params.items():
            attrs[key] = val
        attrs["data_type"] = xnode.outputs_tensor[0].dtype_str.upper()
        attrs["shape"] = xnode.outputs_tensor[0].shape
        logger.debug(f"attributes: {attrs}")

        # create op
        return graph.create_op(
            xnode.op_name,
            xnode.op_type,
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def __create_fixneuron(
        cls,
        name: str,
        parent_names: List[str],
        quant_info: Dict[str, Any],
        out_tensor: np.ndarray,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
    ) -> "xir.Op":
        # create input ops for conv2d
        input_ops = {"input": []}
        for in_op_name in parent_names:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["fix_point"] = quant_info.get("quantize_pos")
        logger.debug(f"attribute: fix_point: {attrs['fix_point']}")
        attrs["bit_width"] = quant_info.get("bit_width")
        logger.debug(f"attribute: bit_width: {attrs['bit_width']}")
        attrs["if_signed"] = quant_info.get("signed")
        logger.debug(f"attribute: if_signed: {attrs['if_signed']}")
        assert (
            quant_info.get("round_mode") is not None
        ), f"[ERROR] 'round_mode' field of FixNeuron should not be None: {name}"
        attrs["round_mode"] = QuantRoundMode[quant_info.get("round_mode")]
        logger.debug(f"attribute: round_mode: {attrs['round_mode']}")

        # create FixedNeuron op
        op: "xir.Op" = graph.create_op(name, "fix", attrs=attrs, input_ops=input_ops)
        # put into op_dict
        assert name not in op_dict, f"Duplicated op: op_name={name}."
        op_dict[name] = op
        logger.info(
            f"FixNeuron created: op_name: {name}, op_type: fix, quant_info: {attrs}"
        )
        return op

    @classmethod
    def __md5sum(cls, fname: Path) -> str:
        with open(fname, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        return md5

    @classmethod
    def __remove_conv2dtranspose_output_shape(cls, xmodel: XModel) -> NoReturn:
        for xnode in xmodel.xnodes:
            if xnode.op_type == "conv2d_transpose":
                if len(xnode.bottom) == 2:
                    out_shape_id = xnode.bottom.pop()
                    out_shape_node = xmodel.get_xnode_by_name(out_shape_id)
                    assert out_shape_node is not None and isinstance(
                        out_shape_node, XModelNodeConst
                    ), f"[ERROR] mismatched node type: actual: {out_shape_node.__class__.__name__}, expected: XModelNodeConst."
                    out_shape_node.top = []
                    xmodel.remove_xnode(out_shape_node)

    @classmethod
    def __create_xir_output_tensor(
        cls, name: str, out_tensor: XTensor
    ) -> "xir_proto.Tensor":
        assert name is not None and isinstance(name, str), "'name' should be a string."
        assert out_tensor is not None and isinstance(
            out_tensor, XTensor
        ), "'out_tensor' should be an XTensor object."

        # output_tensor of op
        output_tensor = xir_proto.Tensor()
        output_tensor.tensor_name = name
        output_tensor.tensor_dim.extend(out_tensor.shape)
        if out_tensor.dtype == np.float32:
            output_tensor.data_type = 9
            output_tensor.tensor_bit_width = 32
        elif out_tensor.dtype == np.int32:
            output_tensor.data_type = 5
            output_tensor.tensor_bit_width = 32
        elif out_tensor.dtype == np.uint8:
            output_tensor.data_type = 2
            output_tensor.tensor_bit_width = 8
        elif out_tensor.dtype == np.int8:
            output_tensor.data_type = 1
            output_tensor.tensor_bit_width = 8
        else:
            raise TypeError(f"[ERROR] Unsupported data type: {out_tensor.dtype}.")

        return output_tensor

    @classmethod
    def __create_xir_const_node_from_xtensor(
        cls, name: str, xtensor: XTensor, quant_info: Dict[str, Any]
    ) -> Tuple["xir_proto.OPNode", "xir_proto.OPNode"]:

        const_node = xir_proto.OPNode()
        const_node.op_name = name
        const_node.op_type = "const"

        # attributes of op
        # data
        content = xir_proto.Bytes()
        content.value = xtensor.tobytes()
        data = xir_proto.AttrValue()
        data.bytes_value.CopyFrom(content)
        const_node.op_attr["data"].CopyFrom(data)
        # shape
        val_shape = xir_proto.Int32Vec()
        val_shape.value.extend(list(xtensor.shape))
        shape = xir_proto.AttrValue()
        shape.int32_vec_value.CopyFrom(val_shape)
        const_node.op_attr["shape"].CopyFrom(shape)
        # data type
        dtype = xir_proto.AttrValue()
        dtype.string_value = xtensor.dtype_str.lower()
        const_node.op_attr["data_type"].CopyFrom(dtype)

        # output_tensor of op
        output_tensor = cls.__create_xir_output_tensor(const_node.op_name, xtensor)
        const_node.output_tensor.CopyFrom(output_tensor)

        fix_node = None
        if (
            quant_info
            and quant_info["quantize_pos"] is not None
            and quant_info["bit_width"] is not None
        ):
            fix_op_name = name + "_fixneuron"
            logger.info(f"*** op_type: fix, op_name: {fix_op_name}")

            fix_node = xir_proto.OPNode()
            fix_node.op_name = fix_op_name
            fix_node.op_type = "fix"

            # attributes of op
            # bit_width
            attr = xir_proto.AttrValue()
            attr.int32_value = quant_info.get("bit_width")
            fix_node.op_attr["bit_width"].CopyFrom(attr)
            fix_node.op_attr["quant_in_bit_width"].CopyFrom(attr)
            fix_node.op_attr["quant_out_bit_width"].CopyFrom(attr)

            # quantize_pos
            attr = xir_proto.AttrValue()
            attr.int32_value = quant_info.get("quantize_pos")
            fix_node.op_attr["fix_point"].CopyFrom(attr)
            fix_node.op_attr["quant_in_quantize_pos"].CopyFrom(attr)
            fix_node.op_attr["quant_out_quantize_pos"].CopyFrom(attr)

            # round_mode
            attr = xir_proto.AttrValue()
            attr.string_value = QuantRoundMode[quant_info.get("round_mode")]
            fix_node.op_attr["round_mode"].CopyFrom(attr)
            attr = xir_proto.AttrValue()
            attr.int32_value = quant_info.get("round_mode")
            fix_node.op_attr["quant_in_round_mode"].CopyFrom(attr)
            fix_node.op_attr["quant_out_round_mode"].CopyFrom(attr)

            # quant_in_signed
            attr = xir_proto.AttrValue()
            attr.bool_value = quant_info.get("signed")
            fix_node.op_attr["if_signed"].CopyFrom(attr)
            fix_node.op_attr["quant_in_signed"].CopyFrom(attr)
            fix_node.op_attr["quant_out_signed"].CopyFrom(attr)

            # args
            op_arg = xir_proto.OpArg()
            op_arg.arg_name = "input"
            op_arg.arg_ops.extend([const_node.op_name])
            fix_node.args.extend([op_arg])

            # output_tensor of op
            output_tensor = cls.__create_xir_output_tensor(fix_node.op_name, xtensor)
            fix_node.output_tensor.CopyFrom(output_tensor)

        return const_node, fix_node

    @classmethod
    def __set_xir_node_quant_info(
        cls, xnode: XModelNode, node: "xir_proto.OPNode"
    ) -> NoReturn:
        assert xnode is not None and isinstance(
            xnode, XModelNode
        ), "'xnode' should be an XModelNode instance."
        assert node is not None, "'node' should be an xir_proto.OPNode instance."

        if xnode.host.origin == "caffe":
            return

        if xnode.quant_in.get("bit_width") and xnode.quant_in.get("quantize_pos"):
            # bit_width
            attr = xir_proto.AttrValue()
            attr.int32_value = xnode.quant_in.get("bit_width")
            node.op_attr["quant_in_bit_width"].CopyFrom(attr)

            # quantize_pos
            attr = xir_proto.AttrValue()
            attr.int32_value = xnode.quant_in.get("quantize_pos")
            node.op_attr["quant_in_quantize_pos"].CopyFrom(attr)

            # round_mode
            attr = xir_proto.AttrValue()
            attr.string_value = QuantRoundMode[xnode.quant_in.get("round_mode")]
            node.op_attr["quant_in_round_mode"].CopyFrom(attr)

            # quant_in_signed
            attr = xir_proto.AttrValue()
            attr.bool_value = xnode.quant_in.get("signed")
            node.op_attr["quant_in_signed"].CopyFrom(attr)

        if xnode.quant_out.get("bit_width") and xnode.quant_out.get("quantize_pos"):
            # bit_width
            attr = xir_proto.AttrValue()
            attr.int32_value = xnode.quant_out.get("bit_width")
            node.op_attr["quant_out_bit_width"].CopyFrom(attr)

            # quantize_pos
            attr = xir_proto.AttrValue()
            attr.int32_value = xnode.quant_out.get("quantize_pos")
            node.op_attr["quant_out_quantize_pos"].CopyFrom(attr)

            # round_mode
            attr = xir_proto.AttrValue()
            attr.string_value = QuantRoundMode[xnode.quant_out.get("round_mode")]
            node.op_attr["quant_out_round_mode"].CopyFrom(attr)

            # quant_in_signed
            attr = xir_proto.AttrValue()
            attr.bool_value = xnode.quant_out.get("signed")
            node.op_attr["quant_out_signed"].CopyFrom(attr)

    # ! to be removed
    @classmethod
    def to_spacetobatchnd(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: space_to_batch_nd, op_name: {xnode.op_name}")
        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))
        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["block_shape"] = xnode.block_shape
        logger.debug(f"attribute: block_shape: {attrs['block_shape']}")
        attrs["paddings"] = xnode.paddings[2:] + xnode.paddings[:2]
        logger.debug(
            f"attribute: paddings (left, right, top, bottom): {attrs['paddings']}"
        )

        # create op
        return graph.create_op(
            xnode.op_name, "space_to_batch_nd", attrs=attrs, input_ops=input_ops
        )

    # ! to be removed
    @classmethod
    def to_batchtospacend(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":

        logger.info(f"*** op_type: batch_to_space_nd, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        attrs["block_shape"] = xnode.block_shape
        logger.debug(f"attribute: block_shape: {attrs['block_shape']}")
        attrs["crops"] = xnode.crops[2:] + xnode.crops[:2]
        logger.debug(f"attribute: crops (left, right, top, bottom): {attrs['crops']}")

        # create op
        return graph.create_op(
            xnode.op_name, "batch_to_space_nd", attrs=attrs, input_ops=input_ops
        )

    @classmethod
    def to_unknown(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []

        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}
        for key, val in xnode.tmp_params.items():
            if key in ['kind', 'name']:
                continue
            else:
                if key != 'shape' and not isinstance(val, (bool, int, float, str, bytes)):
                    val = str(val)
                attrs[key] = val

        if set(['shape', 'data_type']) <= set(list(xnode.tmp_params)):
            if "shape" not in attrs:
                attrs["shape"] = xnode.tmp_params["shape"]
            if "data_type" not in attrs:
                attrs["data_type"] = xnode.tmp_params["data_type"]
        elif len(xnode.outputs_tensor) == 1:
            if "shape" not in attrs:
                attrs["shape"] = xnode.outputs_tensor[0].shape
            if "data_type" not in attrs:
                attrs["data_type"] = xnode.outputs_tensor[0].dtype_str

        # create op
        return graph.create_op(
            xnode.tmp_params['name'],
            xnode.tmp_params['kind'],
            attrs=attrs,
            input_ops=input_ops,
        )

    @classmethod
    def to_elemfloor(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        return cls.to_unknown(xnode, graph, op_dict, layout, origin)

    @classmethod
    def to_expand_dims(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":

        logger.info(f"*** op_type: reshape, op_name: {xnode.op_name}")

        input_ops = {}
        in_op: "xir.Op" = op_dict.get(xnode.bottom[0])
        assert in_op is not None, f"Not found op: name: {xnode.bottom[0]}"

        input_ops["input"] = [in_op]

        assert len(xnode.bottom) == 2
        # shape info as one input
        shape_op: "xir.Op" = op_dict.get(xnode.bottom[1])
        assert (
            shape_op is not None
        ), f"Not found op: name: {xnode.bottom[1]}. Current op type: {xnode.op_type}, name: {xnode.op_name}."
        input_ops["shape"] = [shape_op]

        return graph.create_op(
            xnode.op_name,
            "reshape",
            input_ops=input_ops
        )


    @classmethod
    def to_type_cast(
        cls,
        xnode: XModelNode,
        graph: "xir.Graph",
        op_dict: Dict[str, "xir.Op"],
        layout: str,
        origin: str,
    ) -> "xir.Op":
        logger.info(f"*** op_type: {xnode.op_type}, op_name: {xnode.op_name}")

        # create input ops
        input_ops = {}
        input_ops["input"] = []
        for in_op_name in xnode.bottom:
            input_ops["input"].append(op_dict.get(in_op_name))

        # set attrs for op
        attrs: Dict[str, Any] = {}

        if 'dtype' in xnode.tmp_params:
            attrs["data_type"] = xnode.tmp_params['dtype']
        else:
            attrs["data_type"] = xnode.dst_dtype.__name__
        logger.debug(f"attribute: data_type: {attrs['data_type']}")

        # create op
        return graph.create_op(
            xnode.op_name, "cast", attrs=attrs, input_ops=input_ops
        )

