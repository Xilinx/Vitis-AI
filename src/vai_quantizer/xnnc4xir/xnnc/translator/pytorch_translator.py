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

import json
import logging
import math
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xnnc.xnnc.ir.xnode import RoundMode

import numpy as np
import torch
from torch.onnx import OperatorExportTypes
from tqdm import tqdm

from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import *
from xnnc.optimizer import OptManager
from xnnc.translator.base_translator import ITranslator
from xnnc.utils import helper, pytorch_graph
from xnnc.utils.helper import Layout

# create logger
logger = logging.getLogger(__name__)


# the mapping relation between torch dtype and numpy dtype
TORCH_DTYPE_TO_NUMPY_DTYPE = {
    0: np.float32,
    1: np.float64,
    2: np.float16,
    3: np.uint8,
    4: np.int8,
    5: np.int16,
    6: np.int32,
    7: np.int64,
}


class PyTorchTranslator(ITranslator):
    """
    Convert a PyTorch model into XModel object.
    """

    @classmethod
    def to_xmodel(
        cls, model_files: List[Path], layout: Layout = Layout.NCHW, *args, **kwargs
    ) -> XModel:
        assert model_files is not None, "'model_files' should not be None."
        assert "in_shapes" in kwargs, "Please specify the shape of input feature map."
        assert kwargs["in_shapes"] is not None
        in_shapes = kwargs["in_shapes"]

        # load model architecture
        logger.info("load raw model architecture")
        model_name, model = cls.__load_raw_model(model_files)
        logger.debug(f"raw model name: {model_name}")

        # * jit parse model
        graph_py = pytorch_graph.parse_model(model, in_shapes, dump_image=False)
        nodes_op_dict = {}
        for node in graph_py.nodes_op:
            assert node.op_name not in nodes_op_dict
            nodes_op_dict[node.op_name] = node
        logger.info("* start: translate pytorch layers to xmodel nodes")

        # create an xmodel
        xmodel = XModel(model_name, "pytorch")
        logger.info(
            f"create XModel object: name: {xmodel.name}, type: {xmodel.origin}, layout: {xmodel.layout}"
        )

        # create input node
        input_layers = list(graph_py.nodes_io.values())[: len(in_shapes)]
        layers = []
        for inode in input_layers:
            # create XModelNodeInput object
            node = XModelNodeInput(inode.debug_name)
            logger.debug(f"create XModelNodeInput object: name: {node.op_name}")
            # shape: (N,C,H,W)
            node.shape = inode.tensor_shape
            logger.debug(f"property: shape ({xmodel.layout}): {node.shape}")

            # inputs tensor
            node.inputs_tensor.append(np.zeros(node.shape, dtype=np.float32))

            # top
            node.top = inode.outputs
            # add to xmodel
            xmodel.add_xnode(node)
            # update layers
            for out_name in inode.outputs:
                # ! debug
                # layer = layer_dict.get(out_name)
                layer = nodes_op_dict.get(out_name)
                assert (
                    layer is not None
                ), f"[ERROR] Not found child node_op: alias: {out_name}"
                layers.append(layer)

        # * extract op nodes, which are the real nodes to parse
        node_dict = {}
        while len(layers) > 0:
            layer = layers.pop(0)

            if layer.op_name not in node_dict:
                node_dict[layer.op_name] = layer

            if layer.outputs is not None and len(layer.outputs) > 0:
                for cname in layer.outputs:
                    child = nodes_op_dict.get(cname)
                    assert (
                        child is not None
                    ), f"[ERROR] Not found child layer: alias: {cname}"
                    layers.append(child)
        logger.debug(f"numb. of raw layers: {len(node_dict)}")

        # ! debug
        # pytorch_graph.dump(node_dict, "debug")

        # * translate each pytorch node into xnode
        fix_neuron_to_split: List[XModelNodeFixNeuron] = []
        pbar = tqdm(
            node_dict.values(),
            desc="[INFO] parse raw model",
            bar_format="{desc:23}:{percentage:3.0f}%|{bar}{r_bar:50}",
        )
        for layer in pbar:
            # get op name and type
            op_name: str = str(layer.op_name)
            op_type: str = str(layer.op_type).lower()

            # outputs tensor shape
            outputs_tensor_shape = layer.outputs_tensor_shape

            logger.debug(f"*** source layer info: type: {op_type}, name: {op_name}")

            # translate current layer to XModel node object
            node: XModelNode = None
            if op_type == "convolution":
                (
                    input_id,
                    weight_id,  # for quantized node, it is the id of fix neuron for weight
                    bias_id,  # for quantized node, it is the id of fix neuron for bias
                    stride_id,
                    padding_id,
                    dilation_id,
                    transposed_id,
                    output_padding_id,
                    groups_id,
                    _,
                    _,
                    _,
                ) = list(layer.inputs)

                # get kernel size
                ksize = layer.inputs_tensor_shape[1][-2:]

                # transposed
                transposed = bool(
                    nodes_op_dict[transposed_id].attributes["value"].data.item()
                )

                if transposed:
                    # create XModelNodeConv2dTranspose object
                    node = XModelNodeConv2dTranspose(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeConv2dTranspose object: name: {node.op_name}"
                    )
                else:
                    # create XModelNodeConv2d object
                    node = XModelNodeConv2d(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeConv2d object: name: {node.op_name}"
                    )
                logger.debug(f"property: (kernel_h, kernel_w): {node.kernel_size}")

                # bottom
                node.bottom = [input_id]
                node.inputs_tensor_shape = [layer.inputs_tensor_shape[0]]

                # group
                node.group = (
                    nodes_op_dict.get(groups_id).attributes["value"].item()
                    if groups_id is not None
                    else 1
                )
                logger.debug(f"property: group: {node.group}")
                # round_mode
                node.round_mode = RoundMode.FLOOR
                logger.debug(f"property: round_mode: {node.round_mode}")
                # dilation: [N,C,H,W]
                if dilation_id is not None:
                    dilation = nodes_op_dict.get(dilation_id).attributes["value"]
                    if len(dilation) == 2:
                        node.dilation = [1, 1] + dilation
                    else:
                        raise ValueError(
                            f"Unsupported dilation value format: dilation: {dilation}"
                        )
                else:
                    node.dilation = [1, 1, 1, 1]
                logger.debug(f"property: dilation: {node.dilation}")
                # padding: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
                if padding_id is not None:
                    padding = nodes_op_dict.get(padding_id).attributes["value"]
                    if len(padding) == 2:
                        node.padding = [padding[0]] * 2 + [padding[1]] * 2
                    else:
                        raise ValueError(
                            f"Unsupported padding value format: padding: {padding}"
                        )
                else:
                    node.padding = [0, 0, 0, 0]
                logger.debug(f"property: padding: {node.padding}")
                # stride: [stride_h, stride_w]
                node.strides = (
                    nodes_op_dict.get(stride_id).attributes["value"]
                    if stride_id is not None
                    else [1, 1]
                )
                logger.debug(f"property: strides: {node.strides}")

                # get fix neuron for weight
                fix_neuron_w = nodes_op_dict.get(weight_id)
                assert fix_neuron_w is not None
                assert len(fix_neuron_w.inputs) == 4
                (
                    real_weight_id,
                    log2t_w_id,
                    bit_width_w_id,
                    round_mode_w_id,
                ) = fix_neuron_w.inputs

                # weight
                weight = nodes_op_dict.get(real_weight_id)
                if weight is not None:
                    assert (
                        weight is not None
                    ), f"[ERROR] Not found weight: name: {real_weight_id}"
                    assert len(weight.inputs) > 0
                    weight_tensor_id = weight.inputs[0]

                else:
                    weight_tensor_id = real_weight_id
                weight_tensor: np.ndarray = graph_py.nodes_io.get(
                    weight_tensor_id
                ).tensor.data.numpy()
                # convert (out_ch, in_ch, H, W) into (H, W, in_ch, out_ch)
                node.weights = np.transpose(weight_tensor, (2, 3, 1, 0))
                logger.debug(
                    f"property: weights: shape (in_channels, height, width, out_channels): {node.weights.shape}, dtype: {node.weights.dtype}"
                )

                # compute quantization position
                log2t_w = graph_py.nodes_io.get(log2t_w_id).tensor.item()
                bit_width_w = int(graph_py.nodes_io.get(bit_width_w_id).tensor.item())
                quantize_pos_w = cls.__compute_quantize_pos(bit_width_w, log2t_w, True)
                round_mode_w = int(graph_py.nodes_io.get(round_mode_w_id).tensor.item())
                # set quantize info
                node.quant_weights["bit_width"] = bit_width_w
                node.quant_weights["quantize_pos"] = quantize_pos_w
                node.quant_weights["signed"] = True
                node.quant_weights["round_mode"] = round_mode_w

                # get fix neuron for bias
                fix_neuron_b = nodes_op_dict.get(bias_id)
                assert fix_neuron_b is not None
                if len(fix_neuron_b.inputs) == 4:
                    node.bias_term = True
                    (
                        real_bias_id,
                        bias_log2t_id,
                        bias_bit_width_id,
                        bias_round_mode_id,
                    ) = fix_neuron_b.inputs

                    # bias
                    bias_tensor: np.ndarray = graph_py.nodes_io.get(
                        real_bias_id
                    ).tensor.data.numpy()
                    node.bias = bias_tensor
                    logger.debug(
                        f"property: bias: shape: {node.bias.shape}, dtype: {node.bias.dtype}"
                    )

                    # compute quantization position
                    log2t_b = graph_py.nodes_io.get(bias_log2t_id).tensor.item()
                    bit_width_b = int(
                        graph_py.nodes_io.get(bias_bit_width_id).tensor.item()
                    )
                    quantize_pos_b = cls.__compute_quantize_pos(
                        bit_width_b, log2t_b, True
                    )
                    round_mode_b = int(
                        graph_py.nodes_io.get(bias_round_mode_id).tensor.item()
                    )
                    # set quantization info
                    node.quant_bias["bit_width"] = bit_width_b
                    node.quant_bias["quantize_pos"] = quantize_pos_b
                    node.quant_bias["signed"] = True
                    node.quant_bias["round_mode"] = round_mode_b

                elif len(fix_neuron_b.inputs) == 0:
                    node.bias_term = False
                else:
                    raise ValueError(
                        f"[ERROR] the bias node has the unsupported number of inputs: {fix_neuron_b.inputs}"
                    )

            elif op_type == "quantizekrelu":
                (
                    input_id,
                    log2t_id,
                    bit_width_id,
                    signed_id,
                    split_id,
                    round_mode_id,
                ) = list(layer.inputs)

                # create XModelNodeFixNeuron object
                node = XModelNodeFixNeuron(op_name)
                logger.debug(f"create XModelNodeFixNeuron object: name: {node.op_name}")

                node.bottom = [input_id]
                node.inputs_tensor_shape = [layer.inputs_tensor_shape[0]]

                # get clip value, bit width, and signed flag
                log2t = graph_py.nodes_io.get(log2t_id).tensor.item()
                bit_width = int(graph_py.nodes_io.get(bit_width_id).tensor.item())
                signed = bool(graph_py.nodes_io.get(signed_id).tensor.item())
                round_mode = int(graph_py.nodes_io.get(round_mode_id).tensor.item())

                # compute quantization position
                quantize_pos = cls.__compute_quantize_pos(bit_width, log2t, signed)
                # set quantization info
                node.quant_in["bit_width"] = bit_width
                node.quant_in["quantize_pos"] = quantize_pos
                node.quant_in["signed"] = signed
                node.quant_in["round_mode"] = round_mode
                node.quant_out["bit_width"] = bit_width
                node.quant_out["quantize_pos"] = quantize_pos
                node.quant_out["signed"] = signed
                node.quant_out["round_mode"] = round_mode

                # if split is True, current node needs to be split into relu + fix_neuron
                split = bool(graph_py.nodes_io.get(split_id).tensor.item())
                if split:
                    fix_neuron_to_split.append(node)

            elif op_type in ["maxpool2d", "avgpool2d"]:
                (
                    input_id,
                    ksize_id,
                    stride_id,
                    padding_id,
                    dilation_id,
                    ceil_mode_id,
                ) = list(layer.inputs)

                # kernel size
                ksize: List[int] = nodes_op_dict.get(ksize_id).attributes["value"]

                # create pooling object
                node = None
                if op_type == "maxpool2d":
                    node = XModelNodeMaxPool(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeMaxPool object: name: {node.op_name}"
                    )
                else:
                    node = XModelNodeAvgPool(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeAvgPool object: name: {node.op_name}"
                    )
                logger.debug(f"property: (kernel_h, kernel_w): {node.kernel_size}")

                # bottom
                node.bottom = [input_id]
                node.inputs_tensor_shape = [layer.inputs_tensor_shape[0]]

                # global pooling
                if node.kernel_size == node.inputs_tensor_shape[0][-2:]:
                    node.is_global = True
                logger.debug(f"property: is_global: {node.is_global}")

                # round_mode
                ceil_mode = bool(
                    nodes_op_dict.get(ceil_mode_id).attributes["value"].item()
                )
                node.round_mode = RoundMode.CEIL if ceil_mode else RoundMode.FLOOR
                logger.debug(f"property: round_mode: {node.round_mode}")
                # dilation: (N,C,H,W)
                dilation: List[int] = nodes_op_dict.get(dilation_id).attributes["value"]
                if len(dilation) == 2:
                    node.dilation = [1, 1] + dilation
                else:
                    raise ValueError(
                        f"[ERROR] Invalid dilation value format: dilation: {dilation}"
                    )
                logger.debug(f"property: dilation (N, C, H, W): {node.dilation}")
                # padding: [padding_h_before, padding_h_after, padding_w_before, padding_w_after]
                padding: List[int] = nodes_op_dict.get(padding_id).attributes["value"]
                if len(padding) == 2:
                    node.padding = [padding[0]] * 2 + [padding[1]] * 2
                else:
                    raise ValueError(
                        f"[ERROR] Invalid padding value format: padding: {padding}"
                    )
                logger.debug(f"property: padding: {node.padding}")
                # stride: [stride_h, stride_w]
                node.strides = nodes_op_dict.get(stride_id).attributes["value"]
                logger.debug(f"property: (stride_h, stride_w): {node.strides}")

            elif op_type == "add":
                input_id, other_id, alpha_id = list(layer.inputs)

                # create XModelNodeElemAdd object
                node = XModelNodeElemAdd(op_name)
                logger.debug(f"create XModelNodeElemAdd object: name: {node.op_name}")

                # bottom
                node.bottom = [input_id, other_id]
                node.inputs_tensor_shape = layer.inputs_tensor_shape[:2]

                # set coefficient
                if nodes_op_dict.get(alpha_id):
                    node.alpha = [1.0] + [
                        float(nodes_op_dict.get(alpha_id).attributes["value"].item())
                    ]
                else:
                    node.alpha = [1.0] * len(node.bottom)
                logger.debug(f"property: alpha: {node.alpha}")

            elif op_type == "permute":
                input_id, dims_id = list(layer.inputs)

                # create XModelNodePermute object
                node = XModelNodePermute(op_name)
                logger.debug(f"create XModelNodePermute object: name: {node.op_name}")

                # bottom
                node.bottom = [input_id]
                node.inputs_tensor_shape = [layer.inputs_tensor_shape[0]]

                # dims
                dims: List[int] = nodes_op_dict.get(dims_id).attributes["value"]
                node.order = dims
                logger.debug(f"property: order: {node.order}")

            elif op_type == "contiguous":
                input_id, _ = list(layer.inputs)

                node = XModelNode(op_name, "contiguous")
                logger.debug(
                    f"create XModelNode (contiguous) object: name: {node.op_name}"
                )

                # bottom
                node.bottom = [input_id]
                node.inputs_tensor_shape = [layer.inputs_tensor_shape[0]]

            elif op_type == "size":
                input_id, dims_id = list(layer.inputs)

                # create XModelNodeSize object
                node = XModelNodeSize(op_name)
                logger.debug(f"create XModelNodeSize object: name: {node.op_name}")

                # bottom
                node.bottom = [input_id]
                node.inputs_tensor_shape = [layer.inputs_tensor_shape[0]]

                # dims
                dims: List[int] = [
                    nodes_op_dict.get(dims_id).attributes["value"].data.item()
                ]
                node.dims = dims
                logger.debug(f"property: dims: {node.dims}")

                # update outputs_tensor_shape
                outputs_tensor_shape = [[len(node.dims)]]

            elif op_type == "listconstruct":
                inputs_id, params_id = [], []
                for iname in list(layer.inputs):
                    if len(nodes_op_dict.get(iname).attributes) == 0:
                        inputs_id.append(iname)
                    else:
                        params_id.append(iname)

                node = XModelNode(op_name, "listconstruct")
                logger.debug(
                    f"create XModelNode (contiguous) object: name: {node.op_name}"
                )

                # bottom
                node.bottom = inputs_id

                # constants
                if len(params_id) > 0:
                    constants = []
                    for param_id in params_id:
                        constants.append(
                            nodes_op_dict.get(param_id).attributes["value"].data.item()
                        )
                    node.tmp_params["constants"] = constants
                    logger.debug(f"property: constants: {node.tmp_params['constants']}")

            elif op_type == "view":
                input_id, shape_id = list(layer.inputs)

                # create XModelNodeReshape object
                node = XModelNodeReshape(op_name)
                logger.debug(f"create XModelNodeReshape object: name: {node.op_name}")

                # bottom
                node.bottom = [input_id]
                node.inputs_tensor_shape = [layer.inputs_tensor_shape[0]]

                # shape
                if len(nodes_op_dict.get(shape_id).attributes) == 0:
                    node.bottom.append(shape_id)
                else:
                    node.shape = nodes_op_dict.get(shape_id).attributes["value"].item()
                    logger.debug(f"property: shape: {node.shape}")

            elif op_type == "to":
                assert len(layer.inputs) == 4, f"[ERROR] Unsupported PyTorch 'To' op."

                input_id, dtype_id, non_blocking_id, copy_id = list(layer.inputs)

                node = XModelNode(op_name, "to_dtype")
                logger.debug(f"create XModelNode (to) object: name: {node.op_name}")

                # bottom
                node.bottom = [input_id]
                # ! notice: no inputs_tensor_shape

                # dtype
                dtype = nodes_op_dict.get(dtype_id).attributes["value"].data.item()
                assert (
                    dtype in TORCH_DTYPE_TO_NUMPY_DTYPE
                ), f"[ERROR] Unsupported torch dtype: {dtype}"
                dtype = TORCH_DTYPE_TO_NUMPY_DTYPE.get(dtype)
                node.tmp_params["dtype"] = dtype
                logger.debug(f"property: dtype: {node.tmp_params['dtype']}")

                # non_blocking
                non_blocking = bool(
                    nodes_op_dict.get(non_blocking_id).attributes["value"].data.item()
                )
                node.tmp_params["non_blocking"] = non_blocking
                logger.debug(
                    f"property: non_blocking: {node.tmp_params['non_blocking']}"
                )

                # copy
                copy = bool(nodes_op_dict.get(copy_id).attributes["value"].data.item())
                node.tmp_params["copy"] = copy
                logger.debug(f"property: copy: {node.tmp_params['copy']}")

            elif op_type == "mul":
                input_id, other_id = list(layer.inputs)

                # create XModelNode object
                node = XModelNode(op_name, "mul")
                logger.debug(f"create XModelNode (mul) object: name: {node.op_name}")

                # bottom
                node.bottom = [input_id]

                # other
                other = nodes_op_dict.get(other_id).attributes["value"].data.item()
                node.tmp_params["other"] = other
                logger.debug(f"property: other: {node.tmp_params['other']}")

            elif op_type == "floor":
                # create XModelNode object
                node = XModelNode(op_name, "floor")
                logger.debug(f"create XModelNode (floor) object: name: {node.op_name}")

                # bottom
                node.bottom = layer.inputs

            elif op_type == "cat":
                input_id, dim_id = list(layer.inputs)

                # create XModelNodeConcat object
                node = XModelNodeConcat(op_name)
                logger.debug(f"create XModelNodeConcat object: name: {node.op_name}")

                # bottom
                node.bottom = [input_id]
                node.inputs_tensor_shape = [layer.inputs_tensor_shape[0]]

                # axis
                node.axis = nodes_op_dict.get(dim_id).attributes["value"].data.item()
                logger.debug(f"property: axis: {node.axis}")

            elif op_type == "upsamplenearest2d":
                assert len(layer.inputs) == 2
                input_id, scale_id = list(layer.inputs)

                # create XModelNodeUpsample object
                node = XModelNodeUpsample(op_name)
                logger.debug(f"create XModelNodeUpsample object: name: {node.op_name}")

                # bottom
                node.bottom = [input_id]
                node.inputs_tensor_shape = [layer.inputs_tensor_shape[0]]

                # mode
                node.mode = "nearest"
                logger.debug(f"property: mode: {node.mode}")

                # scale
                if len(nodes_op_dict.get(scale_id).attributes) == 0:
                    node.bottom.append(scale_id)
                else:
                    node.scale = (
                        nodes_op_dict.get(scale_id).attributes["value"].data.item()
                    )

            else:
                # ! debug
                # print(f"op_type={op_type}, op_name={op_name}")
                raise ValueError(
                    f"*** Unsupported op: type: {op_type}, name: {op_name}"
                )

            # outputs
            node.top = list(layer.outputs)
            # outputs_tensor_shape
            node.outputs_tensor_shape = outputs_tensor_shape

            # update xmodel
            xmodel.add_xnode(node)

        assert len(xmodel.xnodes) == len(input_layers) + len(
            node_dict
        ), f"[ERROR] Mismatched node number: # of xnodes: {len(xmodel.xnodes)}, while # of input layers: {len(input_layers)}, # of op layers: {len(node_dict)}."
        logger.info("* end: translate pytorch layers to xmodel nodes")

        # * perform platform-specific optimizations
        OptManager.dispatch(xmodel, "xnnc")

        # * special pass
        cls.__split_quantizekrelu_into_relu_and_fix_pass(xmodel, fix_neuron_to_split)

        # * special pass: data validation
        cls.__suffix_op_name_with_op_type(xmodel)

        # topsort xnodes
        xmodel.topsort()

        # ! debug: render xmodel as svg file
        # xmodel.render()

        return xmodel

    @classmethod
    def __load_raw_model(cls, model_files: List[Path]) -> (str, torch.nn.Module):
        """
        Load raw model files from a list of specified file paths.

        Parameters:
            model_files: a list of specified file paths, which should specify the paths to both caffemodel and prototxt files.

        Returns:
            str: model name
            torch.nn.Module: a PyTorch model instance.
        """
        # check model files
        if len(model_files) != 1:
            logger.error(
                "The 'model_files' argument should contain a Python script named get_model.py."
            )
            sys.exit(1)

        # load model script
        model_file: Path = model_files[0]
        assert model_file.stem == "get_model" and model_file.suffix == ".py"

        # check model architecture file
        logger.debug("check file validity: {0}".format(model_file))
        passed, err_msg, model_file = helper.check_filepath(
            model_file, extension=model_file.suffix
        )
        if not passed:
            logger.error(err_msg)
            sys.exit(1)

        def module_from_file(file_path):
            import importlib, importlib.util

            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        module = module_from_file(model_file)
        model_name, model = getattr(module, "get_model")()

        return model_name, model

    @classmethod
    def __compute_quantize_pos(cls, bit_width: int, log2t: float, signed: bool) -> int:
        """Compute the quantization position

        Parameters
        ----------
        bit_width : int
            the info of bit width of the
        log2t : float
            the clip values
        signed : bool
            the flag for positive/negative values.

        Returns
        -------
        int
            the quantization position
        """
        return int(
            bit_width - math.ceil(log2t) - 1 if signed else bit_width - math.ceil(log2t)
        )

    @classmethod
    def __split_quantizekrelu_into_relu_and_fix_pass(
        cls, xmodel: XModel, fix_neuron_to_split: List[XModelNodeFixNeuron]
    ) -> NoReturn:
        """Split quantizekRelu into Relu and Fix neuron

        Parameters
        ----------
        xmodel : XModel
            an XModel instance
        fix_neuron_to_split: List[XModelNodeFixNeuron]
            list of fixneuron nodes to be split into relu and fixneuron
        """
        if fix_neuron_to_split is None or len(fix_neuron_to_split) == 0:
            return

        logger.info(
            "* start: special pass: split Rulu into Relu + FixNeuron when the Relu's signed is False."
        )

        for xnode in fix_neuron_to_split:
            # create XModelNodeRelu object
            node = XModelNodeRelu(xnode.op_name + "_relu")
            logger.debug(f"create XModelNodeRelu object: name: {node.op_name}")

            # bottom and top
            node.bottom = xnode.bottom
            node.inputs_tensor_shape = xnode.inputs_tensor_shape
            node.top = [xnode.op_name]
            node.outputs_tensor_shape = node.inputs_tensor_shape

            # add node to xmodel
            xmodel.add_xnode(node)

            # update parent nodes
            for pname in xnode.bottom:
                pnode = xmodel.get_xnode_by_name(pname)
                assert pnode is not None
                idx = pnode.top.index(xnode.op_name)
                # replace xnode.op_name with node.op_name
                pnode.top[idx] = node.op_name

            # update xnode
            xnode.bottom = [node.op_name]
            xnode.inputs_tensor_shape = node.outputs_tensor_shape

        # sort
        xmodel.topsort()

        logger.info(
            "* end: special pass: split Rulu into Relu + FixNeuron when the Relu's signed is False."
        )

    @classmethod
    def __reduce_shape_listconstruct_to_reshape(cls, xmodel: XModel) -> NoReturn:
        """Merge shape+listconstruct into reshape as the value of the shape property.

        Parameters
        ----------
        xmodel : XModel
            XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."

        # find all shape+lc+reshape node sequences
        size_lc_view = []
        for xnode in xmodel.xnodes:
            if xnode.op_type == "shape":
                if xnode.top is not None and len(xnode.top) == 1:
                    cnode = xmodel.get_xnode_by_name(xnode.top[0])
                    assert (
                        cnode is not None
                    ), f"Not found in xmodel: node name: {xnode.top[0]}."
                    if cnode.op_type == "listconstruct":
                        if cnode.top is not None and len(cnode.top) == 1:
                            ccnode = xmodel.get_xnode_by_name(cnode.top[0])
                            assert (
                                ccnode is not None
                            ), f"Not found in xmodel: node name: {cnode.top[0]}."
                            if ccnode.op_type == "reshape":
                                size_lc_view.append((xnode, cnode, ccnode))

        # compute the shape value by computing Shape + ListConstruct
        while size_lc_view:
            size, lc, view = size_lc_view.pop()

            # update the shape property
            assert (
                view.shape is None
            ), f"The shape property of Reshape op is not None: {view.shape}."
            assert (
                view.outputs_tensor_shape is not None
                and len(view.outputs_tensor_shape) == 1
            )
            view.shape = view.outputs_tensor_shape[0]

            # update bottom and top
            assert size.bottom is not None and len(size.bottom) == 1
            pnode = xmodel.get_xnode_by_name(size.bottom[0])
            assert (
                pnode is not None
            ), f"Not found in xmodel: node name: {size.bottom[0]}."
            idx = pnode.top.index(size.op_name)
            pnode.top = pnode.top[:idx] + pnode.top[idx + 1 :]
            idx = view.bottom.index(lc.op_name)
            view.bottom = view.bottom[:idx] + view.bottom[idx + 1 :]

            # remove Shape and ListConstruct nodes
            size.bottom = size.top = []
            lc.bottom = lc.top = []
            xmodel.remove_xnode(size)
            xmodel.remove_xnode(lc)

    @classmethod
    def __suffix_op_name_with_op_type(cls, xmodel: XModel) -> NoReturn:
        assert xmodel is not None, "'xmodel' should not be None."

        # topsort
        xmodel.topsort()

        xnodes = [x for x in xmodel.xnodes]
        for xnode in xnodes:
            new_name = xnode.op_name + "_" + xnode.op_type
            new_name = new_name.replace("/", ".")
            if not xmodel.rename_xnode(xnode, new_name):
                raise Exception(
                    f"[ERROR] Unable to rename xnode: type: {xnode.op_type}, name: {xnode.op_name}"
                )

    @classmethod
    def __extract_quantize_pos(cls, xmodel: XModel) -> NoReturn:
        assert xmodel is not None, "'xmodel' should not be None."

        attr_names = [
            "op_name",
            "op_type",
            "quant_in",
            "quant_out",
            "quant_weights",
            "quant_bias",
        ]
        res = []
        for xnode in xmodel.xnodes:
            quant_info = {}

            # parse quant info of current node
            node_info = {}
            for attr_name in attr_names:
                if hasattr(xnode, attr_name):
                    node_info[attr_name] = getattr(xnode, attr_name)
            quant_info["node"] = node_info

            # parse quant info of parent nodes
            parents = []
            if xnode.bottom is not None and len(xnode.bottom) > 0:
                for pname in xnode.bottom:
                    pnode_info = {}
                    pnode = xmodel.get_xnode_by_name(pname)
                    assert pnode is not None, f"[ERROR] Not found parent node: {pname}."
                    for attr_name in attr_names:
                        if hasattr(pnode, attr_name):
                            pnode_info[attr_name] = getattr(pnode, attr_name)
                    parents.append(pnode_info)
            quant_info["parents"] = parents

            # parse quant info of child nodes
            children = []
            if xnode.top is not None and len(xnode.top) > 0:
                for cname in xnode.top:
                    cnode_info = {}
                    cnode = xmodel.get_xnode_by_name(cname)
                    assert cnode is not None, f"[ERROR] Not found child node: {cname}."
                    for attr_name in attr_names:
                        if hasattr(cnode, attr_name):
                            cnode_info[attr_name] = getattr(cnode, attr_name)
                    children.append(cnode_info)
            quant_info["children"] = children

            res.append(quant_info)

        import json

        json_str = json.dumps(res, indent=4)
        with open("quant_info.json", "w") as f:
            f.write(json_str)
