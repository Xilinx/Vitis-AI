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
import sys
from enum import IntEnum
from pathlib import Path, PurePath
from typing import List

import numpy as np
from google.protobuf import json_format
from tqdm import tqdm

from xnnc.entity.xmodel import XModel
from xnnc.entity.xnode import *
from xnnc.proto.onnx_pb2 import proto3_pb2 as onnx_pb2
from xnnc.translator.base_translator import ITranslator
from xnnc.utils import helper
from xnnc.utils.helper import Layout

# create logger
logger = logging.getLogger(__name__)


class AttributeType(IntEnum):
    """
    ONNX Attribute Type

    Reference:
        https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3
    """

    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10


class TensorDataType(IntEnum):
    """
    ONNX Tensor data type

    Reference:
        https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3
    """

    UNDEFINED = 0
    # Basic types.
    FLOAT = 1  # float
    UINT8 = 2  # uint8_t
    INT8 = 3  # int8_t
    UINT16 = 4  # uint16_t
    INT16 = 5  # int16_t
    INT32 = 6  # int32_t
    INT64 = 7  # int64_t
    STRING = 8  # string
    BOOL = 9  # bool

    # IEEE754 half-precision floating-point format (16 bits wide).
    # This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    FLOAT16 = 10

    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14  # complex with float32 real and imaginary components
    COMPLEX128 = 15  # complex with float64 real and imaginary components

    # Non-IEEE floating-point format based on IEEE754 single-precision
    # floating-point number truncated to 16 bits.
    # This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    BFLOAT16 = 16


class ONNXTranslator(ITranslator):
    """
    Convert ONNX model architecture into XModel one
    """

    @classmethod
    def to_xmodel(cls, model_files: List[Path], layout: Layout = Layout.NCHW) -> XModel:
        logger.info("start: onnx_to_xmodel conversion")

        # load model architecture
        model_name, graph_proto = cls.__load_raw_model(model_files)

        # create an xmodel object
        xmodel = cls.__create_xmodel(model_name, graph_proto)

        logger.info("start: onnx_to_xmodel conversion")
        return xmodel

    @classmethod
    def __load_raw_model(cls, model_files: List[Path]) -> (str, dict):

        if len(model_files) != 1:
            logger.error(
                "The 'model_files' argument should contain only one '.onnx' file."
            )
            sys.exit(1)

        # load model architecture file
        onnx_model: Path = None
        if model_files[0].suffix == ".onnx":
            onnx_model = model_files[0]
        if onnx_model is None:
            logger.error("Not found '.onnx' file.")
            sys.exit(1)

        # check onnx_model
        logger.debug("check file validity: {0}".format(onnx_model))
        passed, err_msg, onnx_model = helper.check_filepath(
            onnx_model, extension=".onnx"
        )
        if not passed:
            logger.error(err_msg)
            sys.exit(1)

        logger.info("load {0}".format(onnx_model))
        model_proto = onnx_pb2.ModelProto()
        with open(onnx_model, "rb") as f:
            model_proto.ParseFromString(f.read())
        # model_dict = json_format.MessageToDict(model_proto, including_default_value_fields=False)
        # logger.debug("check if model_dict has 'graph' key.")
        # assert 'graph' in model_dict, "No key is named as 'graph' in model file"

        # # graph_dict
        # graph_dict = model_dict.get('graph')
        # logger.debug("check if graph_dict has 'initializer' key.")
        # assert 'initializer' in graph_dict
        # logger.debug("check if graph_dict has 'input' key.")
        # assert 'input' in graph_dict
        # logger.debug("check if graph_dict has 'node' key.")
        # assert 'node' in graph_dict
        # logger.debug("check if graph_dict has 'output' key.")
        # assert 'output' in graph_dict

        return onnx_model.stem, model_proto.graph

    @classmethod
    def __create_xmodel(
        cls, name: str, graph_proto: onnx_pb2.GraphProto, layout: Layout = Layout.NCHW
    ) -> XModel:
        """
        Create XModel object from ONNX model layers.

        Parameters:
            name: model name.
            graph_proto: ONNX GraphProto object
            layout: data layout.

        Returns:
            An XModel object.
        """
        assert name is not None, "The argument 'name' should not be None."
        assert graph_proto is not None, "The argument 'graph_proto' should not be None."

        # create XModel object
        xmodel = XModel(name, "onnx")

        op_name_list = []
        for layer in graph_proto.node:
            op_name_list.append(layer.name)

        # create a dict for named tensors
        # key: name of tensor
        # value: named tensor object
        named_tensor_dict = {}
        for named_tensor in graph_proto.initializer:
            named_tensor_dict[named_tensor.name] = named_tensor

        # translate onnx node to xmodel node
        logger.info("translate onnx nodes to xmodel nodes")
        # layers = graph_dict.get('node')
        pbar = tqdm(graph_proto.node)
        for layer in pbar:
            pbar.set_description("[INFO] parse raw model")

            # get op name and op_type
            op_name = layer.name
            op_type = layer.op_type.lower()

            # get bottom and top
            bottom = [i for i in layer.input if i in op_name_list]
            top = [o for o in layer.output if o in op_name_list]

            # translate current layer to XModel node
            if op_type == "conv":
                # reference: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
                assert (
                    len(layer.input) >= 2
                ), "'conv' defined in onnx should have two or more input sources."

                # default values for optional properties: strides, padding, dilation, group
                # [kernel_h, kernel_w]
                ksize: List[int] = None
                # [stride_h, stride_w]
                strides: List[int] = [1, 1]
                # [top, down, left, right]
                padding: List[int] = [0, 0, 0, 0]
                # [top, down, left, right]
                dilation: List[int] = [1, 1, 1, 1]
                # number of groups input channels and output channels are divided into
                group: int = 1

                assert len(layer.attribute) > 0
                # set padding, kernel, strides, dilations, group
                for attr in layer.attribute:
                    attr_name = attr.name.lower()
                    # padding
                    if attr_name == "pads":
                        if attr.type == AttributeType.INTS:
                            padding = attr.ints
                        else:
                            raise ValueError("Unsupported type: " + attr.get("type"))
                    # kernel size
                    elif attr_name == "kernel_shape":
                        if attr.type == AttributeType.INTS:
                            ksize = attr.ints
                        else:
                            raise ValueError("Unsupported type: " + attr.type)
                    # strides
                    elif attr_name == "strides":
                        if attr.type == AttributeType.INTS:
                            strides = attr.ints
                        else:
                            raise ValueError("Unsupported type: " + attr.type)
                    # dilation
                    elif attr_name == "dilations":
                        if attr.type == AttributeType.INTS:
                            dilation = attr.ints
                            if len(dilation) == 2:
                                dilation = [
                                    dilation[0],
                                    dilation[0],
                                    dilation[1],
                                    dilation[1],
                                ]
                        else:
                            raise ValueError("Unsupported type: " + attr.type)
                    # group
                    elif attr_name == "group":
                        if attr.type == AttributeType.INT:
                            group = attr.i
                        else:
                            raise ValueError("Unsupported type: " + attr.type)
                    else:
                        raise ValueError("Unsupported attribute: " + attr_name)
                # Notice: kernel_size is required
                assert (
                    ksize is not None
                ), "Error: kernel size must be present in convolution op."

                # set ceil_mode
                ceil_mode: str = "floor"

                # get weights and bias
                weights: np.ndarray = None
                bias_term: bool = False
                bias: np.ndarray = None
                # input[1]: weights, input[2]: bias
                # get weights
                weight_tensor = named_tensor_dict.get(layer.input[1])
                assert weight_tensor is not None
                weight_shape = weight_tensor.dims
                weight_dtype = None
                weight_data = None
                if weight_tensor.data_type == TensorDataType.FLOAT:
                    weight_data = weight_tensor.float_data
                    weight_dtype = np.float32
                else:
                    raise ValueError(
                        "Unsupported tensor type: " + weight_tensor.data_type
                    )
                weights = np.array(weight_data, dtype=weight_dtype).reshape(
                    weight_shape
                )
                # get bias
                if len(layer.input) == 3:
                    bias_term = True
                    bias_tensor = named_tensor_dict.get(layer.input[2])
                    assert bias_tensor is not None
                    bias_shape = bias_tensor.dims
                    bias_dtype = None
                    bias_data = None
                    if bias_tensor.data_type == TensorDataType.FLOAT:
                        bias_data = bias_tensor.float_data
                        bias_dtype = np.float32
                    else:
                        raise ValueError(
                            "Unsupported tensor type: " + bias_tensor.data_type
                        )
                    bias = np.array(bias_data, dtype=bias_dtype).reshape(bias_shape)

                # create XModelNode object
                node = None
                if group == weights.shape[1]:  # group == in_channels
                    node = XModelNodeConv2dDepthwise(op_name, ksize)
                elif 1 <= group < weights.shape[1]:
                    node = XModelNodeConv2d(op_name, ksize)
                else:
                    raise ValueError("Not supported group value: {}".format(group))
                assert node is not None
                # set node properties
                node.group = group
                node.strides = strides
                node.padding = padding
                node.pad_mode = "PADDING"
                node.dilation = dilation
                node.ceil_mode = ceil_mode
                node.weights = weights
                node.bias_term = bias_term
                if node.bias_term:
                    node.bias = bias

            elif op_type in ["averagepool", "globalaveragepool"]:
                # set default values: kernel, strides, padding
                # [kernel_h, kernel_w]
                ksize = [0, 0]
                # [stride_h, stride_w]
                strides = [1, 1]
                # [pad_top, pad_down, pad_left, pad_right]
                padding = [0, 0, 0, 0]

                is_global = False
                if op_type == "globalaveragepool":
                    is_global = True

                # parse parameters if op is AvgPool
                if not is_global:
                    # set kernel, padding, strides
                    for attr in layer.attribute:
                        attr_name = attr.name.lower()
                        assert attr_name is not None, "Attribute should have a name."
                        # kernel: [kernel_h, kernel_w]
                        if attr_name == "kernel_shape":
                            if attr.type == AttributeType.INTS:
                                ksize = attr.ints
                            else:
                                raise ValueError("Unsupported type: " + attr.type)
                        # padding: [pad_top, pad_down, pad_left, pad_right]
                        elif attr_name == "pads":
                            if attr.type == AttributeType.INTS:
                                padding = attr.ints
                            else:
                                raise ValueError("Unsupported type: " + attr.type)
                        # strides: [stride_h, stride_w]
                        elif attr_name == "strides":
                            if attr.type == AttributeType.INTS:
                                strides = attr.ints
                            else:
                                raise ValueError("Unsupported type: " + attr.type)
                        else:
                            raise ValueError("Unsupported attribute: " + attr_name)
                    # Notice: ksize is required
                    assert (
                        ksize is not None
                    ), "Error: kernel size must be present in maxpool op."

                else:
                    # For GlobalAvgPool, reference the definition in
                    # https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalAveragePool
                    pass

                # create XModelNodeAvgPool object
                node = XModelNodeAvgPool(op_name, ksize)

                # set properties
                node.is_global = is_global
                node.strides = strides
                node.padding = padding
                node.pad_mode = "PADDING"

                if not node.is_global:
                    assert node.kernel_size is not None, "Kernel size MUST have values."

            elif op_type == "maxpool":
                assert len(layer.attribute) > 0
                # [kernel_h, kernel_w]
                ksize = [0, 0]
                # [stride_h, stride_w]
                strides = [1, 1]
                # [top, down, left, right]
                padding = [0, 0, 0, 0]
                # set kernel, padding, strides
                for attr in layer.attribute:
                    attr_name = attr.name.lower()
                    assert attr_name is not None, "Attribute should have a name."
                    # kernel: [kernel_h, kernel_w]
                    if attr_name == "kernel_shape":
                        if attr.type == AttributeType.INTS:
                            ksize = attr.ints
                        else:
                            raise ValueError("Unsupported type: " + attr.type)
                    # padding: [pad_top, pad_down, pad_left, pad_right]
                    elif attr_name == "pads":
                        if attr.type == AttributeType.INTS:
                            padding = attr.ints
                        else:
                            raise ValueError("Unsupported type: " + attr.type)
                    # strides: [stride_h, stride_w]
                    elif attr_name == "strides":
                        if attr.type == AttributeType.INTS:
                            strides = attr.ints
                        else:
                            raise ValueError("Unsupported type: " + attr.type)
                    else:
                        raise ValueError("Unsupported attribute: " + attr_name)
                # Notice: ksize is required
                assert (
                    ksize is not None
                ), "Error: kernel size must be present in maxpool op."

                # create XModelNodeMaxPool object
                node = XModelNodeMaxPool(op_name, ksize)
                node.strides = strides
                node.padding = padding
                node.pad_mode = "PADDING"

            elif op_type == "relu":
                # create XModelNodeRelu object
                node = XModelNodeRelu(op_name)

                # set negative_slope
                node.alpha = 0

            elif op_type == "batchnormalization":
                # create XModelNodeBatchNorm object
                node = XModelNodeBatchNorm(op_name)

                # get epsilon, momentum, spatial
                if len(layer.attribute) > 0:
                    for attr in layer.attribute:
                        attr_name = attr.name.lower()
                        # epsilon
                        if attr_name == "epsilon":
                            if attr.type == AttributeType.FLOAT:
                                node.epsilon = attr.f
                            else:
                                raise ValueError("Unsupported type: " + attr.type)
                        # momentum
                        elif attr_name == "momentum":
                            if attr.type == AttributeType.FLOAT:
                                node.momentum = attr.f
                            else:
                                raise ValueError("Unsupported type: " + attr.type)
                        # spatial
                        elif attr_name == "spatial":
                            if attr.type == AttributeType.INT:
                                # TODO: samshin: create spatial property in XModelNodeBatchNorm
                                spatial = attr.i
                            else:
                                raise ValueError("Unsupported type: " + attr.type)
                        else:
                            raise ValueError("Unsupported attribute: " + attr_name)

                # set gamma, beta, running_mean, running_var
                # scale tensor of shape (C)
                gamma_tensor = named_tensor_dict.get(layer.input[1])
                gamma = None
                if gamma_tensor.data_type == AttributeType.FLOAT:
                    gamma = np.array(gamma_tensor.float_data, dtype=np.float32).reshape(
                        gamma_tensor.dims
                    )
                else:
                    raise ValueError("Unsupported type: " + gamma_tensor.data_type)
                node.gamma = gamma

                # bias tensor of shape (C)
                beta_tensor = named_tensor_dict.get(layer.input[2])
                beta = None
                if beta_tensor.data_type == AttributeType.FLOAT:
                    beta = np.array(beta_tensor.float_data, dtype=np.float32).reshape(
                        beta_tensor.dims
                    )
                else:
                    raise ValueError("Unsupported type: " + beta_tensor.data_type)
                node.beta = beta

                # mean tensor of shape (C)
                mean_tensor = named_tensor_dict.get(layer.input[3])
                mean = None
                if mean_tensor.data_type == AttributeType.FLOAT:
                    mean = np.array(mean_tensor.float_data, dtype=np.float32).reshape(
                        mean_tensor.dims
                    )
                else:
                    raise ValueError("Unsupported type: " + mean_tensor.data_type)
                node.mean = mean

                # variance tensor of shape (C)
                variance_tensor = named_tensor_dict.get(layer.input[4])
                variance = None
                if variance_tensor.data_type == AttributeType.FLOAT:
                    variance = np.array(
                        variance_tensor.float_data, dtype=np.float32
                    ).reshape(variance_tensor.dims)
                else:
                    raise ValueError("Unsupported type: " + variance_tensor.data_type)
                node.variance = variance

            elif op_type == "gemm":
                # reference https://github.com/onnx/onnx/blob/master/docs/Operators.md#gemm
                assert len(layer.input) == 3
                assert len(layer.attribute) > 0

                # get attributes
                alpha: float = 1.0
                beta: float = 1.0
                transA: bool = False
                transB: bool = False
                for attr in layer.attribute:
                    attr_name = attr.name.lower()
                    if attr_name == "alpha":
                        if attr.type == AttributeType.FLOAT:
                            alpha = attr.f
                        else:
                            raise ValueError("Unsupported type: " + attr.type)
                    elif attr_name == "beta":
                        if attr.type == AttributeType.FLOAT:
                            beta = attr.f
                        else:
                            raise ValueError("Unsupported type: " + attr.type)
                    elif attr_name == "transa":
                        if attr.type == AttributeType.INT:
                            transA = bool(attr.i)
                        else:
                            raise ValueError("Unsupported type: " + attr.type)
                    elif attr_name == "transb":
                        if attr.type == AttributeType.INT:
                            transB = bool(attr.i)
                        else:
                            raise ValueError("Unsupported type: " + attr.type)
                    else:
                        raise ValueError("Unsupported attribute: " + attr_name)

                # get B input
                B_tensor = named_tensor_dict.get(layer.input[1])
                B_shape = B_tensor.dims
                if B_tensor.data_type == TensorDataType.FLOAT:
                    B_type = np.float32
                    B_data = B_tensor.float_data
                B = np.array(B_data, dtype=B_type).reshape(B_shape)

                # get C input
                C_tensor = named_tensor_dict.get(layer.input[2])
                C_shape = C_tensor.dims
                if C_tensor.data_type == TensorDataType.FLOAT:
                    C_type = np.float32
                    C_data = C_tensor.float_data
                C = np.array(C_data, dtype=C_type).reshape(C_shape)

                # TODO: samshin: create XModelNode object to hold all these things above
                node = XModelNode(op_name, op_type="gemm")

            elif op_type == "sum":
                raise NotImplementedError("Softmax is not implemented.")

            elif op_type == "reshape":
                # reference https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
                assert len(layer.input) == 2
                # get shape param
                iname = layer.input[1]
                shape_tensor = named_tensor_dict.get(iname)
                assert shape_tensor is not None
                shape = None
                if shape_tensor.data_type == TensorDataType.INT64:
                    shape = shape_tensor.int64_data

                # create XModelNodeReshape object
                node = XModelNodeReshape(op_name, shape)

            elif op_type == "add":
                # create XModelNodeElemAdd object
                node = XModelNodeElemAdd(op_name)

                # set coefficient
                node.alpha = [1] * len(bottom)

            elif op_type == "softmax":
                # create XModelNodeSoftmax object
                node = XModelNodeSoftmax(op_name)

                # TODO samshin: set axis
                node.axis = 1

            else:
                raise ValueError("Unsupported op type: {}".format(op_type))

            # udpate bottom and top
            node.bottom, node.top = bottom, top

            # update xmodel
            xmodel.update_xnode(node)

        # create input op
        logger.info("create input node")
        layer = graph_proto.input[0]  # graph_dict.get('input')[0]
        assert layer is not None
        op_name = layer.name
        # create XModelNodeInput object
        root = XModelNodeInput(op_name)
        root.shape = [d.dim_value for d in layer.type.tensor_type.shape.dim]
        # TODO: samshin: add elem type for XModelNodeInput
        elem_type = TensorDataType(layer.type.tensor_type.elem_type)
        # update top of root op
        logger.debug("update top of root op")
        for node in xmodel.nodes:
            if len(node.bottom) > 0:
                for iname in node.bottom:
                    if iname == root.op_name:
                        root.top.append(iname)
        # update xmodel
        xmodel.xnodes = [root] + xmodel.xnodes
        return xmodel
