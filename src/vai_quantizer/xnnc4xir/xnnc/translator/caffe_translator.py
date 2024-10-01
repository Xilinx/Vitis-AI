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

import logging
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from google.protobuf import text_format
from tqdm import tqdm
from xnnc.ir.enums import Layout
from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import *
from xnnc.ir.xnode import RoundMode
from xnnc.optimizer import OptManager
from xnnc.proto.caffe_fixed_neuron_pb2 import caffe_fixed_neuron_pb2 as caffe_pb2
from xnnc.tensor.xtensor import XTensor
from xnnc.tensor.xtensor import XTensorCompute as tc
from xnnc.translator.base_translator import ITranslator
from xnnc.utils import helper

# create logger
logger = logging.getLogger(__name__)


class CaffeTranslator(ITranslator):
    """
    Convert a Caffe model into XModel object.
    """

    @classmethod
    def to_xmodel(
        cls,
        model_files: List[Path],
        layout: Layout = Layout.NCHW,
        in_shapes: Optional[Union[List[List[int]], Dict[str, List[int]]]] = None,
        *args,
        **kwargs,
    ) -> XModel:
        logger.info("start: caffe_to_xmodel conversion")

        # load model architecture
        model_name, proto_param, model_param = cls.__load_raw_model(model_files)

        # create an xmodel object
        xmodel = cls.__create_xmodel(
            model_name,
            proto_param,
            model_param,
            in_shapes=in_shapes,
            batchsize=kwargs.get("batchsize"),
        )

        logger.info("end: caffe_to_xmodel conversion")

        return xmodel

    @classmethod
    def __load_raw_model(
        cls, model_files: List[Path]
    ) -> Tuple[str, "NetParameter", "NetParameter"]:
        """
        Load raw model files from a list of specified file paths.

        Parameters:
            model_files: a list of specified file paths, which should specify the paths to both caffemodel and prototxt files.

        Returns:
            str: model name
            dict: key is layer/node name, value is a layer dict.
        """
        # check model files
        if len(model_files) != 2:
            logger.error(
                "The 'model_files' argument has less or more files than the expected (2 files)."
            )
            sys.exit(1)

        # load prototxt and caffemodel files
        prototxt: Path = None
        caffemodel: Path = None
        for path in model_files:
            if path.suffix == ".prototxt":
                prototxt = path
            if path.suffix == ".caffemodel":
                caffemodel = path
        if prototxt is None:
            logger.error("Not found '.prototxt' file.")
            sys.exit(1)
        if caffemodel is None:
            logger.error("Not found '.caffemodel' file.")
            sys.exit(1)

        # check prototxt
        logger.debug("check file validity: {0}".format(prototxt))
        passed, err_msg, prototxt = helper.check_filepath(
            prototxt, extension=".prototxt"
        )
        if not passed:
            logger.error(err_msg)
            sys.exit(1)

        # check caffemodel
        logger.debug("check file validity: {0}".format(caffemodel))
        passed, err_msg, caffemodel = helper.check_filepath(
            caffemodel, extension=".caffemodel"
        )
        if not passed:
            logger.error(err_msg)
            sys.exit(1)

        # load prototxt
        logger.info("load {0}".format(prototxt))
        proto_net_param = caffe_pb2.NetParameter()
        with open(prototxt, mode="r") as pf:
            text_format.Merge(pf.read(), proto_net_param)

        # load caffemodel
        logger.info("load {0}".format(caffemodel))
        model_net_param = caffe_pb2.NetParameter()
        with open(caffemodel, "rb") as pf:
            model_net_param.ParseFromString(pf.read())

        return prototxt.stem, proto_net_param, model_net_param

    @classmethod
    def __create_xmodel(
        cls,
        name: str,
        proto_param,
        model_param,
        layout: Layout = Layout.NCHW,
        in_shapes: Optional[Union[List[List[int]], Dict[str, List[int]]]] = None,
        batchsize: int = 1,
    ) -> XModel:
        """
        Create XModel object from Caffe model layers.

        Parameters:
            name: model name.
            layers: Caffe model layers.
            layout: data layout.

        Returns:
            An XModel object.
        """
        assert name is not None, "The argument 'name' should not be None."
        assert proto_param is not None
        assert model_param is not None

        # create an xmodel
        xmodel = XModel(name, "caffe", layout.name)

        logger.info(f"numb. of layers in caffemodel: {len(model_param.layer)}")
        logger.info(f"numb. of layers in prototxt: {len(proto_param.layer)}")

        # * zip layer pairs
        layer_dict: Dict[str, List[Any, Any]] = {}
        input_layer_appeared = False
        for layer in proto_param.layer:
            if layer.type.lower() == "input":
                input_layer_appeared = True
            elif layer.type.lower() == "imagedata":
                continue
            if layer.name not in layer_dict:
                layer_dict[layer.name] = [layer, None]
            else:
                raise ValueError(f"[ERROR] duplicated layer: name: {layer.name}")
        for layer in model_param.layer:
            if layer.name in layer_dict:
                layer_dict[layer.name][1] = layer

        # ! do not remove
        # indicate if fix neuron layer is present or not
        fix_neuron_appeared: bool = True

        # * translate Caffe layers into XModel nodes
        logger.info("* start: translate caffe layers to xmodel nodes")
        # preprocess input layer
        if not input_layer_appeared:
            # create XModelNodeInput object
            node = XModelNodeInput(proto_param.input[0])
            logger.debug(f"create XModelNodeInput object: name: {node.op_name}")
            # set shape
            if len(proto_param.input_shape) > 0:
                node.shape = proto_param.input_shape[0].dim
            logger.debug(f"property: shape (N,C,H,W): {node.shape}")
            # update xmodel
            xmodel.add_xnode(node)
        # translate each layer into xnode
        for name, pair in tqdm(
            layer_dict.items(),
            total=len(layer_dict),
            desc="[INFO] parse raw model",
            bar_format="{desc:27}:{percentage:3.0f}%|{bar}{r_bar:50}",
        ):
            layer_proto_param, layer_model_param = pair

            # get op name
            op_name: str = layer_proto_param.name
            # get op type (case sensitive)
            op_type: str = layer_proto_param.type.lower()
            logger.debug(f"*** source layer info: type: {op_type}, name: {op_name}")

            # get bottom and top
            bottom = [b for b in layer_proto_param.bottom if b != op_name]
            assert len(bottom) == len(
                set(bottom)
            ), f"[ERROR] Invalid prototxt file: duplicate names found in the bottom field of layer (name: {layer_proto_param.name}) in prototxt file: {layer_proto_param.bottom}"
            top = [t for t in layer_proto_param.top if t != op_name]
            assert len(top) == len(
                set(top)
            ), f"[ERROR] Invalid prototxt file: duplicate names found in the top field of layer (name: {layer_proto_param.name}) in prototxt file: {layer_proto_param.top}"

            # translate current layer
            node: XModelNode = None
            if input_layer_appeared and op_type == "input":
                # create XModelNodeInput object
                node = XModelNodeInput(op_name)
                logger.debug(f"create XModelNodeInput object: name: {node.op_name}")

                # set shape
                shape = None
                if in_shapes is not None and len(in_shapes) > 0:
                    if in_shapes.__class__.__name__ == "list":
                        shape = in_shapes[0]
                    elif in_shapes.__class__.__name__ == "dict":
                        shape = in_shapes.get(op_name)

                if shape is None:
                    if layer_proto_param is not None:
                        shape = [x for x in layer_proto_param.input_param.shape[0].dim]
                        shape[0] = batchsize

                    elif len(proto_param.input_shape) > 0:
                        shape = proto_param.input_shape[0].dim
                        shape[0] = batchsize

                assert all(
                    [x > 0 for x in shape]
                ), f"[ERROR] Invalid shape of input layer: shape: {shape[1:]} (H,W,C), name: {node.op_name}"

                node.shape = shape
                logger.debug(f"property: shape (N,C,H,W): {node.shape}")

                # layout
                node.init_layout = node.layout = xmodel.layout
                logger.debug(
                    f"property: init layout: {node.init_layout}, current layout: {node.layout}"
                )

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "convolution":
                conv_param = layer_proto_param.convolution_param
                # the group size for group convolution
                group = conv_param.group
                # get the number of outputs
                # for depthwise_conv2d, num_output = IC * channel_multiplier
                num_output = conv_param.num_output
                # get kernel size: [kernel_h, kernel_w]
                if conv_param.HasField("kernel_h") and conv_param.HasField("kernel_w"):
                    ksize = [conv_param.kernel_h, conv_param.kernel_w]
                elif len(conv_param.kernel_size) == 1 and conv_param.kernel_size[0] > 0:
                    ksize = [conv_param.kernel_size[0]] * 2
                elif len(conv_param.kernel_size) == 2 and all(
                    i > 0 for i in conv_param.kernel_size
                ):
                    ksize = conv_param.kernel_size
                else:
                    raise ValueError("Not found valid kernel size.")
                # get strides: [stride_h, stride_w]
                if conv_param.HasField("stride_h") and conv_param.HasField("stride_w"):
                    strides = [conv_param.stride_h, conv_param.stride_w]
                elif len(conv_param.stride) == 1 and conv_param.stride[0] > 0:
                    strides = [conv_param.stride[0]] * 2
                elif len(conv_param.stride) == 2 and all(
                    i > 0 for i in conv_param.stride
                ):
                    strides = conv_param.stride
                else:
                    strides = [1, 1]
                # get padding: [pad_top, pad_down, pad_left, pad_right]
                if conv_param.HasField("pad_h") and conv_param.HasField("pad_w"):
                    padding = [conv_param.pad_h] * 2 + [conv_param.pad_w] * 2
                elif conv_param.HasField("pad_h"):
                    padding = [conv_param.pad_h] * 2 + [0, 0]
                elif conv_param.HasField("pad_w"):
                    padding = [0, 0] + [conv_param.pad_w] * 2
                elif len(conv_param.pad) == 1 and conv_param.pad[0] > 0:
                    padding = [conv_param.pad[0]] * 4
                elif len(conv_param.pad) == 2 and all(i > 0 for i in conv_param.pad):
                    padding = [conv_param.pad[0]] * 2 + [conv_param.pad[1]] * 2
                else:
                    padding = [0, 0, 0, 0]
                # get dilation: [dilation_N, dilation_C, dilation_H, dilation_W]
                if len(conv_param.dilation) == 1 and conv_param.dilation[0] > 0:
                    dilation = [1, 1] + [conv_param.dilation[0]] * 2
                elif len(conv_param.dilation) == 2 and all(
                    i > 0 for i in conv_param.dilation
                ):
                    dilation = [1, 1] + conv_param.dilation
                else:
                    dilation = [1, 1, 1, 1]
                # get bias term
                bias_term = conv_param.bias_term
                # get weights' shape and data
                blob_w = layer_model_param.blobs[0]
                # (OC, IC, H, W) for conv2d
                # (CM, IC, H, W) for depthwise_conv2d, where CM means channel_multiplier
                weights = XTensor(
                    np.array(blob_w.data, dtype=np.float32).reshape(blob_w.shape.dim)
                )
                # (oc/cm, ic, h, w) => (h, w, ic, oc/cm)
                weights = tc.transpose(weights, (2, 3, 1, 0))
                _, _, in_ch, _ = weights.shape

                # get bias's shape and data
                if bias_term:
                    blob_b = layer_model_param.blobs[1]
                    bias = XTensor(
                        np.array(blob_b.data, dtype=np.float32).reshape(
                            blob_b.shape.dim
                        )
                    )

                # create XModelNode object according to group value
                node = None
                if group == 1:
                    node = XModelNodeConv2d(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeConv2d object: name: {node.op_name}"
                    )
                elif group == num_output:
                    node = XModelNodeConv2dDepthwise(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeConv2dDepthwise object: name: {node.op_name}"
                    )
                else:
                    error = f"Unsupported group convolution: group={conv_param.group}"
                    logger.error(error)
                    raise ValueError(error)

                # set fields
                node.group = group
                logger.debug(f"property: group: {node.group}")
                node.num_output = num_output
                logger.debug(f"property: num_output: {node.num_output}")
                node.strides = strides
                logger.debug(f"property: strides (h, w): {node.strides}")
                node.padding = padding
                logger.debug(
                    f"property: padding (top, bottom, left, right): {node.padding}"
                )
                node.pad_mode = PadMode.EXPLICIT
                logger.debug(f"property: pad_mode: {node.pad_mode}")
                node.dilation = dilation
                logger.debug(f"property: dilation (h, w): {node.dilation}")
                node.weights = weights
                logger.debug(
                    f"property: weights: shape (out_channels, in_channels, height, width): {node.weights.shape}, dtype: {node.weights.dtype}"
                )
                if bias_term:
                    node.bias_term = bias_term
                    logger.debug(f"property: bias_term: {node.bias_term}")
                    node.bias = bias
                    logger.debug(f"property: bias: shape: {node.bias.shape}")
                node.round_mode = RoundMode.FLOOR
                logger.debug(f"property: round_mode: {node.round_mode}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(
                        node,
                        layer_model_param.fixed_param.fix_info,
                        True,
                        node.bias_term,
                    )

            elif op_type == "relu":
                # create XModelNodeRelu object
                node = XModelNodeRelu(op_name)
                logger.debug(f"create XModelNodeRelu object: name: {node.op_name}")

                # layout
                node.init_layout = node.layout = xmodel.layout
                logger.debug(
                    f"property: init layout: {node.init_layout}, current layout: {node.layout}"
                )

                relu_param = layer_proto_param.relu_param

                # set negative_slope
                if hasattr(relu_param, "negative_slope"):
                    assert relu_param.negative_slope >= 0
                    node.alpha = relu_param.negative_slope

                logger.debug(f"property: alpha: {node.alpha}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "pooling":
                pooling_param = layer_proto_param.pooling_param

                # According to caffe.proto,
                # MAX: 0
                # AVE: 1
                # STOCHASTIC: 2 (not supported)
                pool_method = pooling_param.pool
                assert pool_method in [0, 1, 2], "Unsupported pool method: {}".format(
                    pool_method
                )

                # kernel size: [kernel_h, kernel_w]
                if (
                    pooling_param.HasField("global_pooling")
                    and pooling_param.global_pooling
                ):
                    ksize = [0, 0]
                else:
                    if pooling_param.HasField("kernel_size") > 0:
                        ksize = [pooling_param.kernel_size] * 2
                    elif pooling_param.HasField("kernel_h") and pooling_param.HasField(
                        "kernel_w"
                    ):
                        ksize = [pooling_param.kernel_h, pooling_param.kernel_w]
                    else:
                        raise ValueError(
                            f"Not found valid kernel size: op name: {op_name}"
                        )
                    # TODO: to be removed
                    # assert ksize[0] == ksize[1] and ksize != [
                    #     0,
                    #     0,
                    # ], f"Invalid kernel size : {ksize}, op name: {op_name}"

                # create XModelNode object
                node = None
                if pool_method == 0:
                    node = XModelNodeMaxPool(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeMaxPool object: name: {node.op_name}"
                    )
                elif pool_method == 1:
                    node = XModelNodeAvgPool(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeAvgPool object: name: {node.op_name}"
                    )
                    # count_include_pad
                    node.count_include_pad = True
                    logger.debug(
                        f"property: count_inclue_pad: {node.count_include_pad}"
                    )
                else:
                    raise NotImplementedError(
                        "STOCHASTIC ReLU is not supported in current version."
                    )
                logger.debug(f"property: kernel_size (h, w): {node.kernel_size}")

                # global pooling
                node.is_global = False
                if pooling_param.HasField("global_pooling"):
                    node.is_global = pooling_param.global_pooling
                    logger.debug(f"property: is_global: {node.is_global}")

                # set round_mode
                if hasattr(pooling_param, "round_mode"):
                    node.round_mode = (
                        RoundMode.CEIL if pooling_param.round_mode else RoundMode.FLOOR
                    )
                elif hasattr(pooling_param, "ceil_mode"):
                    node.round_mode = (
                        RoundMode.CEIL if pooling_param.ceil_mode else RoundMode.FLOOR
                    )
                else:
                    node.round_mode = RoundMode.CEIL
                logger.debug(f"property: round_mode: {node.round_mode}")

                if node.is_global:
                    # logger.info("check if kernel_size, strides and paddings follow global pooling rules")
                    assert node.kernel_size == [
                        0,
                        0,
                    ], "'kernel_size' property of a global pooling must be [0, 0]."
                    assert node.strides == [
                        1,
                        1,
                    ], "'strides' property of a global pooling must be [1, 1]."
                    assert node.padding == [
                        0,
                        0,
                        0,
                        0,
                    ], "'padding' property of a global pooling must be [0, 0, 0, 0]."
                else:
                    # set strides
                    if pooling_param.HasField("stride"):
                        node.strides = [pooling_param.stride] * 2
                    elif pooling_param.HasField("stride_h") and pooling_param.HasField(
                        "stride_w"
                    ):
                        assert pooling_param.stride_h > 0 and pooling_param.stride_w > 0
                        node.strides = [pooling_param.stride_h, pooling_param.stride_w]
                    else:
                        raise ValueError("Not found valid stride info")

                    # set padding: [pad_top, pad_bottom, pad_left, pad_right]
                    if pooling_param.HasField("pad"):
                        node.padding = [pooling_param.pad] * 4
                    elif pooling_param.HasField("pad_h") and pooling_param.HasField(
                        "pad_w"
                    ):
                        node.padding = [pooling_param.pad_h] * 2 + [
                            pooling_param.pad_w
                        ] * 2
                    elif pooling_param.HasField("pad_h"):
                        node.padding = [pooling_param.pad_h] * 2 + [0, 0]
                    elif pooling_param.HasField("pad_w"):
                        node.padding = [0, 0] + [pooling_param.pad_w] * 2
                    else:
                        # default value for padding
                        node.padding = [0, 0, 0, 0]
                logger.debug(f"property: strides (h, w): {node.strides}")
                logger.debug(
                    f"property: padding (top, bottom, left, right): {node.padding}"
                )
                node.pad_mode = PadMode.EXPLICIT
                logger.debug(f"property: pad_mode: {node.pad_mode}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "eltwise":
                elem_param = layer_proto_param.eltwise_param

                # According to EltwiseParameter in caffe.proto,
                # 0: PROD
                # 1: SUM
                # 2: MAX
                if elem_param.operation == 1:
                    # create XModelNodeElemAdd object
                    node = XModelNodeElemAdd(op_name)
                    logger.debug(
                        f"create XModelNodeElemAdd object: name: {node.op_name}"
                    )
                else:
                    error = f"Unsupported operation kind: {elem_param.operation}"
                    logger.error(error)
                    raise ValueError(error)

                if len(elem_param.coeff) > 0:
                    node.alpha = elem_param.coeff
                else:
                    node.alpha = [1] * len(node.bottom)
                logger.debug(f"property: alpha: {node.alpha}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "innerproduct":
                # create XModelNodeDot object
                node = XModelNodeDot(op_name)
                logger.debug(f"create XModelNodeDot object: name: {node.op_name}")

                ip_param = layer_proto_param.inner_product_param
                node.bias_term = ip_param.bias_term
                # num_output
                node.num_output = ip_param.num_output
                # transpose
                node.transpose = ip_param.transpose
                # axis
                assert (
                    ip_param.axis == 1
                ), f"[ERROR] xnnc requires the axis of caffe innerproduct is 1: actual {ip_param.axis}."
                node.axis = ip_param.axis

                # get weights' shape and data
                blob_w = layer_model_param.blobs[0]
                shape = list(blob_w.shape.dim)
                weights = XTensor(
                    np.array(blob_w.data, dtype=np.float32).reshape(shape)
                )

                node.weights = weights
                logger.debug(
                    f"property: weights: shape: {node.weights.shape}, dtype: {node.weights.dtype}"
                )

                # get bias's shape and data
                logger.debug(f"property: bias_term: {node.bias_term}")
                if node.bias_term:
                    blob_b = layer_model_param.blobs[1]
                    node.bias = XTensor(
                        np.array(blob_b.data, dtype=np.float32).reshape(
                            blob_b.shape.dim
                        )
                    )
                    logger.debug(
                        f"property: bias: shape: {node.bias.shape}, dtype: {node.bias.dtype}"
                    )

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(
                        node,
                        layer_model_param.fixed_param.fix_info,
                        True,
                        node.bias_term,
                    )

            elif op_type == "softmax":
                # create XModelNodeSoftmax object
                node = XModelNodeSoftmax(op_name)
                logger.debug(f"create XModelNodeSoftmax object: name: {node.op_name}")

                softmax_param = layer_proto_param.softmax_param

                # set axis
                node.axis = 1
                if softmax_param.HasField("axis"):
                    node.axis = softmax_param.axis
                logger.debug(f"property: axis: {node.axis}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "concat":
                # create XModelNodeConcat object
                node = XModelNodeConcat(op_name)
                logger.debug(f"create XModelNodeConcat object: name: {node.op_name}")

                concat_param = layer_proto_param.concat_param
                # set axis, default 1: indicate the axis of channel
                node.axis = 1
                if concat_param.HasField("axis"):
                    node.axis = concat_param.axis
                logger.debug(f"property: axis: {node.axis}")
                assert node.axis >= 0, f"[ERROR] Invalid axis of concat: {node.axis}"

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "relu6":
                # create XModelNodeRelu6 object
                node = XModelNodeRelu6(op_name)
                logger.debug(f"create XModelNodeRelu6 object: name: {node.op_name}")

                relu6_param = layer_proto_param.relu_param
                node.alpha = 0.0
                if relu6_param.HasField("negative_slope"):
                    node.alpha = relu6_param.negative_slope
                logger.debug(f"property: alpha: {node.alpha}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "flatten":
                # create XModelNodeFlatten object
                node = XModelNodeFlatten(op_name)
                logger.debug(f"create XModelNodeFlatten object: name: {node.op_name}")

                flatten_param = layer_proto_param.flatten_param
                # start dim (default 1)
                node.start_dim = flatten_param.axis
                assert (
                    node.start_dim is not None
                ), f"[ERROR] the start dim should not be None. op name: {op_name}."
                logger.debug(f"property: start_dim: {node.start_dim}")

                # end dim (default -1)
                node.end_dim = flatten_param.end_axis
                assert (
                    node.end_dim is not None
                ), f"[ERROR] the end dim should not be None. op name: {op_name}."
                logger.debug(f"property: end_dim: {node.end_dim}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "fixedneuron":
                # create XModelNode object
                node = XModelNodeFixNeuron(op_name)
                logger.debug(
                    f"create XModelNode (fixedneuron) object: name: {node.op_name}"
                )

                if layer_proto_param is not None and layer_model_param is not None:
                    node.is_quantized = True
                    # bit width
                    fixed_param = layer_proto_param.fixed_param
                    node.quant_in["bit_width"] = node.quant_out[
                        "bit_width"
                    ] = fixed_param.bit_width
                    # fix info
                    blob = layer_model_param.blobs[0]
                    quantize_pos = int(blob.data[0])
                    node.quant_in["quantize_pos"] = node.quant_out[
                        "quantize_pos"
                    ] = quantize_pos

                    logger.debug(
                        f"property: quant_in: bit_width: {node.quant_in['bit_width']}, quantize_pos: {node.quant_in['quantize_pos']}"
                    )
                    logger.debug(
                        f"property: quant_out: bit_width: {node.quant_out['bit_width']}, quantize_pos: {node.quant_out['quantize_pos']}"
                    )

            elif op_type == "permute":
                # create XModelNodePermute object
                node = XModelNodePermute(op_name)
                logger.debug(f"create XModelNodePermute object: name: {node.op_name}")

                # set order
                node.order = [x for x in layer_model_param.permute_param.order]
                logger.debug(f"property: order: {node.order}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "priorbox":
                prior_box_param = layer_proto_param.prior_box_param
                assert prior_box_param is not None
                min_sizes = prior_box_param.min_size
                max_sizes = prior_box_param.max_size
                # create XModelNodePriorBox object
                node = XModelNodePriorBox(op_name, min_sizes, max_sizes)
                logger.debug(f"create XModelNodePriorBox object: name: {node.op_name}")
                logger.debug(f"property: min_sizes: {node.min_sizes}")
                logger.debug(f"property: max_sizes: {node.max_sizes}")

                # aspect ratio
                node.aspect_ratio = prior_box_param.aspect_ratio
                logger.debug(f"property: aspect_ratio: {node.aspect_ratio}")
                # flip
                node.flip = prior_box_param.flip
                logger.debug(f"property: flip: {node.flip}")
                # clip
                node.clip = prior_box_param.clip
                logger.debug(f"property: clip: {node.clip}")
                # variance
                node.variance = prior_box_param.variance
                logger.debug(f"property: variance: {node.variance}")
                # step
                if prior_box_param.step > 0.0:
                    node.step = [prior_box_param.step]
                elif prior_box_param.step_h > 0.0 and prior_box_param.step_w > 0.0:
                    node.step = [prior_box_param.step_h, prior_box_param.step_w]
                else:
                    error_msg = f"Invalid value: property: step: {prior_box_param.step}, step_h: {prior_box_param.step_h}, step_w: {prior_box_param.step_w}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                logger.debug(f"property: step: {node.step}")
                # offset
                node.offset = prior_box_param.offset
                logger.debug(f"property: offset: {node.offset}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "reshape":
                # create XModelNodeReshape object
                node = XModelNodeReshape(op_name)
                logger.debug(f"create XModelNodeReshape object: name: {node.op_name}")

                reshape_param = None
                if layer_model_param is not None and layer_model_param.HasField(
                    "reshape_param"
                ):
                    reshape_param = layer_model_param.reshape_param
                elif layer_proto_param is not None and layer_proto_param.HasField(
                    "reshape_param"
                ):
                    reshape_param = layer_proto_param.reshape_param
                else:
                    raise ValueError(
                        f"[ERROR] Not found reshape param: op name: {op_name}."
                    )
                assert (
                    reshape_param is not None
                ), f"[ERROR] Not found reshape params: op name: {op_name}."

                # shape
                shape: List[int] = reshape_param.shape.dim
                tensor = XTensor(np.array(shape, dtype=np.int32))
                assert tensor is not None
                # create XModelNodeConst object
                const_xnode = XModelNodeConst(op_name=op_name + "_const")
                const_xnode.tensor = tensor
                # update bottom of node with const_xnode's name
                bottom.append(const_xnode.op_name)
                # update xmodel with const_xnode
                xmodel.add_xnode(const_xnode)

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "deconvolution":
                conv_param = layer_proto_param.convolution_param
                # the group size for group convolution
                group = conv_param.group
                # get the number of outputs
                num_output = conv_param.num_output
                # get kernel size: [kernel_h, kernel_w]
                if conv_param.HasField("kernel_h") and conv_param.HasField("kernel_w"):
                    ksize = [conv_param.kernel_h, conv_param.kernel_w]
                elif len(conv_param.kernel_size) == 1 and conv_param.kernel_size[0] > 0:
                    ksize = [conv_param.kernel_size[0]] * 2
                elif len(conv_param.kernel_size) == 2 and all(
                    i > 0 for i in conv_param.kernel_size
                ):
                    ksize = conv_param.kernel_size
                else:
                    raise ValueError("Not found valid kernel size.")
                # get strides: [stride_h, stride_w]
                if conv_param.HasField("stride_h") and conv_param.HasField("stride_w"):
                    strides = [conv_param.stride_h, conv_param.stride_w]
                elif len(conv_param.stride) == 1 and conv_param.stride[0] > 0:
                    strides = [conv_param.stride[0]] * 2
                elif len(conv_param.stride) == 2 and all(
                    i > 0 for i in conv_param.stride
                ):
                    strides = conv_param.stride
                else:
                    strides = [1, 1]
                # get padding: [pad_top, pad_down, pad_left, pad_right]
                if conv_param.HasField("pad_h") and conv_param.HasField("pad_w"):
                    padding = [conv_param.pad_h] * 2 + [conv_param.pad_w] * 2
                elif conv_param.HasField("pad_h"):
                    padding = [conv_param.pad_h] * 2 + [0, 0]
                elif conv_param.HasField("pad_w"):
                    padding = [0, 0] + [conv_param.pad_w] * 2
                elif len(conv_param.pad) == 1 and conv_param.pad[0] > 0:
                    padding = [conv_param.pad[0]] * 4
                elif len(conv_param.pad) == 2 and all(i > 0 for i in conv_param.pad):
                    padding = [conv_param.pad[0]] * 2 + [conv_param.pad[1]] * 2
                else:
                    padding = [0, 0, 0, 0]
                # get dilation: [dilation_N, dilation_C, dilation_H, dilation_W]
                if len(conv_param.dilation) == 1 and conv_param.dilation[0] > 0:
                    dilation = [1, 1] + [conv_param.dilation[0]] * 2
                elif len(conv_param.dilation) == 2 and all(
                    i > 0 for i in conv_param.dilation
                ):
                    dilation = [1, 1] + conv_param.dilation
                else:
                    dilation = [1, 1, 1, 1]
                # get bias term
                bias_term = conv_param.bias_term
                # get weights' shape and data
                blob_w = layer_model_param.blobs[0]
                # (oc, ic, h, w)
                weights = XTensor(
                    np.array(blob_w.data, dtype=np.float32).reshape(blob_w.shape.dim)
                )
                # (oc, ic, h, w) => (ic, oc, h, w)
                weights = tc.transpose(weights, (1, 0, 2, 3))
                # flip elements in (h,w): (ic, oc, h', w')
                weights = tc.flip(weights, axis=(2, 3))
                # (ic, oc, h', w') => (h', w', oc, ic)
                weights = tc.transpose(weights, (2, 3, 1, 0))

                # get bias's shape and data
                if bias_term:
                    blob_b = layer_model_param.blobs[1]
                    bias = XTensor(
                        np.array(blob_b.data, dtype=np.float32).reshape(
                            blob_b.shape.dim
                        )
                    )

                # create XModelNode object
                if num_output == group and group > 1:
                    # create XModelNodeDeconvolutionDepthwise object
                    node = XModelNodeDeconvolutionDepthwise(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeDeconvolutionDepthwise object: name: {node.op_name}"
                    )
                elif group == 1:
                    # create XModelNodeDeconvolution object
                    node = XModelNodeDeconvolution(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeDeconvolution object: name: {node.op_name}"
                    )
                else:
                    raise ValueError(
                        f"[ERROR] Unsupported Caffe group deconvolution: group: expected 1 or {num_output}, actual {group}."
                    )

                # set fields
                node.group = group
                logger.debug(f"property: group: {node.group}")
                node.num_output = num_output
                logger.debug(f"property: num_output: {node.num_output}")
                node.strides = strides
                logger.debug(f"property: strides (h, w): {node.strides}")
                node.padding = padding
                logger.debug(
                    f"property: padding (top, bottom, left, right): {node.padding}"
                )
                node.pad_mode = PadMode.EXPLICIT
                logger.debug(f"property: pad_mode: {node.pad_mode}")
                node.dilation = dilation
                logger.debug(f"property: dilation (h, w): {node.dilation}")
                node.weights = weights
                logger.debug(
                    f"property: weights: shape (height, width, in_channels, out_channels, ): {node.weights.shape}, dtype: {node.weights.dtype}"
                )
                if bias_term:
                    node.bias_term = bias_term
                    logger.debug(f"property: bias_term: {node.bias_term}")
                    node.bias = bias
                    logger.debug(f"property: bias: shape: {node.bias.shape}")
                node.round_mode = RoundMode.FLOOR
                logger.debug(f"property: round_mode: {node.round_mode}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(
                        node,
                        layer_model_param.fixed_param.fix_info,
                        True,
                        node.bias_term,
                    )

            elif op_type == "gstiling":
                # create XModelNodeGSTiling object
                node = XModelNodeGSTiling(op_name)
                logger.debug(f"create XModelNodeGSTiling object: name: {node.op_name}")

                gs_tiling_param = layer_proto_param.gs_tiling_param

                # stride
                node.stride = gs_tiling_param.stride
                # reverse
                node.reverse = gs_tiling_param.reverse

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "reorg":
                reorg_param = layer_proto_param.reorg_param
                # stride: int = reorg_param.stride
                strides: List[int] = [1, 1]
                if isinstance(reorg_param.stride, int):
                    strides = [reorg_param.stride] * 2
                elif isinstance(reorg_param.stride, list):
                    strides = reorg_param.stride
                else:
                    raise TypeError(
                        f"[ERROR] Unsupported type: expected: int or list, actual: {reorg_param.stride.__class__.__name__}"
                    )

                reverse: bool = reorg_param.reverse

                # create XModelNodeReorg object
                node = XModelNodeReorg(op_name, strides)
                logger.debug(f"create XModelNodeReorg object: name: {node.op_name}")
                logger.debug(f"property: stride: {node.stride}")
                # set reverse
                node.reverse = reverse
                logger.debug(f"property: reverse: {node.reverse}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "deephiresize":
                deephi_resize_param = layer_proto_param.deephi_resize_param
                scale: List[float] = [
                    deephi_resize_param.scale_h,
                    deephi_resize_param.scale_w,
                ]

                # create XModelNodeDeephiResize object
                node = XModelNodeDeephiResize(op_name)
                logger.debug(
                    f"create XModelNodeDeephiResize object: name: {node.op_name}"
                )

                # set scale
                node.scale = scale
                logger.debug(f"property: scale: {node.scale}")

                # mode
                node.mode = (
                    "bilinear" if deephi_resize_param.resize_type == 0 else "nearest"
                )
                assert node.mode is not None
                logger.debug(f"property: mode: {node.mode}")

                # * quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    cls.__quantize_xnode(node, layer_model_param.fixed_param.fix_info)

            elif op_type == "scale":
                scale_param = layer_proto_param.scale_param

                # create XModelNodeScale object
                node = XModelNodeScale(op_name)
                logger.debug(f"create XModelNodeScale object: name: {node.op_name}")

                # axis
                node.axis = scale_param.axis
                logger.debug(f"property: axis: {node.axis}")

                assert (
                    len(layer_model_param.fixed_param.fix_info) == 8
                ), f"[ERROR] The size of quantization info should be 8: actual: {layer_model_param.fixed_param.fix_info}"
                fix_info = layer_model_param.fixed_param.fix_info
                node.quant_in["bit_width"] = [fix_info[0]]
                node.quant_in["quantize_pos"] = [fix_info[1]]
                logger.debug(f"property: quant_in: {node.quant_in}")
                node.quant_out["bit_width"] = [fix_info[2]]
                node.quant_out["quantize_pos"] = [fix_info[3]]
                logger.debug(f"property: quant_out: {node.quant_out}")
                # quantization info for scale
                quant_scale = {}
                quant_scale["bit_width"] = fix_info[4]
                quant_scale["quantize_pos"] = fix_info[5]
                quant_scale["round_mode"] = 0  # "STD_ROUND"
                logger.debug(f"property: quant_scale: {quant_scale}")
                # bias
                node.quant_bias["bit_width"] = fix_info[6]
                node.quant_bias["quantize_pos"] = fix_info[7]
                node.quant_bias["round_mode"] = 0  # "STD_ROUND"
                logger.debug(f"property: quant_bias: {node.quant_bias}")

                # scale
                if len(layer_proto_param.bottom) == 1:
                    scale = layer_model_param.blobs[0]
                    # step 1: create an XModelNodeCost object to save scale data
                    const_xnode = XModelNodeConst(op_name + "/scale")
                    const_xnode.tensor = XTensor(
                        np.array(scale.data, dtype=np.float32).reshape(scale.shape.dim)
                    )
                    # update xmodel with quant_xnode
                    xmodel.add_xnode(const_xnode)

                    # step 2: create XModelNodeFixNeuron with quantization info
                    quant_xnode = XModelNodeFixNeuron(op_name + "/fix/scale")
                    quant_xnode.quant_in["bit_width"] = quant_scale["bit_width"]
                    quant_xnode.quant_in["quantize_pos"] = quant_scale["quantize_pos"]
                    quant_xnode.quant_in["round_mode"] = quant_scale["round_mode"]
                    quant_xnode.quant_out["bit_width"] = quant_scale["bit_width"]
                    quant_xnode.quant_out["quantize_pos"] = quant_scale["quantize_pos"]
                    quant_xnode.quant_out["round_mode"] = quant_scale["round_mode"]
                    quant_xnode.is_quantized = True
                    # set input
                    quant_xnode.bottom = [const_xnode.op_name]
                    # update xmodel with quant_xnode
                    xmodel.add_xnode(quant_xnode)

                    # update bottom
                    bottom.append(quant_xnode.op_name)

                # get bias term
                node.bias_term = scale_param.bias_term
                # get bias's shape and data
                if node.bias_term:
                    blob_b = (
                        layer_model_param.blobs[1]
                        if len(layer_proto_param.bottom) == 1
                        else layer_model_param.blobs[0]
                    )
                    node.bias = XTensor(
                        np.array(blob_b.data, dtype=np.float32).reshape(
                            blob_b.shape.dim
                        )
                    )

            elif op_type == "slice":
                slice_param = layer_proto_param.slice_param
                assert slice_param is not None

                # check the size of slice_point
                assert (
                    len(slice_param.slice_point) == len(top) - 1
                ), f"[ERROR] Caffe requires that the number of indices must be equal to the number of top blobs minus one."

                # create XModelNodeSlice object
                node = XModelNodeSlice(op_name)
                logger.debug(f"create XModelNodeSlice object: name: {node.op_name}")

                # axis
                node.axis = slice_param.axis
                logger.debug(f"property: axis: {node.axis}")

                # slice_point
                node.top = top
                node.slice_points = [x for x in slice_param.slice_point]
                logger.debug(f"property: slice_points: {node.slice_points}")

                # quantization info
                if layer_model_param is not None and layer_model_param.HasField(
                    "fixed_param"
                ):
                    fix_info = layer_model_param.fixed_param.fix_info

                    # quant_in
                    quant_in = fix_info[:2]
                    node.quant_in["bit_width"] = [quant_in[0]]
                    node.quant_in["quantize_pos"] = [quant_in[1]]
                    logger.debug(f"property: quant_in: {node.quant_in}")

                    # quant_out
                    quant_out = fix_info[2:]
                    node.quant_out["bit_width"] = quant_out[0::2]
                    node.quant_out["quantize_pos"] = quant_out[1::2]
                    logger.debug(f"property: quant_out: {node.quant_out}")

                    node.is_quantized = True
                    logger.debug(f"property: is_quantized: {node.is_quantized}")

            elif op_type == "silence":
                # create XModelNode object
                node = XModelNode(op_name, "silence")
                logger.debug(f"create XModelNodeIdentity object: name: {node.op_name}")

            else:
                # TODO: a temporary solution to SoftmaxWithLoss and Accuracy ops, which should not be present.
                if op_type in [
                    "multiboxloss",
                    "softmaxwithloss",
                    "accuracy",
                    "detectionoutput",
                    "detectionoutputrefine",
                    "imagedata",
                ]:
                    continue
                error = f"Unsupported op: type: {op_type}, name: {op_name}"
                logger.error(error)
                raise ValueError(error)

            # set layout
            node.init_layout = node.layout = xmodel.layout
            logger.debug(
                f"property: init layout: {node.init_layout}, current layout: {node.layout}"
            )
            # set bottom and top
            node.bottom = bottom
            node.top = top

            # update xmodel
            xmodel.add_xnode(node)
            # set xmodel as quantized
            if not xmodel.is_quantized and node.is_quantized:
                xmodel.is_quantized = True
        logger.info("* end: translate caffe layers to xmodel nodes")

        # create a dict for node lookup
        # key: node name
        # value: node
        node_dict = {}
        for node in xmodel.xnodes:
            assert (
                node.op_name not in node_dict
            ), f"Duplicate xnode: name: {node.op_name}"
            node_dict[node.op_name] = node
        assert len(node_dict) == len(
            xmodel.xnodes
        ), f"size of node_dict={len(node_dict)}, size of xnodes={len(xmodel.xnodes)}"

        blob_as_in = {}
        for xnode in xmodel.xnodes:
            if xnode.bottom is not None and len(xnode.bottom) > 0:
                for iname in xnode.bottom:
                    if iname not in blob_as_in:
                        blob_as_in[iname] = [xnode]
                    else:
                        blob_as_in[iname].append(xnode)

        slice_children = {}
        for xnode in xmodel.xnodes:
            if xnode.op_type == "slice":
                for i in range(len(xnode.top)):
                    blob_name = xnode.top[i]
                    assert blob_name in blob_as_in
                    cnodes = blob_as_in.get(blob_name)
                    for cnode in cnodes:
                        if cnode.op_type != "fixneuron":
                            if xnode.op_name not in slice_children:
                                slice_children[xnode.op_name] = [cnode]
                            else:
                                slice_children[xnode.op_name].append(cnode)

        # create master-slave dict.
        # * master node is a node that has one or more in-place nodes.
        # * slave node is a node, of which the top and bottom properties contain exactly same values.
        # * In Caffe fixed model, slave node could be one of ReLU, Dropout, BatchNorm, Scale and FixedNeuron.
        # key: master's name, value: list of slave nodes
        master_slave_dict: List[str, List[XModelNode]] = {}
        for node in xmodel.xnodes:
            if len(node.top) > 0 and node.top == node.bottom:
                node.is_inplace = True
                # connect a slave with its master
                master_name = node.bottom[0]
                if master_name in master_slave_dict:
                    master_slave_dict[master_name].append(node)
                else:
                    master_slave_dict[master_name] = [node]

        # * locate all nodes, of which op_name is different from name in top field
        # key: name in top field of node, value: op_name of node
        bad_top_nodes = {}
        slice_appears = 0
        for node in xmodel.xnodes:
            if len(node.top) > 0 and not node.is_inplace:
                if len(node.top) == 1:
                    bad_top_nodes[node.top[0]] = node.op_name
                    node.top = []
                elif len(node.top) > 1 and node.op_type == "slice":
                    for oname in node.top:
                        bad_top_nodes[oname] = node.op_name
                    node.top = []
                    slice_appears += 1
                else:
                    raise ValueError(
                        f"[ERROR] The top property has two or more names in current node: type: {node.op_type}, name: {node.op_name}"
                    )
        # fix bad topo
        for node in xmodel.xnodes:
            if len(node.bottom) > 0:
                for i in range(len(node.bottom)):
                    iname = node.bottom[i]
                    if iname in bad_top_nodes:
                        if node.is_inplace:
                            node.bottom[i] = node.top[i] = bad_top_nodes[iname]
                        else:
                            node.bottom[i] = bad_top_nodes[iname]
        # update master name if it is bad
        tmp_dict = {}
        for key, value in master_slave_dict.items():
            if key in bad_top_nodes:
                master_name = bad_top_nodes[key]
                assert isinstance(value, list)
                if master_name in tmp_dict:
                    tmp_dict[master_name] += value
                else:
                    tmp_dict[master_name] = value
            else:
                assert key not in tmp_dict
                tmp_dict[key] = value
        assert len(tmp_dict) + slice_appears == len(master_slave_dict)
        master_slave_dict = tmp_dict

        # * update top field of parent node by current node's bottom field
        for node in xmodel.xnodes:
            # ignore slave node
            if node.top == node.bottom:
                continue

            if node.op_type == "slice":
                cnodes = slice_children.get(node.op_name)
                node.top = [x.op_name for x in cnodes]

            if node.bottom is not None and len(node.bottom) > 0:
                pnames = node.bottom
                for pname in pnames:
                    # get parent node by name
                    pnode = node_dict.get(pname)
                    assert (
                        pnode is not None
                    ), f"Not found parent node: node name: {node.op_name}, node type: {node.op_type}, parent name: {pname}"

                    if pnode.op_type == "slice":
                        continue

                    # append node.op_name to its parent node's top field
                    if node.op_name not in pnode.top:
                        pnode.top.append(node.op_name)

        # * release and merge slaves
        for master_name, slaves in master_slave_dict.items():
            # master node
            master = node_dict.get(master_name)
            assert master is not None

            if master.op_type == "slice":
                for i in range(len(slaves)):
                    cname = master.top[i]
                    slaves[i].bottom = [master_name]
                    slaves[i].top = [cname]
                    master.top[i] = slaves[i].op_name
                    # update the bottom of child node
                    cnode = xmodel.get_xnode_by_name(cname)
                    assert (
                        cnode is not None
                    ), f"[ERROR] Not found child node: name: {cname}."
                    idx = cnode.bottom.index(master_name)
                    cnode.bottom[idx] = slaves[i].op_name
            else:
                # step 1: slaves connect each other in the order
                if len(slaves) > 1:
                    for i in range(len(slaves) - 1):
                        slaves[i].top = [slaves[i + 1].op_name]
                        slaves[i + 1].bottom = [slaves[i].op_name]
                # step 2: connect child nodes of master with last slave in its slave list
                slaves[-1].top = master.top
                if len(master.top) > 0:
                    for child_name in master.top:
                        # child node
                        child = node_dict.get(child_name)
                        assert child is not None
                        # remove master's name and append last slave's name
                        idx = child.bottom.index(master.op_name)
                        child.bottom[idx] = slaves[-1].op_name
                # step 3: connect master with first slave in its slave list
                master.top = [slaves[0].op_name]
                slaves[0].bottom = [master.op_name]

        # * check quantization info of input xnode
        if xmodel.is_quantized:
            data_xnode = xmodel.xnodes[0]
            if not data_xnode.is_quantized:
                for xnode in xmodel.xnodes[1:]:
                    if data_xnode.op_name in xnode.bottom:
                        if xnode.is_quantized:
                            data_xnode.quant_in["bit_width"] = data_xnode.quant_out[
                                "bit_width"
                            ] = xnode.quant_in.get("bit_width")
                            data_xnode.quant_in["quantize_pos"] = data_xnode.quant_out[
                                "quantize_pos"
                            ] = xnode.quant_in.get("quantize_pos")
                            data_xnode.is_quantized = True
                            break

        # * create XModelNodeFixNeuron objects if the raw model is quantized and no fix neuron layer appears
        if not fix_neuron_appeared:
            xnodes_list: List[XModelNode] = []
            for node in xmodel.xnodes:
                xnodes_list.append(node)
                if node.is_quantized:
                    # ignore the case of Conv2d followed by Relu
                    if node.op_type in ["conv2d", "depthwise_conv2d", "deconvolution"]:
                        if len(node.top) == 1:
                            child = xmodel.get_xnode_by_name(node.top[0])
                            if child is not None and child.op_type == "relu":
                                continue
                    # create XModelNodeFixNeuron instance
                    fix_neuron = XModelNodeFixNeuron(node.op_name + "_fix_neuron")
                    logger.debug(
                        f"create XModelNodeFixNeuron instance: name: {fix_neuron.op_name}"
                    )
                    xnodes_list.append(fix_neuron)
                    # set quantization info
                    fix_neuron.quant_in["bit_width"] = fix_neuron.quant_out[
                        "bit_width"
                    ] = node.quant_out["bit_width"]
                    fix_neuron.quant_in["quantize_pos"] = fix_neuron.quant_out[
                        "quantize_pos"
                    ] = node.quant_out["quantize_pos"]
                    logger.debug(
                        f"property: quant_in: {fix_neuron.quant_in}, quant_out: {fix_neuron.quant_out}"
                    )

                    # * update top of node, bottom of node's child nodes, and bottom and top of fix_neuron
                    # step 1: update bottom of fix_neuron with node's name
                    fix_neuron.bottom = [node.op_name]
                    if node.top is not None and len(node.top) > 0:
                        # step 2: iterate child nodes: update bottom of each child with fix_neuron's name, and
                        # update top of fix_neuron with the child nodes' names
                        for cname in node.top:
                            cnode = node_dict.get(cname)
                            assert (
                                cnode is not None
                            ), f"[ERROR] Not found node: op_name: {cname}"
                            # replace node's name with fix_neuron's name (keep the original order)
                            idx = cnode.bottom.index(node.op_name)
                            cnode.bottom[idx] = fix_neuron.op_name
                            # update top of fix_neuron with the child's name
                            if fix_neuron.top is None:
                                fix_neuron.top = [cname]
                            else:
                                fix_neuron.top.append(cname)
                    # step 3: udpate top of node
                    node.top = [fix_neuron.op_name]
            # update nodes of xmodel
            xmodel.update_xnodes(xnodes_list)

        # topsort xnodes
        xmodel.topsort()

        # ! this pass is a temp solution to deal with fixneuron nodes
        cls.__fill_in_fixneuron_pass(xmodel)

        # * Special Pass: set round_mode property of xmodel
        cls.__set_fixneuron_round_mode(xmodel)

        # remove flatten nodes, which are directly followed by dot nodes (caffe innerproduct)
        for xnode in xmodel.xnodes:
            if xnode.op_type == "flatten":
                if xnode.top is not None and len(xnode.top) > 0:
                    cname_list = []
                    cnames = [x for x in xnode.top]
                    for cname in cnames:
                        cnode = xmodel.get_xnode_by_name(cname)
                        assert cnode is not None
                        if cnode.op_type == "dot":
                            cnode.bottom = xnode.bottom
                            xnode.top.remove(cname)
                            cname_list.append(cname)
                    pnode = xmodel.get_xnode_by_name(xnode.bottom[0])
                    if len(cname_list) == len(cnames):
                        pnode.top.remove(xnode.op_name)
                        xnode.bottom = xnode.top = []
                        xmodel.remove_xnode(xnode)
                    pnode.top += cname_list

        if not xmodel.infer_shape(Layout.NHWC):
            print(f"[ERROR] Failed to infer xmodel with the {xmodel.layout} layout")
            sys.exit(1)

        # * perform platform-specific optimizations
        OptManager.dispatch(xmodel, "xnnc")

        return xmodel

    @classmethod
    def __fill_in_fixneuron_pass(cls, xmodel: XModel) -> NoReturn:
        """fill in fixneuron node with its parents and children's quant info

        Parameters
        ----------
        xmodel : XModel
            an XModel instance
        """
        for xnode in xmodel.xnodes:
            if xnode.op_type == "fixneuron" and not xnode.is_quantized:
                # update quant_in of current xnode
                if xnode.bottom is not None:
                    assert (
                        len(xnode.bottom) == 1
                    ), f"[ERROR] FixNeuron only supports single-in-single-out: {len(xnode.bottom)} inputs, name: {xnode.op_name}"
                    # get parent node
                    pnode = xmodel.get_xnode_by_name(xnode.bottom[0])
                    idx = pnode.top.index(xnode.op_name)
                    # update quant info of current xnode with the info of its parent node
                    xnode.quant_in["bit_width"] = pnode.quant_out["bit_width"][idx]
                    xnode.quant_in["quantize_pos"] = pnode.quant_out["quantize_pos"][
                        idx
                    ]

                # update quant_out of current xnode
                if xnode.top is not None and len(xnode.top) > 0:
                    bit_width_list = []
                    quantize_pos_list = []
                    num_priorbox = 0
                    for cname in xnode.top:
                        cnode = xmodel.get_xnode_by_name(cname)
                        assert (
                            cnode is not None
                        ), f"[ERROR] Not found node: name:{cname}."
                        if cnode.op_type == "priorbox":
                            num_priorbox += 1
                            continue
                        if len(cnode.bottom) > 1:
                            if (
                                len(cnode.quant_in["bit_width"])
                                == len(cnode.quant_in["quantize_pos"])
                                == 1
                            ):
                                bit_width_list.append(cnode.quant_in["bit_width"][0])
                                quantize_pos_list.append(
                                    cnode.quant_in["quantize_pos"][0]
                                )
                            elif (
                                len(cnode.quant_in["bit_width"])
                                == len(cnode.quant_in["quantize_pos"])
                                == len(cnode.bottom)
                            ):
                                idx = cnode.bottom.index(xnode.op_name)
                                bit_width_list.append(cnode.quant_in["bit_width"][idx])
                                quantize_pos_list.append(
                                    cnode.quant_in["quantize_pos"][idx]
                                )
                            else:
                                raise ValueError(
                                    f"[ERROR] Mismatched inputs and quantization info: number of inputs: {len(cnode.bottom)}, number of quantization info: {len(cnode.quant_in['bit_width'])}"
                                )
                        else:
                            bit_width_list.append(cnode.quant_in["bit_width"][0])
                            quantize_pos_list.append(cnode.quant_in["quantize_pos"][0])

                    assert len(bit_width_list) + num_priorbox == len(
                        xnode.top
                    ), f"size: top: {len(xnode.top)}, bit: {len(bit_width_list)}"
                    assert len(quantize_pos_list) + num_priorbox == len(
                        xnode.top
                    ), f"size: top: {len(xnode.top)}, bit: {len(quantize_pos_list)}"
                    if len(bit_width_list) > 1:
                        assert all(
                            [bit_width_list[0] == x for x in bit_width_list[1:]]
                        ), f"{bit_width_list}, op name: {xnode.op_name}"
                        assert all(
                            [quantize_pos_list[0] == x for x in quantize_pos_list[1:]]
                        ), f"{quantize_pos_list}, op_name: {xnode.op_name}, op_type: {xnode.op_type}"

                    xnode.quant_out["bit_width"] = bit_width_list[0]
                    xnode.quant_out["quantize_pos"] = quantize_pos_list[0]

                # verify the bit_width and quantize_pos of quant_in and quant_out
                if xnode.quant_in is not None and xnode.quant_out is not None:
                    if (
                        xnode.quant_in["bit_width"] is not None
                        and xnode.quant_out["bit_width"] is not None
                    ):
                        assert (
                            xnode.quant_in["bit_width"] == xnode.quant_out["bit_width"]
                        ), f"[ERROR] Invalid bit_width: in: {xnode.quant_in['bit_width']}, out: {xnode.quant_out['bit_width']}. Node name: {xnode.op_name}, type: {xnode.op_type}."

                    if (
                        xnode.quant_in["quantize_pos"] is not None
                        and xnode.quant_out["quantize_pos"] is not None
                    ):
                        assert (
                            xnode.quant_in["quantize_pos"]
                            == xnode.quant_out["quantize_pos"]
                        ), f"[ERROR] Invalid quantize_pos: in: {xnode.quant_in['quantize_pos']}, out: {xnode.quant_out['quantize_pos']}. Node name: {xnode.op_name}, type: {xnode.op_type}."

                xnode.is_quantized = True

    @classmethod
    def __quantize_xnode(
        cls,
        node: XModelNode,
        fix_info: List[int],
        weights_term: bool = False,
        bias_term: bool = False,
    ) -> NoReturn:
        """Parse quantization info for XModelNode instance.

        Parameters
        ----------
        node : XModelNode
            XModelNode instance.
        fix_info : List[int]
            List of integers, which are pairs of bitwidth and quantized position.
        weights_term : bool, optional
            If True, the quantization info for weights is present; otherwise False., by default False
        bias_term : bool, optional
            If True, the quantization info for bias is present; otherwise False., by default False

        Returns
        -------
        NoReturn
            No return.
        """
        assert node is not None, "'node' should not be None."
        assert fix_info is not None, "'fix_info' should not be None."
        assert len(fix_info) % 2 == 0

        node.is_quantized = True
        if weights_term and bias_term:
            assert len(fix_info) == 8
            # bias
            node.quant_bias["bit_width"] = fix_info[6]
            node.quant_bias["quantize_pos"] = fix_info[7]
            node.quant_bias["round_mode"] = 0  # "STD_ROUND"
            logger.debug(f"property: quant_bias: {node.quant_bias}")
            # weights
            node.quant_weights["bit_width"] = fix_info[4]
            node.quant_weights["quantize_pos"] = fix_info[5]
            node.quant_weights["round_mode"] = 0  # "STD_ROUND"
            logger.debug(f"property: quant_weights: {node.quant_weights}")

            node.quant_in["bit_width"] = [fix_info[0]]
            node.quant_in["quantize_pos"] = [fix_info[1]]
            logger.debug(f"property: quant_in: {node.quant_in}")
            node.quant_out["bit_width"] = [fix_info[2]]
            node.quant_out["quantize_pos"] = [fix_info[3]]
            logger.debug(f"property: quant_out: {node.quant_out}")
        elif weights_term:
            assert (
                len(fix_info) >= 6
            )  # ! deconvolution: when there is no bias, fix_info is still of length 6.
            # weights
            node.quant_weights["bit_width"] = fix_info[4]
            node.quant_weights["quantize_pos"] = fix_info[5]
            node.quant_weights["round_mode"] = 0  # "STD_ROUND"
            logger.debug(f"property: quant_weights: {node.quant_weights}")
            node.quant_in["bit_width"] = [fix_info[0]]
            node.quant_in["quantize_pos"] = [fix_info[1]]
            logger.debug(f"property: quant_in: {node.quant_in}")
            node.quant_out["bit_width"] = [fix_info[2]]
            node.quant_out["quantize_pos"] = [fix_info[3]]
            logger.debug(f"property: quant_out: {node.quant_out}")
        elif len(fix_info) >= 4:
            quant_in = fix_info[: len(fix_info) - 2]
            node.quant_in["bit_width"] = quant_in[0::2]
            node.quant_in["quantize_pos"] = quant_in[1::2]
            logger.debug(f"property: quant_in: {node.quant_in}")
            quant_out = fix_info[-2:]
            node.quant_out["bit_width"] = [quant_out[0]]
            node.quant_out["quantize_pos"] = [quant_out[1]]
            logger.debug(f"property: quant_out: {node.quant_out}")
        elif len(fix_info) == 2:
            node.quant_in["bit_width"] = [fix_info[0]]
            node.quant_in["quantize_pos"] = [fix_info[1]]
            logger.debug(f"property: quant_in: {node.quant_in}")
            node.quant_out["bit_width"] = [fix_info[0]]
            node.quant_out["quantize_pos"] = [fix_info[1]]
            logger.debug(f"property: quant_out: {node.quant_out}")
        else:
            raise ValueError(f"[ERROR] Unsupported fix info: {fix_info}")

    @classmethod
    def __set_fixneuron_round_mode(cls, xmodel: XModel) -> None:
        """Set the round_mode property of FixNeuron node.

        The round_mode property of a FixNeuron node is "STD_ROUND", if its parent node is Const node; otherwise, "DPU_ROUND".

        Parameters
        ----------
        xmodel : XModel
            XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."

        for xnode in xmodel.xnodes:
            if xnode.op_type == "fixneuron":
                if xnode.bottom is not None and len(xnode.bottom) > 0:
                    assert len(xnode.bottom) == 1
                    pnode = xmodel.get_xnode_by_name(xnode.bottom[0])
                    assert (
                        pnode is not None
                    ), f"[ERROR] Not found parent node: {xnode.bottom[0]}."
                    if pnode.op_type == "const":
                        xnode.quant_in["round_mode"] = 0  # "STD_ROUND"
                        xnode.quant_out["round_mode"] = 0
                    else:
                        xnode.quant_in["round_mode"] = 1  # "DPU_ROUND"
                        xnode.quant_out["round_mode"] = 1
                    # update the round_mode of parent node
                    pnode.quant_out["round_mode"] = xnode.quant_in["round_mode"]

                if xnode.top is not None and len(xnode.top) > 0:
                    for cname in xnode.top:
                        cnode = xmodel.get_xnode_by_name(cname)
                        quantize_pos_in = cnode.quant_in.get("quantize_pos")
                        if quantize_pos_in is not None:
                            cnode.quant_in["round_mode"] = xnode.quant_out["round_mode"]
