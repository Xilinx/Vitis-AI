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
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple
import copy

from numpy.core.arrayprint import dtype_short_repr
from xnnc.ir.xnode import RoundMode

import numpy as np
from tqdm import tqdm
from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import *
from xnnc.optimizer import OptManager
from xnnc.tensor.xtensor import XTensor
from xnnc.tensor.xtensor import XTensorCompute as tc
from xnnc.translator.base_translator import ITranslator
from xnnc.utils import helper

# from xnnc.utils.helper import Layout

try:
    import h5py
except ImportError:
    h5py = None


DT_NAME = {
    1: "DT_FLOAT",
    2: "DT_DOUBLE",
    3: "DT_INT32",
    4: "DT_UINT8",
    5: "DT_INT16",
    6: "DT_INT8",
    9: "DT_INT64",
    14: "DT_BFLOAT16",
    19: "DT_HALF",
}

TF_TO_NP = {
    "DT_INT8": np.int8,
    "DT_INT16": np.int16,
    "DT_INT32": np.int32,
    "DT_INT64": np.int64,
    "DT_UINT8": np.uint8,
    "DT_HALF": np.float16,
    "DT_FLOAT": np.float32,
    "DT_BFLOAT16": np.float16, # Hard Code : numpy has no data type 'bfloat16'
    "DT_DOUBLE": np.float64,
}


# create logger
logger = logging.getLogger(__name__)


class TFTranslator(ITranslator):
    """
    Convert TensorFlow model architecture into XModel one
    """

    @classmethod
    def to_xmodel(
        cls,
        model_files: List[Path],
        layout: Layout = Layout.NHWC,
        in_shapes: Optional[Union[List[List[int]], Dict[str, List[int]]]] = None,
        *args,
        **kwargs,
    ) -> XModel:
        # model type
        model_type = "tensorflow"
        if "model_type" in kwargs:
            model_type = kwargs.get("model_type")

        # load model architecture
        logger.info("load raw model architecture")
        model_fmt, (model_name, raw_nodes) = cls.load_raw_model(model_files)
        logger.debug(
            f"raw model name: {model_name}, numb. of raw layers: {len(raw_nodes)}"
        )

        # create an xmodel object
        logger.info("generate xmodel")
        xmodel = cls.create_xmodel(
            model_name,
            raw_nodes,
            layout,
            in_shapes,
            kwargs.get("batchsize"),
            model_fmt,
            model_type,
        )
        logger.debug(f"numb. of xnodes hosted by xmodel: {len(xmodel.xnodes)}")

        return xmodel

    @classmethod
    def load_raw_model(cls, model_files: List[Path]) -> Tuple[str, str, List[Any]]:
        """
        Load raw model files from a list of specified file paths.

        Parameters:
            model_files: a list of specified file paths, which should specify the paths to TensorFlow model files.

        Returns:
            str: model name
            dict: key is layer/node name, value is a layer dict.
        """
        # check model files
        if len(model_files) != 1:
            logger.error(
                "The 'model_files' argument should contain only one file with '.pb' or '.h5' extension."
            )
            sys.exit(1)

        # load model architecture file
        tfmodel: Path = None
        if model_files[0].suffix == ".pb":
            return "pb", cls.__load_from_pb(model_files)
        elif model_files[0].suffix == ".h5":
            return "h5", cls.__load_from_hdf5(model_files)
        else:
            raise ValueError(
                "[ERROR] xnnc tensorflow_translator only supports the file formats: '.pb' or '.h5'"
            )

    @classmethod
    def create_xmodel(
        cls,
        name: str,
        layers: List["node_def_pb2.NodeDef"],
        layout: Layout = Layout.NHWC,
        in_shapes: Optional[Union[List[List[int]], Dict[str, List[int]]]] = None,
        batchsize: int = 1,
        model_format: str = "pb",
        model_type: str = "tensorflow",
    ) -> XModel:
        """
        Create an XModel object from TensorFlow model layers.

        Parameters:
            name: model name
            layers: TensorFlow model layers.
            layout: data layout of original model.
            in_shapes: shape of model inputs

        Returns:
            An XModel object.
        """
        assert name is not None, "The argument 'name' should not be None."
        assert layers is not None, "The argument 'layers' should not be None."
        assert isinstance(layout, Layout)
        assert model_format in ["pb", "h5"]

        if model_type == "tensorflow":
            xmodel = cls.__create_xmodel_from_tf1(
                name, layers, layout, in_shapes, batchsize
            )
        else:
            xmodel = cls.__create_xmodel_from_tf2(
                name, layers, layout, in_shapes, batchsize
            )
        return xmodel

    @classmethod
    def __create_xmodel_from_tf1(
        cls,
        name: str,
        layers: List[Any],
        layout: Layout = Layout.NHWC,
        in_shapes: Optional[Union[List[List[int]], Dict[str, List[int]]]] = None,
        batchsize: int = 1,
    ) -> XModel:
        """
        Create an XModel object from TensorFlow model layers.

        Parameters:
            name: model name
            layers: TensorFlow model layers.
            layout: data layout of original model.
            in_shapes: shape of model inputs

        Returns:
            An XModel object.
        """
        assert name is not None, "The argument 'name' should not be None."
        assert layers is not None, "The argument 'layers' should not be None."

        xmodel_name: str = name

        # * create const_layer_dict to store the Const layers
        # key: name of Const layer, value: Const layer
        const_layer_dict = {}
        # * create identity_layer_dict to store the Identity layers
        # key: name of Identity layer, value: Identity layer
        identity_layer_dict = {}
        # * create bias_add_dict to store the BiasAdd layers
        # key: name of BiasAdd layer, value: BiasAdd layer
        bias_add_dict = {}
        for layer in layers:
            if layer.op.lower() == "const":
                const_layer_dict[layer.name] = layer
            elif layer.op.lower() == "biasadd":
                bias_add_dict[layer.name] = layer
            elif layer.op.lower() == "identity":
                identity_layer_dict[layer.name] = layer

        # filter out both const layers and biasadd layers
        layers = [
            layer
            for layer in layers
            if layer.name not in const_layer_dict and layer.name not in bias_add_dict
        ]

        # * create fixed_neuron_dict to store the FixedNeuron layers which has a Const layer as input
        # key: name of FixedNeuron layer, value: FixedNeuron layer
        fixed_neuron_dict = {}
        for layer in layers:
            if layer.op.lower() == "fixneuron" and layer.input[0] in const_layer_dict:
                fixed_neuron_dict[layer.name] = layer

        logger.debug("merge FixedNeuron and Const into super const layers")
        # * create super_const_dict to store SuperConst which is produced by merging FixedNeuron and Const
        # key: name of FixedNeuron layer, value: super_const which is a dict of properties of interest, including quantize_pos, bit_width, tensor, and etc.
        super_const_dict: Dict[str, Any] = {}
        for name, fixed_neuron_layer in fixed_neuron_dict.items():
            super_const = {}
            # query Const layer as input of current FixedNeuron layer
            const_layer = const_layer_dict.pop(fixed_neuron_layer.input[0], None)
            assert (
                const_layer is not None
            ), f"Not found const layer as input of current fixed neuron layer (name: {name})."

            logger.debug(f"*** merge: fixed_neuron: {name}, const: {const_layer.name}")

            # merge properties of interest in Const and FixedNeuron layers
            super_const["quantize_pos"]: int = fixed_neuron_layer.attr["quantize_pos"].i
            super_const["bit_width"]: int = fixed_neuron_layer.attr["bit_width"].i
            super_const["tensor"]: XTensor = cls.__get_tensor(const_layer.attr["value"])
            # update super_const_dict
            super_const_dict[name] = super_const

        identity_const_layers = []
        if identity_layer_dict is not None and len(identity_layer_dict) > 0:
            for name, id_layer in identity_layer_dict.items():
                # query Const layer as input of current FixedNeuron layer
                const_layer = const_layer_dict.pop(id_layer.input[0], None)

                if const_layer is not None:
                    super_const = {}
                    super_const["tensor"]: XTensor = cls.__get_tensor(
                        const_layer.attr["value"]
                    )
                    super_const_dict[name] = super_const
                    identity_const_layers.append(name)

        # filter out fixed neuron layers
        layers = [
            layer
            for layer in layers
            if layer.name not in fixed_neuron_dict
            and layer.name not in identity_const_layers
        ]

        # * translate each layer into xnode
        xmodel: XModel = cls.__generate_xmodel(
            xmodel_name,
            layout,
            layers,
            const_layer_dict,
            super_const_dict,
            in_shapes,
            batchsize,
        )

        # create dict for querying XModelNode objects
        # key: name of xnode, value: xnode
        xnode_dict: Dict[str, XModelNode] = {}
        for xnode in xmodel.xnodes:
            xnode_dict[xnode.op_name] = xnode

        # * Special Pass: convert SpaceToBatchND + Conv2d (or similar op) with dr=1 + BatchToSpaceND into Conv2d with dr>1
        cls.__recover_dilated_conv_pass(xmodel, xnode_dict)

        # * Special Pass: reduce BiasAdd op and merge into its parent node
        cls.__reduce_biasadd_pass(xmodel, xnode_dict, bias_add_dict, super_const_dict)

        # * update top property of each xnode
        logger.info("* start: update top property of each xnode")
        for xnode in xmodel.xnodes:
            if xnode.bottom is not None and len(xnode.bottom) > 0:
                # iterate each parent node
                for pname in xnode.bottom:
                    try:
                        pnode = xnode_dict.get(pname)
                        if pnode:
                            if pnode.top is None:
                                pnode.top = []
                            if xnode.op_name not in pnode.top:
                                pnode.top.append(xnode.op_name)
                        else:
                            error = f"Not found xnode (name: {pname}), which is input of xnode (name: {xnode.op_name}, type: {xnode.op_type})."
                            logger.error(error)
                            raise KeyError(error)
                    except KeyError:
                        error = f"Not found xnode (name: {pname}), which is input of xnode (name: {xnode.op_name}, type: {xnode.op_type})."
                        logger.error(error)
                        raise KeyError(error)
        logger.info("* end: update top property of each xnode")

        # topsort xnodes
        xmodel.topsort()

        cls.__specialcase_replace_elemadd_with_biasadd(xmodel)

        # * Special Pass: set round_mode property of xmodel
        cls.__set_fixneuron_round_mode(xmodel)

        if not xmodel.infer_shape(Layout.NHWC):
            print(f"[ERROR] Failed to infer xmodel with the {xmodel.layout} layout")
            sys.exit(1)

        # * perform platform-specific optimizations
        OptManager.dispatch(xmodel, "xnnc")

        return xmodel

    @classmethod
    def __create_xmodel_from_tf2(
        cls,
        name: str,
        layers: List[Any],
        layout: Layout = Layout.NHWC,
        in_shapes: Optional[Union[List[List[int]], Dict[str, List[int]]]] = None,
        batchsize: int = 1,
    ) -> XModel:
        logger.info("* start: translate tensorflow layers to xmodel nodes")

        assert "+" in name
        model_name, keras_api = name.split("+")

        # create an xmodel
        xmodel = XModel(model_name, "tensorflow2", layout=str(layout).split(".")[-1])
        logger.info(
            f"create XModel object: name: {xmodel.name}, type: {xmodel.origin}, layout: {xmodel.layout}"
        )

        # translate tensorflow ir into xnnc ir
        wrapper_to_xnode = {}
        # mapping for post activation
        post_act_map: Dict[str, str] = {}
        pbar = tqdm(
            layers,
            desc="[INFO] parse raw model",
            bar_format="{desc:27}:{percentage:3.0f}%|{bar}{r_bar:50}",
        )
        for i, (layer, param) in enumerate(pbar):
            # get name and type of wrapper layer
            wrapper_name: str = (
                layer.get("name")
                if keras_api == "functional"
                else layer.get("config").get("name")
            )
            wrapper_type: str = layer.get("class_name")

            if wrapper_type == "TensorFlowOpLayer":
                op_type = layer.get("config").get("node_def").get("op")
                assert op_type is not None
                op_type = op_type.lower()

                config = layer.get("config")

            else:
                if wrapper_type.startswith("Vitis>"):
                    wrapper_type = wrapper_type[len("Vitis>") :]

                if wrapper_type == "VitisQuantize":
                    wrapper_type = "QuantizeLayer"

                # layer config
                config = layer.get("config")
                assert config is not None

                if wrapper_type == "QuantizeWrapper":
                    inner_layer = config.get("layer")
                    op_type = inner_layer.get("class_name")
                    inner_layer_config = inner_layer.get("config")
                    op_name = inner_layer_config.get("name")

                    if op_type.startswith("Vitis>Vitis"):
                        op_type = op_type[len("Vitis>Vitis") :]
                    elif op_type.startswith("Vitis>"):
                        op_type = op_type[len("Vitis>") :]

                    if op_type == "Activation":
                        activ = inner_layer_config.get("activation")
                        activ_config = activ.get("config")
                        activ_type = activ_config.get("activation")
                        if isinstance(activ_type, str):
                            op_type = activ_type

                elif wrapper_type == "Activation":
                    assert (
                        "activation" in config
                    ), f"[ERROR] Not found activation attribute in 'config' of wrapper layer: layer type: {wrapper_type}, name: {wrapper_name}."
                    op_type = config.get("activation")

                else:
                    op_type = wrapper_type
                    op_name = config.get("name")
                    if wrapper_name is None:
                        wrapper_name = op_name

            logger.debug(
                f"*** source layer info: type: {op_type}, name: {wrapper_name}"
            )

            # translate current layer to XModel node object
            xnode: XModelNode = None
            if op_type == "InputLayer":
                # create XModelNodeInput object
                xnode = XModelNodeInput(wrapper_name)
                logger.debug(f"create XModelNodeInput object: name: {xnode.op_name}")

                # get shape info
                shape = None
                if in_shapes is not None and len(in_shapes) > 0:
                    if in_shapes.__class__.__name__ == "list":
                        shape = in_shapes[0]
                    elif in_shapes.__class__.__name__ == "dict":
                        shape = in_shapes.get(op_name)

                if shape is None:
                    shape = config.get("batch_input_shape")
                    assert (
                        shape is not None
                    ), f"[ERROR] Failed to extract the input shape of the input layer: {xnode.op_name}. Please check the model or provide input shape via command line option."
                    shape[0] = batchsize

                assert all(
                    [isinstance(x, int) and x > 0 for x in shape]
                ), f"[ERROR] Invalid shape of input layer: shape: {shape} (N,H,W,C), name: {xnode.op_name}"

                # shape of input featuremap
                xnode.shape = shape
                logger.debug(f"property: shape: {xnode.shape}")

            elif op_type == "QuantizeLayer":
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # create XModelNodeFixNeuron object
                xnode = XModelNodeFixNeuron(wrapper_name)
                logger.debug(
                    f"create XModelNodeFixNeuron object: name: {xnode.op_name}"
                )

                quantizer = config.get("quantizer")
                assert (
                    quantizer is not None
                ), f"[ERROR] Not found 'quantizer' field: op type:{xnode.op_type}, name:{xnode.op_name}."
                quantizer_config = quantizer.get("config")
                assert (
                    quantizer_config is not None
                ), f"[ERROR] Not found 'config' field in quantizer: op type:{xnode.op_type}, name:{xnode.op_name}."

                # bit_width
                assert "bit_width" in quantizer_config
                bit_width = quantizer_config.get("bit_width")
                xnode.quant_in["bit_width"] = xnode.quant_out["bit_width"] = int(
                    bit_width
                )

                # quantize_pos
                if param:
                    param_info = param.get(op_name)
                    assert param_info is not None
                    for key, value in param_info.items():
                        if "pos" in key:
                            xnode.quant_in["quantize_pos"] = xnode.quant_out[
                                "quantize_pos"
                            ] = int(value)
                            break

                    logger.debug(
                        f"property: quant_in: bit_width: {xnode.quant_in['bit_width']}, quantize_pos: {xnode.quant_in['quantize_pos']}"
                    )
                    logger.debug(
                        f"property: quant_out: bit_width: {xnode.quant_out['bit_width']}, quantize_pos: {xnode.quant_out['quantize_pos']}"
                    )

            elif op_type == "ZeroPadding2D":
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # pad_mode
                pad_mode = "constant"

                # padding
                padding = inner_layer_config.get("padding")
                assert padding is not None and len(padding) == 2
                padding = [0] * 2 + padding[0] + padding[1] + [0] * 2

                # constant_values
                dtype = inner_layer_config.get("dtype")
                if dtype == "float32":
                    constant_values = [0.0] * 8
                elif dtype == "int32":
                    constant_values = [0] * 8
                else:
                    raise TypeError(f"[ERROR] Unsupported data type: {dtype}")

                # create XModelNodePad object
                xnode = XModelNodePad(wrapper_name, padding, pad_mode, constant_values)
                logger.debug(f"create XModelNodePad object: name: {xnode.op_name}")
                logger.debug(
                    f"property: padding (top, bottom, left, right): {xnode.padding}"
                )
                logger.debug(f"property: pad_mode: {xnode.pad_mode}")
                logger.debug(f"property: constant_values: {xnode.constant_values}")

                if param:
                    for key, value in param.get(wrapper_name).items():
                        if "output_0_pos" in key:
                            xnode.quant_out["quantize_pos"] = int(value)
                            # bit_width
                            quantizer = config.get("quantize_config")
                            qc_config = quantizer.get("config")
                            bit_width = (
                                qc_config.get("output_quantizers")[0]
                                .get("quantizer_params")
                                .get("bit_width")
                            )
                            xnode.quant_out["bit_width"] = int(bit_width)
                            logger.debug(f"property: quant_out: {xnode.quant_out}")
                            break

            elif op_type == "Conv2D":
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # kernel size: (kernel_h, kernel_w)
                ksize = inner_layer_config.get("kernel_size")

                # create XModelNodeConv2d object
                xnode = XModelNodeConv2d(wrapper_name, ksize)
                logger.debug(f"create XModelNodeConv2d object: name: {xnode.op_name}")
                logger.debug(f"property: (kernel_h, kernel_w): {xnode.kernel_size}")

                # group
                xnode.group = inner_layer_config.get("group")
                logger.debug(f"property: group: {xnode.group}")

                # round_mode
                xnode.round_mode = RoundMode.CEIL
                logger.debug(f"property: round_mode: {xnode.round_mode}")

                # [stride_h, stride_w]
                xnode.strides = inner_layer_config.get("strides")
                logger.debug(f"property: (stride_h, stride_w): {xnode.strides}")

                # [dilation_n, dilation_c, dilation_h, dilation_w]
                dilation = inner_layer_config.get("dilation_rate")
                assert len(dilation) == 2
                xnode.dilation = [1, 1] + dilation
                logger.debug(
                    f"property: (dilation_n, dilation_c, dilation_h, dilation_w): {xnode.dilation}"
                )

                # pad mode
                xnode.pad_mode = PadMode[inner_layer_config.get("padding").upper()]
                logger.debug(f"property: pad_mode: {xnode.pad_mode}")

                # bias term
                xnode.bias_term = inner_layer_config.get("use_bias")
                logger.debug(f"property: bias_term: {xnode.bias_term}")

                # data of bias and weights
                if op_name in param:
                    for key, value in param.get(op_name).items():
                        if "bias" in key:
                            xnode.bias = value
                        elif "kernel" in key:
                            xnode.weights = value
                elif wrapper_name in param:
                    xnode.weights = param.get(wrapper_name).get("kernel:0")
                    assert xnode.weights is not None
                    logger.debug(
                        f"property: weights: shape (height, width, in_channels, out_channels): {xnode.weights.shape}, dtype: {xnode.weights.dtype.name}"
                    )
                    if xnode.bias_term:
                        xnode.bias = param.get(wrapper_name).get("bias:0")
                        assert xnode.bias is not None
                        logger.debug(
                            f"property: bias: shape: {xnode.bias.shape}, dtype: {xnode.bias.dtype.name}"
                        )

                # quantization info
                quantize_config = config.get("quantize_config")
                qc_config = quantize_config.get("config")
                quant_info_dict = param.get(wrapper_name)

                assert "kernel_pos:0" in quant_info_dict
                xnode.quant_weights["quantize_pos"] = int(
                    quant_info_dict.get("kernel_pos:0")
                )
                xnode.quant_weights["round_mode"] = 2  # "PY3_ROUND"
                bit_width = (
                    qc_config.get("weight_quantizers")[0]
                    .get("quantizer_params")
                    .get("bit_width")
                )
                xnode.quant_weights["bit_width"] = int(bit_width)
                logger.debug(f"property: quant_weights: {xnode.quant_weights}")

                if xnode.bias_term:
                    assert "bias_pos:0" in quant_info_dict
                    xnode.quant_bias["quantize_pos"] = int(
                        quant_info_dict.get("bias_pos:0")
                    )
                    xnode.quant_bias["round_mode"] = 2  # "PY3_ROUND"

                    if len(qc_config.get("weight_quantizers")) == 2:
                        bit_width = (
                            qc_config.get("weight_quantizers")[1]
                            .get("quantizer_params")
                            .get("bit_width")
                        )
                    elif qc_config.get("bias_quantizers") and \
                          len(qc_config.get("bias_quantizers")) == 1:
                        bit_width = (
                            qc_config.get("bias_quantizers")[0]
                            .get("quantizer_params")
                            .get("bit_width")
                        )
                    else:
                        raise ValueError(
                            f"[ERROR] Not found bias quantize info. Op type: {op_type}, name: {xnode.op_name}"
                        )

                    xnode.quant_bias["bit_width"] = int(bit_width)
                    logger.debug(f"property: quant_bias: {xnode.quant_bias}")

                if "activation" in inner_layer_config:
                    activ_config = inner_layer_config.get("activation").get("config")
                    if "activation" in activ_config:
                        activ_type = activ_config.get("activation")
                        if isinstance(activ_type, str):
                            node = None
                            if activ_type == "sigmoid":
                                # create XModelNodeSigmoid object
                                node = XModelNodeSigmoid(wrapper_name + "_sigmoid")
                                logger.debug(
                                    f"create XModelNodeSigmoid object: name: {node.op_name}"
                                )

                            elif activ_type == "softmax":
                                # create XModelNodeSoftmax object
                                node = XModelNodeSoftmax(wrapper_name + "_softmax")
                                logger.debug(
                                    f"create XModelNodeSoftmax object: name: {node.op_name}"
                                )

                                # -1 means the last axis
                                node.axis = -1
                                logger.debug(f"property: axis: {node.axis}")

                            elif activ_type == "linear":
                                # do nothing according to keras linear activation

                                if "post_activation_pos:0" in quant_info_dict:
                                    # quantize_pos
                                    xnode.quant_out["quantize_pos"] = int(
                                        quant_info_dict.get("post_activation_pos:0")
                                    )

                                    # bit_width
                                    assert (
                                        len(qc_config.get("quantizable_activations"))
                                        == 1
                                    )
                                    bit_width = (
                                        qc_config.get("activation_quantizers")[0]
                                        .get("quantizer_params")
                                        .get("bit_width")
                                    )
                                    xnode.quant_out["bit_width"] = int(bit_width)

                                    logger.debug(
                                        f"property: quant_out: {xnode.quant_out}"
                                    )

                                else:
                                    raise ValueError(f"[ERROR] Unsupported case.")

                            elif activ_type == "relu":
                                # create XModelNodeRelu object
                                node = XModelNodeRelu(wrapper_name + "_relu")
                                logger.debug(
                                    f"create XModelNodeRelu object: name: {node.op_name}"
                                )

                            else:
                                raise ValueError(
                                    f"[ERROR] Unsupported keras activation: {activ_type}. Node name: {xnode.op_name}."
                                )

                            if node is not None:
                                # set layout
                                node.init_layout = node.layout = xmodel.layout
                                logger.debug(
                                    f"property: init layout: {node.init_layout}, current layout: {node.layout}"
                                )

                                # quantization
                                if "pre_activation_pos:0" in quant_info_dict:
                                    # quantize_pos
                                    node.quant_in["quantize_pos"] = int(
                                        quant_info_dict.get("pre_activation_pos:0")
                                    )

                                    # bit_width
                                    quantize_config = config.get("quantize_config")
                                    bit_width = (
                                        quantize_config.get("config")
                                        .get("activation_quantizers")[0]
                                        .get("quantizer_params")
                                        .get("bit_width")
                                    )
                                    node.quant_in["bit_width"] = int(bit_width)

                                    logger.debug(f"property: quant_in: {node.quant_in}")

                                elif "post_activation_pos:0" in quant_info_dict:
                                    # quantize_pos
                                    node.quant_out["quantize_pos"] = int(
                                        quant_info_dict.get("post_activation_pos:0")
                                    )

                                    # bit_width
                                    quantize_config = config.get("quantize_config")
                                    bit_width = (
                                        quantize_config.get("config")
                                        .get("activation_quantizers")[0]
                                        .get("quantizer_params")
                                        .get("bit_width")
                                    )
                                    node.quant_out["bit_width"] = int(bit_width)

                                    logger.debug(
                                        f"property: quant_out: {node.quant_out}"
                                    )

                                # set bottom
                                node.bottom = [wrapper_name]
                                # update xmodel
                                xmodel.add_xnode(node)
                                wrapper_to_xnode[node.op_name] = node

                                # create a mapping for post activation
                                post_act_map[xnode.op_name] = node.op_name

            elif op_type == "Conv2DTranspose":
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
                output_padding = inner_layer_config.get("output_padding")
                assert (
                    output_padding is None
                ), f"[ERROR] 'output_padding' defined in keras Conv2dTranspose is not None: actual: {output_padding}. Node name: {wrapper_name}."

                # kernel size: (kernel_h, kernel_w)
                ksize = inner_layer_config.get("kernel_size")

                # create XModelNodeConv2dTranspose object
                xnode = XModelNodeConv2dTranspose(wrapper_name, ksize)
                logger.debug(
                    f"create XModelNodeConv2dTranspose object: name: {xnode.op_name}"
                )
                logger.debug(f"property: (kernel_h, kernel_w): {xnode.kernel_size}")

                # group
                xnode.group = inner_layer_config.get("group")
                logger.debug(f"property: group: {xnode.group}")

                # round_mode
                xnode.round_mode = RoundMode.CEIL
                logger.debug(f"property: round_mode: {xnode.round_mode}")

                # [stride_h, stride_w]
                xnode.strides = inner_layer_config.get("strides")
                logger.debug(f"property: (stride_h, stride_w): {xnode.strides}")

                # [dilation_n, dilation_c, dilation_h, dilation_w]
                dilation = inner_layer_config.get("dilation_rate")
                assert len(dilation) == 2
                xnode.dilation = [1, 1] + dilation
                logger.debug(
                    f"property: (dilation_n, dilation_c, dilation_h, dilation_w): {xnode.dilation}"
                )

                # pad mode
                xnode.pad_mode = PadMode[inner_layer_config.get("padding").upper()]
                logger.debug(f"property: pad_mode: {xnode.pad_mode}")

                # bias term
                xnode.bias_term = inner_layer_config.get("use_bias")
                logger.debug(f"property: bias_term: {xnode.bias_term}")

                # data of bias and weights
                if param:
                    if op_name in param:
                        for key, value in param.get(op_name).items():
                            if "bias" in key:
                                xnode.bias = value
                            elif "kernel" in key:
                                # (h,w,oc,ic)
                                weights = value
                                _, _, oc, _ = weights.shape
                                assert oc == inner_layer_config.get("filters")
                                xnode.weights = weights

                    elif wrapper_name in param:
                        # (h,w,oc,ic)
                        weights = param.get(wrapper_name).get("kernel:0")
                        assert weights is not None
                        _, _, oc, _ = weights.shape
                        assert oc == inner_layer_config.get("filters")
                        xnode.weights = weights
                        logger.debug(
                            f"property: weights: shape (height, width, out_channels, in_channels): {xnode.weights.shape}, dtype: {xnode.weights.dtype.name}"
                        )
                        if xnode.bias_term:
                            xnode.bias = param.get(wrapper_name).get("bias:0")
                            assert xnode.bias is not None
                            logger.debug(
                                f"property: bias: shape: {xnode.bias.shape}, dtype: {xnode.bias.dtype.name}"
                            )

                # quantization info
                quantize_config = config.get("quantize_config")
                qc_config = quantize_config.get("config")

                # quantize_pos of quant_out of conv2d
                if param:
                    for key, value in param.get(wrapper_name).items():
                        if "post_activation_pos" in key:
                            xnode.quant_out["quantize_pos"] = int(value)

                            # bit_width
                            assert len(qc_config.get("quantizable_activations")) == 1
                            bit_width = (
                                qc_config.get("activation_quantizers")[0]
                                .get("quantizer_params")
                                .get("bit_width")
                            )
                            xnode.quant_out["bit_width"] = int(bit_width)
                            logger.debug(f"property: quant_out: {xnode.quant_out}")

                        elif "kernel_pos" in key:
                            xnode.quant_weights["quantize_pos"] = int(value)
                            xnode.quant_weights["round_mode"] = 2  # "PY3_ROUND"
                            bit_width = (
                                qc_config.get("weight_quantizers")[0]
                                .get("quantizer_params")
                                .get("bit_width")
                            )
                            xnode.quant_weights["bit_width"] = int(bit_width)
                            logger.debug(
                                f"property: quant_weights: {xnode.quant_weights}"
                            )

                        elif "bias_pos" in key:
                            assert xnode.bias_term
                            xnode.quant_bias["quantize_pos"] = int(value)
                            xnode.quant_bias["round_mode"] = 2  # "PY3_ROUND"

                            if len(qc_config.get("weight_quantizers")) == 2:
                                bit_width = (
                                      qc_config.get("weight_quantizers")[1]
                                      .get("quantizer_params")
                                      .get("bit_width")
                                )
                            elif qc_config.get("bias_quantizers") and \
                                  len(qc_config.get("bias_quantizers")) == 1:
                                bit_width = (
                                      qc_config.get("bias_quantizers")[0]
                                      .get("quantizer_params")
                                      .get("bit_width")
                                )
                            else:
                                raise ValueError(
                                    f"[ERROR] Not found bias quantize info. Op type: {op_type}, name: {xnode.op_name}"
                                )

                            xnode.quant_bias["bit_width"] = int(bit_width)
                            logger.debug(f"property: quant_bias: {xnode.quant_bias}")

            elif op_type in ["relu", "ReLU"]:
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # max_value
                max_value = 0
                if (
                    "max_value" in inner_layer_config
                    and inner_layer_config.get("max_value") is not None
                ):
                    max_value = int(inner_layer_config.get("max_value"))

                if max_value == 0:
                    # create XModelNodeRelu object
                    xnode = XModelNodeRelu(wrapper_name)
                    logger.debug(f"create XModelNodeRelu object: name: {xnode.op_name}")
                elif max_value == 6:
                    # create XModelNodeRelu6 object
                    xnode = XModelNodeRelu6(wrapper_name)
                    logger.debug(f"create XModelNodeRelu object: name: {xnode.op_name}")
                else:
                    raise ValueError(
                        f"[ERROR] Unsupported keras relu: max_value:{max_value}"
                    )

                # set negative_slope
                xnode.alpha = 0.0
                if "negative_slope" in inner_layer_config:
                    xnode.alpha = inner_layer_config.get("negative_slope")
                logger.debug(f"property: alpha: {xnode.alpha}")

                # quantization info
                if param:
                    if "post_activation_pos:0" in param.get(wrapper_name):
                        # quantize_pos
                        quantize_pos = param.get(wrapper_name).get(
                            "post_activation_pos:0"
                        )
                        assert quantize_pos is not None
                        xnode.quant_out["quantize_pos"] = int(quantize_pos)

                        # bit_width
                        quantize_config = config.get("quantize_config")
                        qc_config = quantize_config.get("config")
                        bit_width = (
                            qc_config.get("activation_quantizers")[0]
                            .get("quantizer_params")
                            .get("bit_width")
                        )
                        xnode.quant_out["bit_width"] = int(bit_width)
                        logger.debug(f"property: quant_out: {xnode.quant_out}")
                    elif "output_0_pos:0" in param.get(wrapper_name):
                        # quantize_pos
                        quantize_pos = param.get(wrapper_name).get("output_0_pos:0")
                        assert quantize_pos is not None
                        xnode.quant_out["quantize_pos"] = int(quantize_pos)

                        # bit_width
                        quantize_config = config.get("quantize_config")
                        qc_config = quantize_config.get("config")
                        bit_width = (
                            qc_config.get("output_quantizers")[0]
                            .get("quantizer_params")
                            .get("bit_width")
                        )
                        xnode.quant_out["bit_width"] = int(bit_width)
                        logger.debug(f"property: quant_out: {xnode.quant_out}")

            elif op_type == "LeakyReLU":
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # create XModelNodeRelu object
                xnode = XModelNodeRelu(wrapper_name)
                logger.debug(f"create XModelNodeRelu object: name: {xnode.op_name}")

                # set negative_slope
                # for those leaky_relu layers with alpha=0.1015625, they are quantized, and there is an inner layer, which holds the alpha value.
                # for other leaky_relu layers, they are not quantized, and there is no inner layer. The alpha value is held directly by outer layer.
                alpha = 0.0
                if "layer" in config:
                    inner_layer_config = config.get("layer").get("config")
                    alpha = inner_layer_config.get("alpha")
                else:
                    alpha = config.get("alpha")
                xnode.alpha = alpha
                logger.debug(f"property: alpha: {xnode.alpha}")

                # quantization info
                if param:
                    if "post_activation_pos:0" in param.get(wrapper_name):
                        # quantize_pos
                        quantize_pos = param.get(wrapper_name).get(
                            "post_activation_pos:0"
                        )
                        assert quantize_pos is not None
                        xnode.quant_out["quantize_pos"] = int(quantize_pos)

                        # bit_width
                        quantize_config = config.get("quantize_config")
                        qc_config = quantize_config.get("config")
                        bit_width = (
                            qc_config.get("activation_quantizers")[0]
                            .get("quantizer_params")
                            .get("bit_width")
                        )
                        xnode.quant_out["bit_width"] = int(bit_width)
                        logger.debug(f"property: quant_out: {xnode.quant_out}")
                    elif "output_0_pos:0" in param.get(wrapper_name):
                        # quantize_pos
                        quantize_pos = param.get(wrapper_name).get("output_0_pos:0")
                        assert quantize_pos is not None
                        xnode.quant_out["quantize_pos"] = int(quantize_pos)

                        # bit_width
                        quantize_config = config.get("quantize_config")
                        qc_config = quantize_config.get("config")
                        bit_width = (
                            qc_config.get("output_quantizers")[0]
                            .get("quantizer_params")
                            .get("bit_width")
                        )
                        xnode.quant_out["bit_width"] = int(bit_width)
                        logger.debug(f"property: quant_out: {xnode.quant_out}")

            elif op_type in ["MaxPooling2D", "AveragePooling2D"]:
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # kernel size
                ksize = inner_layer_config.get("pool_size")

                # create pooling object
                xnode = None
                if op_type == "MaxPooling2D":
                    xnode = XModelNodeMaxPool(wrapper_name, ksize)
                    logger.debug(
                        f"create XModelNodeMaxPool object: name: {xnode.op_name}"
                    )
                else:
                    xnode = XModelNodeAvgPool(wrapper_name, ksize)
                    logger.debug(
                        f"create XModelNodeAvgPool object: name: {xnode.op_name}"
                    )
                    # count_include_pad
                    xnode.count_include_pad = False
                    logger.debug(
                        f"property: count_inclue_pad: {xnode.count_include_pad}"
                    )
                logger.debug(f"property: (kernel_h, kernel_w): {xnode.kernel_size}")

                # round_mode
                xnode.round_mode = RoundMode.CEIL
                logger.debug(f"property: round_mode: {xnode.round_mode}")

                # [stride_h, stride_w]
                xnode.strides = inner_layer_config.get("strides")
                logger.debug(f"property: (stride_h, stride_w): {xnode.strides}")

                # pad mode
                xnode.pad_mode = PadMode[inner_layer_config.get("padding").upper()]
                logger.debug(f"property: pad_mode: {xnode.pad_mode}")

                # quantize_pos
                if param:
                    if "output_0_pos:0" in param.get(wrapper_name):
                        # quantize_pos
                        xnode.quant_out["quantize_pos"] = int(
                            param.get(wrapper_name).get("output_0_pos:0")
                        )
                        # bit_width
                        quantizer = config.get("quantize_config")
                        qc_config = quantizer.get("config")
                        bit_width = (
                            qc_config.get("output_quantizers")[0]
                            .get("quantizer_params")
                            .get("bit_width")
                        )
                        xnode.quant_out["bit_width"] = int(bit_width)
                        logger.debug(f"property: quant_out: {xnode.quant_out}")

            elif op_type in ["Add", "addv2"]:
                bottom = []
                for x in layer.get("inbound_nodes")[0]:
                    bottom += [y for y in x if isinstance(y, str)]
                assert (
                    len(bottom) >= 2
                ), f"[ERROR] keras {op_type} requires two or more inputs: actual:{len(bottom)}"
                if bottom[0] in post_act_map:
                    bottom[0] = post_act_map.get(bottom[0])

                # create XModelNodeElemAdd object
                xnode = XModelNodeElemAdd(wrapper_name)
                logger.debug(f"create XModelNodeElemAdd object: name: {xnode.op_name}")

                # set coefficient
                xnode.alpha = [1.0] * len(bottom)
                logger.debug(f"property: alpha: {xnode.alpha}")

                # quantization info
                if param:
                    for key, value in param.get(wrapper_name).items():
                        if "output_0_pos" in key:
                            # quantize_pos
                            xnode.quant_out["quantize_pos"] = int(value)

                            # bit_width
                            quantize_config = config.get("quantize_config")
                            qc_config = quantize_config.get("config")
                            bit_width = (
                                qc_config.get("output_quantizers")[0]
                                .get("quantizer_params")
                                .get("bit_width")
                            )
                            xnode.quant_out["bit_width"] = int(bit_width)

                            logger.debug(f"property: quant_out: {xnode.quant_out}")
                            break

            elif op_type == "Multiply":
                bottom = []
                for x in layer.get("inbound_nodes")[0]:
                    bottom += [y for y in x if isinstance(y, str)]
                assert (
                    len(bottom) >= 2
                ), f"[ERROR] keras {op_type} requires two or more inputs: actual:{len(bottom)}"
                if bottom[0] in post_act_map:
                    bottom[0] = post_act_map.get(bottom[0])

                # create XModelNodeElemMul object
                xnode = XModelNodeElemMul(wrapper_name)
                logger.debug(f"create XModelNodeElemMul object: name: {xnode.op_name}")

                # quantization info
                if param:
                    for key, value in param.get(wrapper_name).items():
                        if "output_0_pos" in key:
                            # quantize_pos
                            xnode.quant_out["quantize_pos"] = int(value)

                            # bit_width
                            quantize_config = config.get("quantize_config")
                            qc_config = quantize_config.get("config")
                            bit_width = (
                                qc_config.get("output_quantizers")[0]
                                .get("quantizer_params")
                                .get("bit_width")
                            )
                            xnode.quant_out["bit_width"] = int(bit_width)

                            logger.debug(f"property: quant_out: {xnode.quant_out}")
                            break

            elif op_type == "GlobalAveragePooling2D":
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # get kernel size
                ksize = [0, 0]

                # create pooling object
                xnode = XModelNodeAvgPool(wrapper_name, ksize)
                logger.debug(f"create XModelNodeAvgPool object: name: {xnode.op_name}")

                # global
                xnode.is_global = True
                logger.debug(f"property: is_global: {xnode.is_global}")

                # count_include_pad
                xnode.count_include_pad = False
                logger.debug(f"property: count_inclue_pad: {xnode.count_include_pad}")

                # quantization info
                if param:
                    for key, value in param.get(wrapper_name).items():
                        if "output_0_pos" in key:
                            # quantize_pos
                            xnode.quant_out["quantize_pos"] = int(value)

                            # bit_width
                            quantize_config = config.get("quantize_config")
                            qc_config = quantize_config.get("config")
                            bit_width = (
                                qc_config.get("output_quantizers")[0]
                                .get("quantizer_params")
                                .get("bit_width")
                            )
                            xnode.quant_out["bit_width"] = int(bit_width)

                            logger.debug(f"property: quant_out: {xnode.quant_out}")
                            break

            elif op_type == "Concatenate":
                bottom = []
                for x in layer.get("inbound_nodes")[0]:
                    bottom += [y for y in x if isinstance(y, str)]
                assert (
                    len(bottom) >= 1
                ), f"[ERROR] keras {op_type} requires one or more inputs: actual:{len(bottom)}"
                if bottom[0] in post_act_map:
                    bottom[0] = post_act_map.get(bottom[0])

                # create XModelNodeConcat object
                xnode = XModelNodeConcat(wrapper_name)
                logger.debug(f"create XModelNodeConcat object: name: {xnode.op_name}")

                # set axis
                assert "axis" in inner_layer_config
                axis = inner_layer_config.get("axis")
                assert axis is not None
                xnode.axis = int(axis)
                logger.debug(f"property: axis: {xnode.axis}")

                # quantize_pos
                if param and "output_0_pos:0" in param.get(wrapper_name):
                    # quantize_pos
                    xnode.quant_out["quantize_pos"] = int(
                        param.get(wrapper_name).get("output_0_pos:0")
                    )
                    # bit_width
                    quantizer = config.get("quantize_config")
                    qc_config = quantizer.get("config")
                    bit_width = (
                        qc_config.get("output_quantizers")[0]
                        .get("quantizer_params")
                        .get("bit_width")
                    )
                    xnode.quant_out["bit_width"] = int(bit_width)
                    logger.debug(f"property: quant_out: {xnode.quant_out}")

            elif op_type == "Dense":
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # create XModelNodeMatMul object
                xnode = XModelNodeMatMul(wrapper_name)
                logger.debug(f"create XModelNodeMatMul object: name: {xnode.op_name}")

                # bias term
                bias_term = inner_layer_config.get("use_bias")
                if bias_term:
                    bias = param.get(wrapper_name).get("bias:0")
                    if bias is None:
                        bias = param.get(op_name).get("bias:0")
                    assert (
                        bias is not None
                    ), f"[ERROR] Not found bias tensor. Op type: {op_type}, name: {wrapper_name}"
                    xnode.bias = bias
                    logger.debug(
                        f"property: bias: shape: {xnode.bias.shape}, dtype: {xnode.bias.dtype.name}"
                    )

                # quantization info
                quantize_config = config.get("quantize_config")
                weight_quantizers = quantize_config.get("config").get(
                    "weight_quantizers"
                )

                # quantization info of bias
                if xnode.bias is not None:
                    bias_pos = param.get(wrapper_name).get("bias_pos:0")
                    assert bias_pos is not None
                    xnode.quant_bias["quantize_pos"] = int(bias_pos)
                    xnode.quant_bias["round_mode"] = 2  # "PY3_ROUND"

                    if len(weight_quantizers) == 2:
                        xnode.quant_bias["bit_width"] = int(
                            weight_quantizers[1].get("quantizer_params").get("bit_width")
                        )
                    elif quantize_config['config'].get("bias_quantizers") and \
                          len(quantize_config['config'].get("bias_quantizers")) == 1:
                        xnode.quant_bias["bit_width"] = int(
                            quantize_config['config'].get("bias_quantizers")[0].get("quantizer_params").get("bit_width")
                        )
                    else:
                        raise ValueError(
                          f"[ERROR] Not found bias quantize info. Op type: {op_type}, name: {xnode.op_name}"
                        )

                    logger.debug(f"property: quant_bias: {xnode.quant_bias}")

                # weights
                weights = param.get(wrapper_name).get("kernel:0")
                if weights is None:
                    weights = param.get(op_name).get("kernel:0")
                assert (
                    weights is not None
                ), f"[ERROR] Not found weights tensor. Op type: {op_type}, name: {wrapper_name}."
                # units
                units = inner_layer_config.get("units")
                assert weights.shape[-1] == units

                # step 1: create XModelNodeConst object
                const_xnode = XModelNodeConst(wrapper_name + "_const")
                const_xnode.init_layout = const_xnode.layout = xmodel.layout
                const_xnode.tensor = weights
                const_xnode.tensor.data_format = DataFormat[xmodel.layout]
                # update xmodel with const_node
                xmodel.add_xnode(const_xnode)
                wrapper_to_xnode[const_xnode.op_name] = const_xnode

                # step 2: create XModelNodeFixNeuron with quantization info
                quant_xnode = XModelNodeFixNeuron(const_xnode.op_name + "_fix")

                kernel_pos = param.get(wrapper_name).get("kernel_pos:0")
                assert kernel_pos is not None
                kernel_pos = int(kernel_pos)
                bit_width = int(
                    weight_quantizers[0].get("quantizer_params").get("bit_width")
                )
                quant_xnode.init_layout = quant_xnode.layout = xmodel.layout
                quant_xnode.quant_in["bit_width"] = bit_width
                quant_xnode.quant_in["quantize_pos"] = kernel_pos
                quant_xnode.quant_out["bit_width"] = bit_width
                quant_xnode.quant_out["quantize_pos"] = kernel_pos
                quant_xnode.is_quantized = True
                # set input
                quant_xnode.bottom = [const_xnode.op_name]
                # update xmodel with quant_xnode
                xmodel.add_xnode(quant_xnode)
                wrapper_to_xnode[quant_xnode.op_name] = quant_xnode

                # step 3: add quant_xnode as a parent of xnode
                bottom.append(quant_xnode.op_name)

                if "activation" in inner_layer_config:
                    activ_config = inner_layer_config.get("activation").get("config")
                    if "activation" in activ_config:

                        activ_type = activ_config.get("activation")
                        if isinstance(activ_type, str):
                            # quantization info
                            quantize_config = config.get("quantize_config")
                            qc_config = quantize_config.get("config")
                            quant_info_dict = param.get(wrapper_name)

                            node = None
                            if activ_type == "sigmoid":
                                # create XModelNodeSigmoid object
                                node = XModelNodeSigmoid(wrapper_name + "_sigmoid")
                                logger.debug(
                                    f"create XModelNodeSigmoid object: name: {node.op_name}"
                                )

                            elif activ_type == "softmax":
                                # create XModelNodeSoftmax object
                                node = XModelNodeSoftmax(wrapper_name + "_softmax")
                                logger.debug(
                                    f"create XModelNodeSoftmax object: name: {node.op_name}"
                                )

                                # -1 means the last axis
                                node.axis = -1
                                logger.debug(f"property: axis: {node.axis}")

                            elif activ_type == "linear":
                                # do nothing according to keras linear activation

                                if "post_activation_pos:0" in quant_info_dict:
                                    # quantize_pos
                                    xnode.quant_out["quantize_pos"] = int(
                                        quant_info_dict.get("post_activation_pos:0")
                                    )

                                    # bit_width
                                    assert (
                                        len(qc_config.get("quantizable_activations"))
                                        == 1
                                    )
                                    bit_width = (
                                        qc_config.get("activation_quantizers")[0]
                                        .get("quantizer_params")
                                        .get("bit_width")
                                    )
                                    xnode.quant_out["bit_width"] = int(bit_width)

                                    logger.debug(
                                        f"property: quant_out: {xnode.quant_out}"
                                    )

                                else:
                                    raise ValueError(f"[ERROR] Unsupported case.")

                            elif activ_type == "relu":
                                # create XModelNodeRelu object
                                node = XModelNodeRelu(wrapper_name + "_relu")
                                logger.debug(
                                    f"create XModelNodeRelu object: name: {node.op_name}"
                                )

                            else:
                                raise ValueError(
                                    f"[ERROR] Unsupported keras activation: {activ_type}. Node name: {xnode.op_name}, type: {xnode.op_type}"
                                )

                            if node is not None:
                                # set layout
                                node.init_layout = node.layout = xmodel.layout
                                logger.debug(
                                    f"property: init layout: {node.init_layout}, current layout: {node.layout}"
                                )

                                # quantization
                                if "pre_activation_pos:0" in quant_info_dict:
                                    # quantize_pos
                                    node.quant_in["quantize_pos"] = int(
                                        quant_info_dict.get("pre_activation_pos:0")
                                    )

                                    # bit_width
                                    quantize_config = config.get("quantize_config")
                                    bit_width = (
                                        quantize_config.get("config")
                                        .get("activation_quantizers")[0]
                                        .get("quantizer_params")
                                        .get("bit_width")
                                    )
                                    node.quant_in["bit_width"] = int(bit_width)

                                    logger.debug(f"property: quant_in: {node.quant_in}")

                                elif "post_activation_pos:0" in quant_info_dict:
                                    # quantize_pos
                                    node.quant_out["quantize_pos"] = int(
                                        quant_info_dict.get("post_activation_pos:0")
                                    )

                                    # bit_width
                                    quantize_config = config.get("quantize_config")
                                    bit_width = (
                                        quantize_config.get("config")
                                        .get("activation_quantizers")[0]
                                        .get("quantizer_params")
                                        .get("bit_width")
                                    )
                                    node.quant_out["bit_width"] = int(bit_width)

                                    logger.debug(
                                        f"property: quant_out: {node.quant_out}"
                                    )

                                # set bottom
                                node.bottom = [wrapper_name]
                                # update xmodel
                                xmodel.add_xnode(node)
                                wrapper_to_xnode[node.op_name] = node

                                # create a mapping for post activation
                                post_act_map[xnode.op_name] = node.op_name

            elif op_type in ["softmax", "Softmax"]:
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # create XModelNodeSoftmax object
                xnode = XModelNodeSoftmax(wrapper_name)
                logger.debug(f"create XModelNodeSoftmax object: name: {xnode.op_name}")

                # -1 means the last axis
                axis = -1
                if "axis" in activ_config:
                    axis = activ_config.get("axis")
                    assert isinstance(axis, int)
                xnode.axis = axis
                logger.debug(f"property: axis: {xnode.axis}")

                if param:
                    # quantize_pos of quant_in
                    assert "pre_activation_pos:0" in param.get(wrapper_name)
                    pre_activation_pos = param.get(wrapper_name).get(
                        "pre_activation_pos:0"
                    )
                    assert pre_activation_pos is not None
                    xnode.quant_in["quantize_pos"] = int(pre_activation_pos)

                    # bit_width of quant_in
                    quantize_config = config.get("quantize_config")
                    assert quantize_config is not None
                    qc_config = quantize_config.get("config")
                    assert qc_config is not None
                    bit_width = (
                        qc_config.get("activation_quantizers")[0]
                        .get("quantizer_params")
                        .get("bit_width")
                    )
                    assert bit_width is not None
                    xnode.quant_in["bit_width"] = int(bit_width)
                    logger.debug(f"property: quant_in: {xnode.quant_in}")

            elif op_type == "DepthwiseConv2D":
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                data_format = inner_layer_config.get("data_format")
                assert (
                    data_format == "channels_last"
                ), f"[ERROR] xnnc only supports the keras DepthwiseConv2D with the data format of 'channel_last'."

                # kernel_size
                ksize = inner_layer_config.get("kernel_size")

                # create XModelNodeConv2dDepthwise object
                xnode = XModelNodeConv2dDepthwise(wrapper_name, ksize)
                logger.debug(
                    f"create XModelNodeConv2dDepthwise object: name: {xnode.op_name}"
                )
                logger.debug(f"property: (kernel_h, kernel_w): {xnode.kernel_size}")

                # round_mode
                xnode.round_mode = RoundMode.CEIL
                logger.debug(f"property: round_mode: {xnode.round_mode}")

                # strides: [stride_h, stride_w]
                xnode.strides = inner_layer_config.get("strides")
                logger.debug(f"property: (stride_h, stride_w): {xnode.strides}")

                # dilations: [dilation_n, dilation_c, dilation_h, dilation_w]
                xnode.dilation = [1, 1] + inner_layer_config.get("dilation_rate")
                logger.debug(
                    f"property: (dilation_n, dilation_c, dilation_h, dilation_w): {xnode.dilation}"
                )

                # pad_mode
                xnode.pad_mode = PadMode[inner_layer_config.get("padding").upper()]
                logger.debug(f"property: pad_mode: {xnode.pad_mode}")

                # bias_term
                xnode.bias_term = inner_layer_config.get("use_bias")

                # data of bias and weights
                if op_name in param:
                    for key, value in param.get(op_name).items():
                        if "bias:0" in key:
                            xnode.bias = value
                        elif "depthwise_kernel:0" in key:
                            xnode.weights = value
                elif wrapper_name in param:
                    xnode.weights = param.get(wrapper_name).get("depthwise_kernel:0")
                    assert xnode.weights is not None
                    logger.debug(
                        f"property: weights: shape (height, width, in_channels, out_channels): {xnode.weights.shape}, dtype: {xnode.weights.dtype.name}"
                    )
                    if xnode.bias_term:
                        xnode.bias = param.get(wrapper_name).get("bias:0")
                        assert xnode.bias is not None
                        logger.debug(
                            f"property: bias: shape: {xnode.bias.shape}, dtype: {xnode.bias.dtype.name}"
                        )
                assert xnode.weights is not None
                if xnode.bias_term:
                    assert xnode.bias is not None

                _, _, IC, CM = xnode.weights.shape
                # group: in_channels
                xnode.group = IC
                logger.debug(f"property: group: {xnode.group}")

                # num of output: in_channels * channel_multiplier(also, oc)
                xnode.num_output = IC * CM
                logger.debug(f"property: num_output: {xnode.num_output}")

                # quantization info
                quantize_config = config.get("quantize_config")
                assert quantize_config is not None
                qc_config = quantize_config.get("config")
                assert qc_config is not None
                weight_quantizers = qc_config.get("weight_quantizers")
                assert weight_quantizers is not None and len(weight_quantizers) > 0

                xnode.quant_weights["bit_width"] = int(
                    weight_quantizers[0].get("quantizer_params").get("bit_width")
                )
                xnode.quant_weights["quantize_pos"] = int(
                    param.get(wrapper_name).get("depthwise_kernel_pos:0")
                )
                xnode.quant_weights["round_mode"] = 2  # "PY3_ROUND"
                logger.debug(f"property: quant_weights: {xnode.quant_weights}")

                if xnode.bias_term:
                    if len(weight_quantizers) == 2:
                        xnode.quant_bias["bit_width"] = int(
                            weight_quantizers[1].get("quantizer_params").get("bit_width")
                        )
                    elif qc_config.get("bias_quantizers") and \
                          len(qc_config.get("bias_quantizers")) == 1:
                        xnode.quant_bias["bit_width"] = int(
                            qc_config.get("bias_quantizers")[0].get("quantizer_params").get("bit_width")
                        )
                    else:
                        raise ValueError(
                            f"[ERROR] Not found bias quantize info. Op type: {op_type}, name: {xnode.op_name}"
                        )

                    xnode.quant_bias["quantize_pos"] = int(
                        param.get(wrapper_name).get("bias_pos:0")
                    )
                    xnode.quant_bias["round_mode"] = 2  # "PY3_ROUND"
                    logger.debug(f"property: quant_bias: {xnode.quant_bias}")

                if "activation" in inner_layer_config:
                    activ_config = inner_layer_config.get("activation").get("config")
                    if "activation" in activ_config:
                        activ_type = activ_config.get("activation")
                        if isinstance(activ_type, str):
                            quant_info_dict = param.get(wrapper_name)
                            node = None
                            if activ_type == "sigmoid":
                                # create XModelNodeSigmoid object
                                node = XModelNodeSigmoid(wrapper_name + "_sigmoid")
                                logger.debug(
                                    f"create XModelNodeSigmoid object: name: {node.op_name}"
                                )

                            elif activ_type == "softmax":
                                # create XModelNodeSoftmax object
                                node = XModelNodeSoftmax(wrapper_name + "_softmax")
                                logger.debug(
                                    f"create XModelNodeSoftmax object: name: {node.op_name}"
                                )

                                # -1 means the last axis
                                node.axis = -1
                                logger.debug(f"property: axis: {node.axis}")

                            elif activ_type == "linear":
                                # do nothing according to keras linear activation

                                if "post_activation_pos:0" in quant_info_dict:
                                    # quantize_pos
                                    xnode.quant_out["quantize_pos"] = int(
                                        quant_info_dict.get("post_activation_pos:0")
                                    )

                                    # bit_width
                                    assert (
                                        len(qc_config.get("quantizable_activations"))
                                        == 1
                                    )
                                    bit_width = (
                                        qc_config.get("activation_quantizers")[0]
                                        .get("quantizer_params")
                                        .get("bit_width")
                                    )
                                    xnode.quant_out["bit_width"] = int(bit_width)

                                    logger.debug(
                                        f"property: quant_out: {xnode.quant_out}"
                                    )

                                else:
                                    raise ValueError(f"[ERROR] Unsupported case.")

                            elif activ_type == "relu":
                                # create XModelNodeRelu object
                                node = XModelNodeRelu(wrapper_name + "_relu")
                                logger.debug(
                                    f"create XModelNodeRelu object: name: {node.op_name}"
                                )

                            else:
                                raise ValueError(
                                    f"[ERROR] Unsupported keras activation: {activ_type}. Node name: {xnode.op_name}."
                                )

                            if node is not None:
                                # set layout
                                node.init_layout = node.layout = xmodel.layout
                                logger.debug(
                                    f"property: init layout: {node.init_layout}, current layout: {node.layout}"
                                )

                                # quantization
                                if "pre_activation_pos:0" in quant_info_dict:
                                    # quantize_pos
                                    node.quant_in["quantize_pos"] = int(
                                        quant_info_dict.get("pre_activation_pos:0")
                                    )

                                    # bit_width
                                    quantize_config = config.get("quantize_config")
                                    bit_width = (
                                        quantize_config.get("config")
                                        .get("activation_quantizers")[0]
                                        .get("quantizer_params")
                                        .get("bit_width")
                                    )
                                    node.quant_in["bit_width"] = int(bit_width)

                                    logger.debug(f"property: quant_in: {node.quant_in}")

                                elif "post_activation_pos:0" in quant_info_dict:
                                    # quantize_pos
                                    node.quant_out["quantize_pos"] = int(
                                        quant_info_dict.get("post_activation_pos:0")
                                    )

                                    # bit_width
                                    quantize_config = config.get("quantize_config")
                                    bit_width = (
                                        quantize_config.get("config")
                                        .get("activation_quantizers")[0]
                                        .get("quantizer_params")
                                        .get("bit_width")
                                    )
                                    node.quant_out["bit_width"] = int(bit_width)

                                    logger.debug(
                                        f"property: quant_out: {node.quant_out}"
                                    )

                                # set bottom
                                node.bottom = [wrapper_name]
                                # update xmodel
                                xmodel.add_xnode(node)
                                wrapper_to_xnode[node.op_name] = node

                                # create a mapping for post activation
                                post_act_map[xnode.op_name] = node.op_name

            elif op_type in ["reshape", "Reshape"]:
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # create XModelNodeReshape object
                xnode = XModelNodeReshape(wrapper_name)
                logger.debug(f"create XModelNodeReshape object: name: {xnode.op_name}")

                # target_shape: output shape is (batch_size, ) + target_shape
                # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape#output_shape
                target_shape = inner_layer_config.get("target_shape")
                assert target_shape is not None
                # [1] is a dummy batch
                xnode.shape = [1] + target_shape
                logger.debug(f"property: shape: {xnode.shape}")

            elif op_type == "UpSampling2D":
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # create XModelNodeUpsample object
                xnode = XModelNodeUpsample(wrapper_name)
                logger.debug(f"create XModelNodeUpsample object: name: {xnode.op_name}")

                # scale: (N,C,H,W)
                size = inner_layer_config.get("size")
                assert size is not None and len(size) == 2
                xnode.scale = [1, 1] + size
                logger.debug(f"property: scale (n,c,h,w): {xnode.scale}")

                # interpolation mode
                mode = inner_layer_config.get("interpolation")
                assert mode is not None and mode in ["nearest", "bilinear"]
                xnode.mode = mode
                logger.debug(f"property: mode: {xnode.mode}")

                # quantization info
                if param and "output_0_pos:0" in param.get(wrapper_name):
                    # quantize_pos
                    quantize_pos = param.get(wrapper_name).get("output_0_pos:0")
                    assert quantize_pos is not None
                    xnode.quant_out["quantize_pos"] = int(quantize_pos)

                    # bit_width
                    quantize_config = config.get("quantize_config")
                    qc_config = quantize_config.get("config")
                    bit_width = (
                        qc_config.get("output_quantizers")[0]
                        .get("quantizer_params")
                        .get("bit_width")
                    )
                    xnode.quant_out["bit_width"] = int(bit_width)
                    logger.debug(f"property: quant_out: {xnode.quant_out}")

            elif op_type == "Flatten":
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # create XModelNodeFlatten object
                xnode = XModelNodeFlatten(wrapper_name)
                logger.debug(f"create XModelNodeFlatten object: name: {xnode.op_name}")

                # Flattens the input. Does not affect the batch size.
                # start_dim
                xnode.start_dim = 1
                # end_dim
                xnode.end_dim = -1

            elif op_type == "linear":
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # create XModelNode object
                xnode = XModelNode(wrapper_name, "activation_linear")
                logger.debug(f"create XModelNode object: name: {xnode.op_name}")

            elif op_type == "UpsampleLike":
                # ! Notice (smashin)
                # ! UpsampleLike is a custom operator by customers, which may different from other implementations

                bottom = []
                for x in layer.get("inbound_nodes")[0]:
                    bottom += [y for y in x if isinstance(y, str)]
                assert (
                    len(bottom) >= 1
                ), f"[ERROR] keras {op_type} requires one or more input: actual:{len(bottom)}"
                if bottom[0] in post_act_map:
                    bottom[0] = post_act_map.get(bottom[0])

                # create XModelNodeResizeBiliear object
                xnode = XModelNodeResize(wrapper_name, "bilinear")
                logger.debug(
                    f"create XModelNodeResizeBiliear object: name: {xnode.op_name}"
                )

                # half_pixel_centers
                xnode.half_pixel_centers = True

                # quantize_pos
                if param is not None and "output_0_pos:0" in param.get(wrapper_name):
                    # quantize_pos
                    xnode.quant_out["quantize_pos"] = int(
                        param.get(wrapper_name).get("output_0_pos:0")
                    )
                    # bit_width
                    quantizer = config.get("quantize_config")
                    qc_config = quantizer.get("config")
                    bit_width = (
                        qc_config.get("output_quantizers")[0]
                        .get("quantizer_params")
                        .get("bit_width")
                    )
                    xnode.quant_out["bit_width"] = int(bit_width)
                    logger.debug(f"property: quant_out: {xnode.quant_out}")

            elif op_type in ["sigmoid", "Sigmoid"]:
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # create XModelNodeSigmoid object
                xnode = XModelNodeSigmoid(wrapper_name)
                logger.debug(f"create XModelNodeSigmoid object: name: {xnode.op_name}")

                # hard sigmoid or activation sigmoid
                if inner_layer.get("class_name").endswith("VitisSigmoid"):
                    xnode.tmp_params["hard_sigmoid"] = True
                else:
                    xnode.tmp_params["hard_sigmoid"] = False

                # quantization info
                if param and "output_0_pos:0" in param.get(wrapper_name):
                    # quantize_pos
                    quantize_pos = param.get(wrapper_name).get("output_0_pos:0")
                    assert quantize_pos is not None
                    xnode.quant_out["quantize_pos"] = int(quantize_pos)

                    quantize_config = config.get("quantize_config")
                    qc_config = quantize_config.get("config")
                    # bit_width
                    bit_width = (
                        qc_config.get("output_quantizers")[0]
                        .get("quantizer_params")
                        .get("bit_width")
                    )
                    xnode.quant_out["bit_width"] = int(bit_width)
                    logger.debug(f"property: quant_out: {xnode.quant_out}")

            elif op_type == "mul":
                bottom = []
                for x in layer.get("inbound_nodes")[0]:
                    bottom += [y for y in x if isinstance(y, str)]
                if bottom[0] in post_act_map:
                    bottom[0] = post_act_map.get(bottom[0])

                if len(bottom) == 1:
                    constants = config.get("constants")
                    assert (
                        constants is not None
                    ), f"[ERROR] Not found 'constants' field when translate tf2 {op_type}: {wrapper_name}."

                    # create an XModelNodeConst object
                    const_xnode = XModelNodeConst(f"{wrapper_name}_const")

                    # tensor
                    if "0" in constants:
                        value = constants.get("0")
                        const_xnode.tensor = XTensor(np.array([value])).astype(
                            np.float32
                        )
                        bottom = [const_xnode.op_name] + bottom
                    elif "1" in constants:
                        value = constants.get("1")
                        const_xnode.tensor = XTensor(np.array([value])).astype(
                            np.float32
                        )
                        bottom.append(const_xnode.op_name)
                    else:
                        raise ValueError(
                            f"ERROR] Unsupported value for the 'constants' field when translate tf2 {op_type}: {wrapper_name}."
                        )

                    # update xmodel
                    xmodel.add_xnode(const_xnode)
                    wrapper_to_xnode[const_xnode.op_name] = const_xnode

                # create XModelNodeElemMul object
                xnode = XModelNodeElemMul(wrapper_name)
                logger.debug(f"create XModelNodeElemMul object: name: {xnode.op_name}")

                # quantize_pos
                if param is not None and "output_0_pos:0" in param.get(wrapper_name):
                    # quantize_pos
                    xnode.quant_out["quantize_pos"] = int(
                        param.get(wrapper_name).get("output_0_pos:0")
                    )
                    # bit_width
                    quantizer = config.get("quantize_config")
                    qc_config = quantizer.get("config")
                    bit_width = (
                        qc_config.get("output_quantizers")[0]
                        .get("quantizer_params")
                        .get("bit_width")
                    )
                    xnode.quant_out["bit_width"] = int(bit_width)
                    logger.debug(f"property: quant_out: {xnode.quant_out}")

            elif op_type == 'PReLU':
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # create XModelNodePRelu object
                xnode = XModelNodePRelu(wrapper_name)
                logger.debug(f"create XModelNodePRelu object: name: {xnode.op_name}")

                xnode.alpha = param.get(wrapper_name).get("alpha:0")
                logger.debug(f"property: alpha: {xnode.alpha}")

                # quantization info
                if param:
                    if "alpha:0" in param.get(wrapper_name) and "alpha_pos:0" in param.get(wrapper_name):
                        # quantize_pos
                        quantize_pos = param.get(wrapper_name).get(
                            "alpha_pos:0"
                        )

                        assert quantize_pos is not None
                        xnode.quant_alpha["quantize_pos"] = int(quantize_pos)

                        quantize_config = config.get("quantize_config")
                        qc_config = quantize_config.get("config")
                        # round_mode
                        round_mode = (
                            qc_config.get("weight_quantizers")[0]
                            .get("quantizer_params")
                            .get("round_mode")
                        )
                        xnode.quant_alpha["round_mode"] = round_mode
                        # bit_width
                        bit_width = (
                            qc_config.get("weight_quantizers")[0]
                            .get("quantizer_params")
                            .get("bit_width")
                        )
                        xnode.quant_alpha["bit_width"] = int(bit_width)
                        logger.debug(f"property: quant_alpha: {xnode.quant_alpha}")
                    if "output_0_pos:0" in param.get(wrapper_name):
                        # quantize_pos
                        quantize_pos = param.get(wrapper_name).get("output_0_pos:0")
                        assert quantize_pos is not None
                        xnode.quant_out["quantize_pos"] = int(quantize_pos)

                        # bit_width
                        quantize_config = config.get("quantize_config")
                        qc_config = quantize_config.get("config")
                        bit_width = (
                            qc_config.get("output_quantizers")[0]
                            .get("quantizer_params")
                            .get("bit_width")
                        )
                        xnode.quant_out["bit_width"] = int(bit_width)
                        logger.debug(f"property: quant_out: {xnode.quant_out}")

            elif op_type == 'CustomOpWrapper':
                bottom = []
                if keras_api == "functional":
                    if len(np.array(layer.get("inbound_nodes")).shape) == 2:
                        for x in layer.get("inbound_nodes"):
                            bottom += [y for y in x if isinstance(y, str)]
                    elif len(np.array(layer.get("inbound_nodes")).shape) == 3:
                        for x in layer.get("inbound_nodes")[0]:
                            bottom += [y for y in x if isinstance(y, str)]
                        if bottom[0] in post_act_map:
                            bottom[0] = post_act_map.get(bottom[0])

                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                xnode = XModelNodeUnknown(wrapper_name)
                logger.debug(f"create XModelNodeUnknown object: name: {xnode.op_name}")

                xnode.tmp_params["data_type"] = config['dtype']
                xnode.tmp_params["trainable"] = config['trainable']
                xnode.tmp_params["shape"] = config['shape']
                xnode.tmp_params["name"] = config['layer']['config']['name']

                for k, v in config['layer']['config'].items():
                    if k not in ['dtype', 'trainable', 'name']:
                        xnode.tmp_params[k] = v

                if param:
                    if wrapper_name in param and "output_0_pos:0" in param.get(wrapper_name):
                        # quantize_pos
                        quantize_pos = param.get(wrapper_name).get("output_0_pos:0")
                        assert quantize_pos is not None
                        xnode.quant_out["quantize_pos"] = int(quantize_pos)

                        # bit_width
                        quantize_config = config.get('layer').get('config').get('quantize_config')
                        qc_config = quantize_config.get("config")
                        bit_width = (
                            qc_config.get("output_quantizers")[0]
                            .get("quantizer_params")
                            .get("bit_width")
                        )
                        xnode.quant_out["bit_width"] = int(bit_width)
                        logger.debug(f"property: quant_out: {xnode.quant_out}")

                        xnode.tmp_params["kind"] = config['layer']['config']['layer']['class_name']

                    param_name = list(param)[0]
                    # weights
                    weights = param[param_name]

                    assert (
                        weights is not None
                    ), f"[ERROR] Not found weights tensor. Op type: {op_type}, name: {wrapper_name}."

                    for k, v in weights.items():
                        if isinstance(v, XTensor):
                            const_xnode_name = param_name + '/' + k
                            # step 1: create XModelNodeConst object
                            const_xnode = XModelNodeConst(const_xnode_name)
                            const_xnode.init_layout = const_xnode.layout = xmodel.layout
                            const_xnode.tensor = v
                            const_xnode.tensor.data_format = DataFormat[xmodel.layout]
                            # update xmodel with const_node
                            xmodel.add_xnode(const_xnode)
                            wrapper_to_xnode[const_xnode.op_name] = const_xnode
                            bottom.append(const_xnode.op_name)

                if 'kind' not in xnode.tmp_params:
                    xnode.tmp_params["kind"] = config['layer']['class_name']

            elif op_type == 'Permute':
                bottom = []
                if keras_api == "functional":
                    for x in layer.get("inbound_nodes")[0]:
                        bottom += [y for y in x if isinstance(y, str)]
                    assert (
                        len(bottom) == 1
                    ), f"[ERROR] keras {op_type} requires one input: actual:{len(bottom)}"
                    if bottom[0] in post_act_map:
                        bottom[0] = post_act_map.get(bottom[0])
                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                # create XModelNodePermute object
                xnode = XModelNodePermute(op_name)
                logger.debug(f"create XModelNodePermute object: name: {xnode.op_name}")

                if layout.name == 'NHWC':
                    xnode.order = [0] + config['dims']

            else:
                bottom = []
                if keras_api == "functional":
                    if len(np.array(layer.get("inbound_nodes")).shape) == 2:
                        for x in layer.get("inbound_nodes"):
                            bottom += [y for y in x if isinstance(y, str)]
                    elif len(np.array(layer.get("inbound_nodes")).shape) == 3:
                        for x in layer.get("inbound_nodes")[0]:
                            bottom += [y for y in x if isinstance(y, str)]
                        if bottom[0] in post_act_map:
                            bottom[0] = post_act_map.get(bottom[0])

                else:
                    parent = layers[i - 1][0]
                    bottom.append(parent.get("config").get("name"))

                xnode = XModelNodeUnknown(wrapper_name)
                logger.debug(f"create XModelNodeUnknown object: name: {xnode.op_name}")

                for k, v in config.items():
                    if v == None:
                        continue
                    if type(v) == dict:
                        v = str(v)
                    if k == 'dtype':
                        k = 'data_type'

                    if not isinstance(v, (list, bool, int, float, str, bytes, map)):
                        assert isinstance(v, (list, bool, int, float, str, bytes, map)), f"Unsupported data type!"

                    xnode.tmp_params[k] = v

                xnode.tmp_params["name"] = layer['name']
                xnode.tmp_params["kind"] = layer['class_name']

                if param:
                    if wrapper_name in param and "output_0_pos:0" in param.get(wrapper_name):
                        # quantize_pos
                        quantize_pos = param.get(wrapper_name).get("output_0_pos:0")
                        assert quantize_pos is not None
                        xnode.quant_out["quantize_pos"] = int(quantize_pos)

                        # bit_width
                        quantize_config = config.get("quantize_config")
                        qc_config = quantize_config.get("config")
                        bit_width = (
                            qc_config.get("output_quantizers")[0]
                            .get("quantizer_params")
                            .get("bit_width")
                        )
                        xnode.quant_out["bit_width"] = int(bit_width)
                        logger.debug(f"property: quant_out: {xnode.quant_out}")


                    param_name = list(param)[0]
                    # weights
                    weights = param[param_name]

                    assert (
                        weights is not None
                    ), f"[ERROR] Not found weights tensor. Op type: {op_type}, name: {wrapper_name}."

                    for k, v in weights.items():
                        if isinstance(v, XTensor):
                            const_xnode_name = param_name + '/' + k
                            # step 1: create XModelNodeConst object
                            const_xnode = XModelNodeConst(const_xnode_name)
                            const_xnode.init_layout = const_xnode.layout = xmodel.layout
                            const_xnode.tensor = v
                            const_xnode.tensor.data_format = DataFormat[xmodel.layout]
                            # update xmodel with const_node
                            xmodel.add_xnode(const_xnode)
                            wrapper_to_xnode[const_xnode.op_name] = const_xnode
                            bottom.append(const_xnode.op_name)

            if set(['shape', 'dtype']) <= set(list(layer['config'])):
                xnode.tmp_params['shape'] = config['shape']
                xnode.tmp_params['data_type'] = config['dtype']

            # set layout
            xnode.init_layout = xnode.layout = xmodel.layout
            logger.debug(
                f"property: init layout: {xnode.init_layout}, current layout: {xnode.layout}"
            )
            # set bottom
            if xnode.op_type != "input":
                xnode.bottom = bottom  # layer.input
            # update xmodel
            xmodel.add_xnode(xnode)
            wrapper_to_xnode[wrapper_name] = xnode

        logger.info("* end: translate tensorflow nodes to xmodel nodes")

        # update top
        for xnode in xmodel.xnodes:
            if xnode.bottom is not None and len(xnode.bottom) > 0:
                for i in range(len(xnode.bottom)):
                    pname = xnode.bottom[i]
                    if pname in post_act_map and xnode.op_name != post_act_map.get(
                        pname
                    ):
                        xnode.bottom[i] = post_act_map.get(pname)
                        pname = xnode.bottom[i]
                    pnode = wrapper_to_xnode.get(pname)
                    assert (
                        pnode is not None
                    ), f"[ERROR] Not found parent node: name: {pname}. Current node name: {xnode.op_name}"
                    pnode.top.append(xnode.op_name)

        xmodel.topsort()

        # extract fixneuron
        xnodes = [x for x in xmodel.xnodes]
        for xnode in xnodes:
            if xnode.op_type != "fixneuron":
                quantize_pos_in = xnode.quant_in.get("quantize_pos")
                if quantize_pos_in is not None:
                    xnode_fix = XModelNodeFixNeuron(xnode.op_name + "_fix")
                    xnode_fix.quant_in["bit_width"] = xnode_fix.quant_out[
                        "bit_width"
                    ] = xnode.quant_in["bit_width"]
                    xnode_fix.quant_in["quantize_pos"] = xnode_fix.quant_out[
                        "quantize_pos"
                    ] = xnode.quant_in["quantize_pos"]
                    xnode_fix.top = [xnode.op_name]
                    xnode_fix.bottom = xnode.bottom
                    # udpate parent node
                    if xnode.bottom is not None and len(xnode.bottom) > 0:
                        for pname in xnode.bottom:
                            pnode = xmodel.get_xnode_by_name(pname)
                            assert (
                                pnode is not None
                            ), f"[ERROR] Not found parent node: {pname}. The current node is {xnode.op_name}."
                            idx = pnode.top.index(xnode.op_name)
                            pnode.top[idx] = xnode_fix.op_name
                    # update xnode
                    xnode.bottom = [xnode_fix.op_name]
                    xmodel.add_xnode(xnode_fix)

                quantize_pos_out = xnode.quant_out.get("quantize_pos")
                if quantize_pos_out is not None:
                    xnode_fix = XModelNodeFixNeuron(xnode.op_name + "_fix")
                    xnode_fix.quant_in["bit_width"] = xnode_fix.quant_out[
                        "bit_width"
                    ] = xnode.quant_out["bit_width"]
                    xnode_fix.quant_in["quantize_pos"] = xnode_fix.quant_out[
                        "quantize_pos"
                    ] = xnode.quant_out["quantize_pos"]
                    xnode_fix.bottom = [xnode.op_name]
                    xnode_fix.top = xnode.top
                    # update child node
                    if xnode.top is not None and len(xnode.top) > 0:
                        for cname in xnode.top:
                            cnode = xmodel.get_xnode_by_name(cname)
                            assert (
                                cnode is not None
                            ), f"[ERROR] Not found child node: {cname}. The current node is {xnode.op_name}."
                            idx = cnode.bottom.index(xnode.op_name)
                            cnode.bottom[idx] = xnode_fix.op_name
                    # update xnode
                    xnode.top = [xnode_fix.op_name]
                    xmodel.add_xnode(xnode_fix)

        # * Special Pass: set round_mode property of xmodel
        cls.__set_fixneuron_round_mode(xmodel)

        # * Special Pass: remove activation linear
        cls.__remove_activation_linear(xmodel)

        if not xmodel.infer_shape(Layout.NHWC):
            print(f"[ERROR] Failed to infer xmodel with the {xmodel.layout} layout")
            sys.exit(1)

        # * perform platform-specific optimizations
        OptManager.dispatch(xmodel, "xnnc")

        return xmodel

    @classmethod
    def __load_from_pb(
        cls, model_files: List[Path]
    ) -> Tuple[str, List["node_def_pb2.NodeDef"]]:
        """
        Load raw model files from a list of specified file paths.

        Parameters:
            model_files: a list of specified file paths, which should specify the paths to TensorFlow model files.

        Returns:
            str: model name
            dict: key is layer/node name, value is a layer dict.
        """
        # check model files
        if len(model_files) != 1:
            logger.error(
                "The 'model_files' argument should contain only one '.pb' file."
            )
            sys.exit(1)

        # load model architecture file
        tfmodel: Path = None
        if model_files[0].suffix == ".pb":
            tfmodel = model_files[0]
        if tfmodel is None:
            logger.info("Not found '.pb' file.")
            sys.exit(1)

        # check tfmodel
        passed, err_msg, tfmodel = helper.check_filepath(tfmodel, extension=".pb")
        if not passed:
            logger.error(err_msg)
            sys.exit(1)

        # load model file
        try:
          from tensorflow.core.framework import graph_pb2
        except:
          from xnnc.proto.tf_pb2 import graph_pb2
        graph_def = graph_pb2.GraphDef()
        with open(tfmodel, "rb") as pf:
            graph_def.ParseFromString(pf.read())

        return tfmodel.stem, list(graph_def.node)

    @classmethod
    def __load_from_hdf5(cls, model_files: List[Path]) -> Tuple[str, List[Any]]:
        """
        Load raw model files from a list of specified file paths.

        Parameters:
            model_files: a list of specified file paths, which should specify the paths to TensorFlow model files.

        Returns:
            str: model name
            dict: key is layer/node name, value is a layer dict.
        """
        if h5py is None:
            raise ImportError(
                "[ERROR] xnnc requires 'h5py' package to parse tensorflow frozen model of HDF5 format."
            )

        # check model files
        if len(model_files) != 1:
            logger.error(
                "The 'model_files' argument should contain only one '.h5' file."
            )
            sys.exit(1)

        # load model architecture file
        tfmodel: Path = None
        if model_files[0].suffix == ".h5":
            tfmodel = model_files[0]
        if tfmodel is None:
            logger.info("Not found '.h5' file.")
            sys.exit(1)

        # check tfmodel
        passed, err_msg, tfmodel = helper.check_filepath(tfmodel, extension=".h5")
        if not passed:
            logger.error(err_msg)
            sys.exit(1)

        # load model file
        with h5py.File(tfmodel, "r") as h5f:
            # keras version
            print(f"[INFO] keras version: {h5f.attrs.get('keras_version')}")

            model_config = h5f.attrs.get("model_config")
            assert model_config is not None, "[ERROR] No model found in config file."

            if isinstance(model_config, bytes):
                model_config = model_config.decode("utf-8")
            model_config = json.loads(model_config)

            # keras api type
            keras_api_type = model_config.get("class_name").lower()
            print(f"[INFO] Tensorflow Keras model type: {keras_api_type}")

            # model name
            model_name = model_config.get("config").get("name")

            # layers
            layers = model_config.get("config").get("layers")
            # input_layers = model_config.get("config").get("input_layers")
            # output_layers = model_config.get("config").get("outputs")

            # model weights
            assert (
                "model_weights" in h5f
            ), "[ERROR] Not found 'model_weights' group in the current HDF5 model file."
            model_weights = h5f.get("model_weights")

            def backtrack(group, names, model_param_dict):
                if group is None or len(group) == 0:
                    return

                if "weight_names" in group.attrs:
                    weight_names = [
                        x if isinstance(x, str) else x.decode("utf-8")
                        for x in group.attrs.get("weight_names").tolist()
                    ]
                    param_dict = {}
                    for wname in weight_names:
                        assert wname not in param_dict
                        segments = wname.split("/")
                        assert len(segments) >= 2
                        ds = segments.pop()
                        gname = "/".join(segments)
                        value = group
                        for seg in segments:
                            value = value.get(seg)
                            assert value is not None, f"{segments}"
                        value = value.get(ds)[()]

                        if isinstance(value, np.ndarray):
                            if value.dtype == np.float32:
                                # solve the type issue caused by numpy type promotion
                                value = value.astype(np.float32)
                            value = XTensor(value)

                        if gname not in param_dict:
                            param_dict[gname] = {ds: value}
                        else:
                            param_dict[gname][ds] = value
                    model_param_dict["/".join(names)] = param_dict
                    return

                for name, g in group.items():
                    backtrack(g, names + [name], model_param_dict)

            model_param_dict = {}
            for wrapper_name, param in model_weights.items():
                assert wrapper_name not in model_param_dict
                if len(param.keys()) > 0:
                    names = [wrapper_name]
                    backtrack(param, names, model_param_dict)

            layer_param_list = []
            for layer in layers:
                name = (
                    layer.get("name")
                    if keras_api_type == "functional"
                    else layer.get("config").get("name")
                )
                param = model_param_dict.get(name)

                # Add '_output_shapes'
                if 'optimizer_weights' in h5f.keys() and name+':0' in h5f['optimizer_weights'].keys():
                    layer['config']['shape'] = h5f['optimizer_weights'][name+':0'].__array__().tolist()

                layer_param_list.append((layer, param))
        return f"{model_name}+{keras_api_type}", layer_param_list

    @classmethod
    def __generate_xmodel(
        cls,
        name: str,
        layout: Layout,
        layers: List["node_def_pb2.NodeDef"],
        const_layer_dict,
        super_const_dict,
        in_shapes: Optional[Union[List[List[int]], Dict[str, List[int]]]] = None,
        batchsize: int = 1,
    ) -> XModel:

        logger.info("* start: translate tensorflow layers to xmodel nodes")

        # create an xmodel
        xmodel = XModel(name, "tensorflow", layout=str(layout).split(".")[-1])
        logger.info(
            f"create XModel object: name: {xmodel.name}, type: {xmodel.origin}, layout: {xmodel.layout}"
        )

        # translate tensorflow ir into xnnc ir
        pbar = tqdm(
            layers,
            desc="[INFO] parse raw model",
            bar_format="{desc:27}:{percentage:3.0f}%|{bar}{r_bar:50}",
        )
        for layer in pbar:
            # get op name and type
            op_name: str = layer.name
            op_type: str = layer.op.lower()

            if op_name == "label_batch":
                continue

            # input layer names
            bottom: List[str] = [x for x in layer.input]

            logger.debug(f"*** source layer info: type: {op_type}, name: {op_name}")

            # translate current layer to XModel node object
            node: XModelNode = None

            if op_type == "placeholder":
                # create XModelNodeInput object
                node = XModelNodeInput(op_name)
                logger.debug(f"create XModelNodeInput object: name: {node.op_name}")

                # get shape info
                shape = None
                if in_shapes is not None and len(in_shapes) > 0:
                    if in_shapes.__class__.__name__ == "list":
                        shape = in_shapes[0]
                    elif in_shapes.__class__.__name__ == "dict":
                        shape = in_shapes.get(op_name)

                if shape is None:
                  # shape = cls.__get_shape(layer.attr["shape"], layout)
                    if layer.attr["_output_shapes"].HasField("list"):
                        shape = cls.__get_output_shapes(layer.attr["_output_shapes"], layout)
                    else:
                        shape = cls.__get_shape(layer.attr["shape"], layout)
                    assert (
                        shape is not None
                    ), f"[ERROR] Failed to extract the input shape of the input layer: {op_name}. Please check the model or provide input shape via command line option."
                    shape[0] = batchsize

                assert all(
                    [isinstance(x, int) and x > 0 for x in shape]
                ), f"[ERROR] Invalid shape of input layer: shape: {shape} (N,H,W,C), name: {op_name}"

                # shape of input featuremap
                node.shape = shape
                logger.debug(f"property: shape: {node.shape}")

            elif op_type == "conv2d":
                # get SuperConst layer
                super_const = None
                assert (
                    bottom is not None and len(bottom) == 2
                ), f"[ERROR] TF Conv2d requires two inputs: actual: {bottom}."
                _, weights_id = bottom
                super_const = super_const_dict.get(weights_id)
                assert (
                    super_const is not None
                ), f"[ERROR] Not found weights for current Conv2d op (name: {op_name})."
                bottom.remove(weights_id)

                # get weights: (h, w, ic, oc)
                weights = super_const.get("tensor")

                # get kernel size: (kernel_h, kernel_w)
                ksize = list(weights.shape)[:2]

                # create XModelNodeConv2d object
                node = XModelNodeConv2d(op_name, ksize)
                logger.debug(f"create XModelNodeConv2d object: name: {node.op_name}")
                logger.debug(f"property: (kernel_h, kernel_w): {node.kernel_size}")

                node.group = 1
                logger.debug(f"property: group: {node.group}")
                node.round_mode = RoundMode.CEIL
                logger.debug(f"property: round_mode: {node.round_mode}")

                # set weights
                node.weights = weights
                logger.debug(
                    f"property: weights: shape (height, width, in_channels, out_channels): {node.weights.shape}, dtype: {node.weights.dtype.name}"
                )

                # set quantization info
                node.quant_weights["bit_width"] = super_const.get("bit_width")
                node.quant_weights["quantize_pos"] = super_const.get("quantize_pos")
                node.quant_weights["round_mode"] = 0  # "STD_ROUND"
                logger.debug(f"property: quant_weights: {node.quant_weights}")

                # set strides, dilations, padding
                for key, value in layer.attr.items():
                    lkey = key.lower()
                    if lkey == "strides":
                        strides = cls.__get_strides(value, layout)
                        # [stride_h, stride_w]
                        node.strides = strides
                        logger.debug(f"property: (stride_h, stride_w): {node.strides}")

                    elif lkey == "dilations":
                        dilation = cls.__get_dilations(value, layout)
                        # [dilation_n, dilation_c, dilation_h, dilation_w]
                        node.dilation = dilation
                        logger.debug(
                            f"property: (dilation_n, dilation_c, dilation_h, dilation_w): {node.dilation}"
                        )

                    elif lkey == "padding":
                        pad_mode = cls.__get_padding(value).upper()
                        node.pad_mode = PadMode[pad_mode]
                        logger.debug(f"property: pad_mode: {node.pad_mode}")

            elif op_type == "relu":
                # create XModelNodeRelu object
                node = XModelNodeRelu(op_name)
                logger.debug(f"create XModelNodeRelu object: name: {node.op_name}")

                # set negative_slope
                node.alpha = 0
                logger.debug(f"property: alpha: {node.alpha}")

            elif op_type == "leakyrelu":
                # create XModelNodeRelu object
                node = XModelNodeRelu(op_name)
                logger.debug(
                    f"create XModelNodeRelu (leaky) object: name: {node.op_name}"
                )

                # set negative_slope
                node.alpha = 0.2
                alpha = layer.attr.get("alpha")
                if alpha is not None:
                    node.alpha = alpha.f
                logger.debug(f"property: alpha: {node.alpha}")

            elif op_type in ["pad", "mirrorpad"]:
                assert (
                    len(bottom) == 2
                ), f"[ERROR] tf pad op requires two inputs: actual: {len(bottom)}"
                _, padding_id = bottom
                if padding_id in const_layer_dict:
                    pad_const = const_layer_dict.get(padding_id)
                    bottom.remove(padding_id)
                    # get padding: layout is NHWC
                    padding = tc.ravel(
                        cls.__get_tensor(pad_const.attr["value"])
                    ).tolist()

                else:
                    padding = [None] * 8

                # pad_mode
                pad_mode = None
                if "mode" in layer.attr:
                    pad_mode = layer.attr.get("mode").s.decode("utf-8").lower()
                else:
                    pad_mode = "constant"
                assert pad_mode in ["constant", "reflect", "symmetric", "edge"]

                # get constant_values
                constant_values = None
                if pad_mode == "constant":
                    # TODO samshin: hard code
                    constant_values = [0.0] * len(padding)
                # assert constant_values is not None

                # create XModelNodePad object
                node = XModelNodePad(op_name, padding, pad_mode, constant_values)
                logger.debug(f"create XModelNodePad object: name: {node.op_name}")
                logger.debug(
                    f"property: padding (top, bottom, left, right): {node.padding}"
                )
                logger.debug(f"property: pad_mode: {node.pad_mode}")
                if pad_mode == "constant":
                    logger.debug(f"property: constant_values: {node.constant_values}")

            elif op_type in ["maxpool", "avgpool"]:
                # get kernel size
                ksize = cls.__get_ksize(layer.attr["ksize"], layout)
                # create pooling object
                node = None
                if op_type == "maxpool":
                    node = XModelNodeMaxPool(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeMaxPool object: name: {node.op_name}"
                    )
                else:
                    node = XModelNodeAvgPool(op_name, ksize)
                    logger.debug(
                        f"create XModelNodeAvgPool object: name: {node.op_name}"
                    )
                    # count_include_pad
                    node.count_include_pad = False
                    logger.debug(
                        f"property: count_inclue_pad: {node.count_include_pad}"
                    )
                logger.debug(f"property: (kernel_h, kernel_w): {node.kernel_size}")

                # set round_mode
                node.round_mode = RoundMode.CEIL
                logger.debug(f"property: round_mode: {node.round_mode}")

                for key, value in layer.attr.items():
                    lkey = key.lower()
                    if lkey == "strides":
                        strides = cls.__get_strides(value, layout)
                        # [stride_h, stride_w]
                        node.strides = strides
                        logger.debug(f"property: (stride_h, stride_w): {node.strides}")
                    elif lkey == "padding":
                        pad_mode = cls.__get_padding(value).upper()
                        node.pad_mode = PadMode[pad_mode]
                        logger.debug(f"property: pad_mode: {node.pad_mode}")

            elif op_type in ["add", "addv2"]:
                # create XModelNodeElemAdd object
                node = XModelNodeElemAdd(op_name)
                logger.debug(f"create XModelNodeElemAdd object: name: {node.op_name}")

                # set coefficient
                node.alpha = [1.0] * len(layer.input)
                logger.debug(f"property: alpha: {node.alpha}")

            elif op_type in ["addn"]:
                # create XModelNodeElemAddn object
                node = XModelNodeElemAddn(op_name)
                logger.debug(f"create XModelNodeElemAddn object: name: {node.op_name}")

            elif op_type == "mean":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] tensorflow Mean op requires two inputs: actual: {len(bottom)}."

                # get reduction axes
                axis_id = bottom.pop()
                assert axis_id in const_layer_dict, f"[ERROR] Not found 'axis' info."
                const_layer = const_layer_dict.get(axis_id)
                axis: XTensor = cls.__get_tensor(const_layer.attr["value"])

                # create XModelNodeMean object
                node = XModelNodeMean(op_name)
                logger.debug(f"create XModelNodeMean object: name: {node.op_name}")

                # get attribute info
                attr = layer.attr
                # set keep_dims
                keepdims = attr["keep_dims"].b
                node.keep_dims = keepdims if keepdims is not None else False
                logger.debug(f"property: keep_dims: {node.keep_dims}")

                # set axis
                node.axis = axis.tolist() if axis is not None else None
                if node.axis is not None:
                    for i in range(len(node.axis)):
                        # preprocess negative value
                        if node.axis[i] < 0:
                            node.axis[i] += 4
                    node.axis.sort()
                logger.debug(f"property: axis: {node.axis}")

            elif op_type == "squeeze":
                # create XModelNodeSqueeze object
                node = XModelNodeSqueeze(op_name)
                logger.debug(f"create XModelNodeSqueeze object: name: {node.op_name}")

                # set other fields
                for key, value in layer.attr.items():
                    lkey = key.lower()
                    if lkey == "squeeze_dims":
                        node.axis = list(value.list.i)
                        logger.debug(f"property: axis: {node.axis}")
                        break

            elif op_type == "reshape":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF Reshape op requires two inputs: actual: {len(bottom)}. Reshape op name: {op_name}"

                # create XModelNodeReshape object
                node = XModelNodeReshape(op_name)
                logger.debug(f"create XModelNodeReshape object: name: {node.op_name}")

            elif op_type == "softmax":
                # create XModelNodeSoftmax object
                node = XModelNodeSoftmax(op_name)
                logger.debug(f"create XModelNodeSoftmax object: name: {node.op_name}")

                # -1 means the last axis
                axis = -1
                if hasattr(layer.attr, "softmax_dim"):
                    raise NotImplementedError()
                node.axis = axis
                logger.debug(f"property: axis: {node.axis}")

            elif op_type == "concatv2":
                # get axis
                assert (
                    bottom is not None and len(bottom) >= 3
                ), f"[ERROR] TF ConcatV2 requires at least three inputs: actual: {bottom}."
                concat_axis_id = bottom.pop()
                const_layer = const_layer_dict.get(concat_axis_id)
                assert (
                    const_layer is not None
                ), f"[ERROR] Not found axis for concat op (name: {op_name})."
                axis = const_layer.attr["value"].tensor.int_val[0]

                # create XModelNodeConcat object
                node = XModelNodeConcat(op_name)
                logger.debug(f"create XModelNodeConcat object: name: {node.op_name}")

                # set axis
                node.axis = axis
                logger.debug(f"property: axis: {node.axis}")

            elif op_type == "relu6":
                # create XModelNodeRelu6 object
                node = XModelNodeRelu6(op_name)
                logger.debug(f"create XModelNodeRelu6 object: name: {node.op_name}")

            elif op_type == "depthwiseconv2dnative":
                # get SuperConst layer
                super_const = None
                assert (
                    bottom is not None and len(bottom) == 2
                ), f"[ERROR] TF Conv2dDepthwise requires two inputs: actual: {bottom}."
                _, weights_id = bottom
                super_const = super_const_dict.get(weights_id)
                assert (
                    super_const is not None
                ), f"[ERROR] Not found weights for current Conv2d op (name: {op_name})."
                bottom.remove(weights_id)

                # get weights: (h,w,ic,cm)
                weights = super_const.get("tensor")
                KH, KW, IC, CM = weights.shape
                # get group: in_channels
                group = IC
                # get num of output: in_channels * channel_multiplier(also, oc)
                num_output = IC * CM
                # get kernel size
                ksize = [KH, KW]

                # create XModelNodeConv2dDepthwise object
                node = XModelNodeConv2dDepthwise(op_name, ksize)
                logger.debug(
                    f"create XModelNodeConv2dDepthwise object: name: {node.op_name}"
                )
                logger.debug(f"property: (kernel_h, kernel_w): {node.kernel_size}")

                node.group = group
                logger.debug(f"property: group: {node.group}")
                node.num_output = num_output
                logger.debug(f"property: num_output: {node.num_output}")
                node.round_mode = RoundMode.CEIL
                logger.debug(f"property: round_mode: {node.round_mode}")

                # set weights
                node.weights = weights
                logger.debug(
                    f"property: weights: shape (height, width, in_channels, out_channels): {node.weights.shape}, dtype: {node.weights.dtype}"
                )
                # set quantization info
                node.quant_weights["bit_width"] = super_const.get("bit_width")
                node.quant_weights["quantize_pos"] = super_const.get("quantize_pos")
                node.quant_weights["round_mode"] = 0  # "STD_ROUND"
                logger.debug(f"property: quant_weights: {node.quant_weights}")

                # set strides, dilations, padding
                for key, value in layer.attr.items():
                    lkey = key.lower()
                    if lkey == "strides":
                        strides = cls.__get_strides(value, layout)
                        # [stride_h, stride_w]
                        node.strides = strides
                        logger.debug(f"property: (stride_h, stride_w): {node.strides}")
                    elif lkey == "dilations":
                        dilation = cls.__get_dilations(value, layout)
                        # [dilation_n, dilation_c, dilation_h, dilation_w]
                        node.dilation = dilation
                        logger.debug(
                            f"property: (dilation_n, dilation_c, dilation_h, dilation_w): {node.dilation}"
                        )
                    elif lkey == "padding":
                        pad_mode = cls.__get_padding(value).upper()
                        node.pad_mode = PadMode[pad_mode]
                        logger.debug(f"property: pad_mode: {node.pad_mode}")

            elif op_type == "fixneuron":

                # create XModelNodeFixNeuron object
                node = XModelNodeFixNeuron(op_name)
                logger.debug(f"create XModelNodeFixNeuron object: name: {node.op_name}")
                # bit width
                node.quant_in["bit_width"] = layer.attr["bit_width"].i
                node.quant_out["bit_width"] = layer.attr["bit_width"].i
                # fix info
                node.quant_in["quantize_pos"] = layer.attr["quantize_pos"].i
                node.quant_out["quantize_pos"] = layer.attr["quantize_pos"].i
                logger.debug(
                    f"property: quant_in: bit_width: {node.quant_in['bit_width']}, quantize_pos: {node.quant_in['quantize_pos']}"
                )
                logger.debug(
                    f"property: quant_out: bit_width: {node.quant_out['bit_width']}, quantize_pos: {node.quant_out['quantize_pos']}"
                )

            elif op_type in ["resizenearestneighbor", "resizebilinear"]:
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF resize should have two inputs: name: {op_name}."

                # mode
                mode = "nearest" if op_type == "resizenearestneighbor" else "bilinear"

                # create XModelNodeResize object
                node = XModelNodeResize(op_name, mode=mode)
                logger.debug(
                    f"create XModelNodeResize object: name: {node.op_name}, mode: {node.mode}"
                )

                # align_corners
                node.align_corners = layer.attr["align_corners"].b
                # half_pixel_centers
                node.half_pixel_centers = layer.attr["half_pixel_centers"].b

            elif op_type == "shape":
                # create XModelNodeShape object
                node = XModelNodeShape(op_name)
                logger.debug(f"create XModelNodeShape: name: {node.op_name}")

                # output dtype
                tf_dtype = DT_NAME[layer.attr["out_type"].type]
                np_dtype = TF_TO_NP[tf_dtype]
                node.out_type = np_dtype.__name__

            elif op_type == "neg":
                # create XModelNodeElemNegative object
                node = XModelNodeElemNegative(op_name)
                logger.debug(f"create XModelNodeElemNegative: name: {node.op_name}")

            elif op_type == "mul":
                # create XModelNodeElemMul object
                node = XModelNodeElemMul(op_name)
                logger.debug(f"create XModelNodeElemMul object: name: {node.op_name}")

            elif op_type == "sub":
                # create XModelNodeElemSub object
                node = XModelNodeElemSub(op_name)
                logger.debug(f"create XModelNodeElemSub object: name: {node.op_name}")

            elif op_type == "stridedslice":
                # set begin
                assert bottom[1] in const_layer_dict
                begin: List[int] = cls.__get_tensor(
                    const_layer_dict.get(bottom[1]).attr["value"]
                ).tolist()
                # set end
                assert bottom[2] in const_layer_dict
                end: List[int] = cls.__get_tensor(
                    const_layer_dict.get(bottom[2]).attr["value"]
                ).tolist()
                # set strides
                assert bottom[3] in const_layer_dict
                strides: List[int] = cls.__get_tensor(
                    const_layer_dict.get(bottom[3]).attr["value"]
                ).tolist()
                # only keep the first element as input
                bottom = bottom[:1]

                # create XModelNodeStridedSlice object
                node = XModelNodeStridedSlice(
                    op_name, begin=begin, end=end, strides=strides
                )
                logger.debug(
                    f"create XModelNodeStridedSlice object: name: {node.op_name}"
                )

                # set attributes
                if "begin_mask" in layer.attr:
                    node.begin_mask = layer.attr["begin_mask"].i
                if "end_mask" in layer.attr:
                    node.end_mask = layer.attr["end_mask"].i
                if "ellipsis_mask" in layer.attr:
                    node.ellipsis_mask = layer.attr["ellipsis_mask"].i
                if "new_axis_mask" in layer.attr:
                    node.new_axis_mask = layer.attr["new_axis_mask"].i
                if "shrink_axis_mask" in layer.attr:
                    node.shrink_axis_mask = layer.attr["shrink_axis_mask"].i

            elif op_type == "spacetobatchnd":
                # set paddings
                const_layer = const_layer_dict.get(bottom.pop())
                assert (
                    const_layer is not None
                ), "Not found const layer for 'paddings' property."
                pad_data = cls.__get_tensor(const_layer.attr["value"]).tolist()
                # paddings order: (top, bottom, left, right)
                paddings: List[int] = pad_data[0] + pad_data[1]
                # set block_shape
                const_layer = const_layer_dict.get(bottom.pop())
                assert (
                    const_layer is not None
                ), "Not found const layer for 'block_shape' property."
                block_shape: List[int] = cls.__get_tensor(
                    const_layer.attr["value"]
                ).tolist()

                # create XModelNodeSpaceToBatchND object
                node = XModelNodeSpaceToBatchND(op_name, block_shape, paddings)
                logger.debug(
                    f"create XModelNodeSpaceToBatchND object: name: {node.op_name}"
                )
                logger.debug(f"property: block_shape: {node.block_shape}")
                logger.debug(
                    f"property: paddings (top, bottom, left, right): {node.paddings}"
                )

            elif op_type == "batchtospacend":
                # set crops
                const_layer = const_layer_dict.get(bottom.pop())
                assert (
                    const_layer is not None
                ), "Not found const layer for 'paddings' property."
                crop_data = cls.__get_tensor(const_layer.attr["value"]).tolist()
                # crops order: (top, bottom, left, right)
                crops: List[int] = crop_data[0] + crop_data[1]
                # set block_shape
                const_layer = const_layer_dict.get(bottom.pop())
                assert (
                    const_layer is not None
                ), "Not found const layer for 'block_shape' property."
                block_shape: List[int] = cls.__get_tensor(
                    const_layer.attr["value"]
                ).tolist()

                # create XModelNodeBatchToSpaceND object
                node = XModelNodeBatchToSpaceND(op_name, block_shape, crops)
                logger.debug(
                    f"create XModelNodeSpaceToBatchND object: name: {node.op_name}"
                )
                logger.debug(f"property: block_shape: {node.block_shape}")
                logger.debug(
                    f"property: crops (top, bottom, left, right): {node.crops}"
                )

            elif op_type == "pack":
                # create XModelNodeStack object
                node = XModelNodeStack(op_name)
                logger.debug(f"create XModelNodeElemStack object: name: {node.op_name}")
                # set axis
                node.axis = layer.attr["axis"].i
                logger.debug(f"property: axis: {node.axis}")

            elif op_type == "matmul":

                # create XModelNodeMatMul object
                node = XModelNodeMatMul(op_name)
                logger.debug(f"create XModelNodeMatMul object: name: {node.op_name}")
                # set transpose params
                node.transpose_a: bool = layer.attr["transpose_a"].b
                node.transpose_b: bool = layer.attr["transpose_b"].b
                logger.debug(
                    f"property: transpose_a: {node.transpose_a}, transpose_b: {node.transpose_b}"
                )

            elif op_type == "identity":
                # create XModelNodeIdentity object
                node = XModelNodeIdentity(op_name)
                logger.debug(f"create XModelNodeIdentity object: name: {node.op_name}")

            elif op_type == "conv2dbackpropinput":
                assert (
                    len(layer.input) == 3
                ), f"[ERROR] tf Conv2DBackpropInput op requires 3 inputs: actual: {len(layer.input)}. op name: {op_name}"

                output_shape_id, weights_id, input_id = layer.input

                # update bottom
                if output_shape_id in const_layer_dict:
                    # update bottom
                    bottom = [input_id, output_shape_id]

                elif output_shape_id in super_const_dict:
                    raise NotImplementedError(
                        "[ERROR] Not support the quantized output_shape of TF Conv2dBackpropInput op."
                    )

                else:
                    # update bottom
                    bottom = [input_id, output_shape_id]

                # get weights: (h,w,oc,ic)
                assert (
                    weights_id in super_const_dict
                ), f"[ERROR] Not found op in super_const_dict: name: {weights_id}"
                super_const = super_const_dict.get(weights_id)
                weights = super_const.get("tensor")

                # get kernel size
                ksize = weights.shape[:2]

                # create XModelNodeConv2dTranspose object
                node = XModelNodeConv2dTranspose(op_name, ksize)
                logger.debug(f"create XModelNodeConv2d object: name: {node.op_name}")
                logger.debug(f"property: (kernel_h, kernel_w): {node.kernel_size}")

                # set weights: (h,w,oc,ic)
                node.weights = weights
                logger.debug(
                    f"property: weights: shape (height, width, in_channels, out_channels): {node.weights.shape}, dtype: {node.weights.dtype}"
                )
                # set quantization info
                node.quant_weights["bit_width"] = super_const.get("bit_width")
                node.quant_weights["quantize_pos"] = super_const.get("quantize_pos")
                node.quant_weights["round_mode"] = 0  # "STD_ROUND"
                logger.debug(f"property: quant_weights: {node.quant_weights}")

                # set strides, dilations, padding
                for key, value in layer.attr.items():
                    lkey = key.lower()
                    if lkey == "strides":
                        strides = cls.__get_strides(value, layout)
                        # [stride_h, stride_w]
                        node.strides = strides
                        logger.debug(f"property: (stride_h, stride_w): {node.strides}")

                    elif lkey == "dilations":
                        dilation = cls.__get_dilations(value, layout)
                        # [dilation_n, dilation_c, dilation_h, dilation_w]
                        node.dilation = dilation
                        logger.debug(
                            f"property: (dilation_n, dilation_c, dilation_h, dilation_w): {node.dilation}"
                        )

                    elif lkey == "padding":
                        pad_mode = cls.__get_padding(value).upper()
                        assert pad_mode in [
                            "SAME",
                            "VALID",
                        ], f"[ERROR] Unsupported pad mode: {pad_mode}. op type: {op_type}, name: {op_name}"
                        node.pad_mode = PadMode[pad_mode]
                        logger.debug(f"property: pad_mode: {node.pad_mode}")

            elif op_type == "transpose":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF Transpose op requires two inputs: actual: {len(bottom)}. Transpose op name: {op_name}"

                # create XModelNodePermute object
                node = XModelNodePermute(op_name)
                logger.debug(f"create XModelNodePermute object: name: {node.op_name}")

                _, perm_id = bottom

                # permutation
                assert perm_id in const_layer_dict
                const_layer = const_layer_dict.get(perm_id)
                node.order = cls.__get_tensor(const_layer.attr["value"]).tolist()
                bottom.pop()

            elif op_type == "prod":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF prod op requires two inputs: actual: {len(bottom)}. Prod op name: {op_name}"

                # create XModelNodeReduceProd object
                node = XModelNodeReduceProd(op_name)
                logger.debug(
                    f"create XModelNodeReduceProd object: name: {node.op_name}"
                )

                _, axis_id = bottom

                # axis
                assert axis_id in const_layer_dict
                const_layer = const_layer_dict.get(axis_id)
                node.axis = cls.__get_tensor(const_layer.attr["value"]).tolist()
                logger.debug(f"property: axis: {node.axis}")
                bottom.pop()

                # keep_dims
                keepdims = layer.attr["keep_dims"].b
                node.keep_dims = keepdims if keepdims is not None else False
                logger.debug(f"property: keep_dims: {node.keep_dims}")

            elif op_type == "sum":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF sum op requires two inputs: actual: {len(bottom)}. Sum op name: {op_name}"

                # create XModelNodeReduceSum object
                node = XModelNodeReduceSum(op_name)
                logger.debug(f"create XModelNodeReduceSum object: name: {node.op_name}")

                _, axis_id = bottom

                # axis
                assert axis_id in const_layer_dict
                const_layer = const_layer_dict.get(axis_id)
                node.axis = cls.__get_tensor(const_layer.attr["value"]).tolist()
                logger.debug(f"property: axis: {node.axis}")
                bottom.pop()

                # keep_dims
                keepdims = layer.attr["keep_dims"].b
                node.keep_dims = keepdims if keepdims is not None else False
                logger.debug(f"property: keep_dims: {node.keep_dims}")

            elif op_type == "max":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF reduce_max op requires two inputs: actual: {len(bottom)}. Max op name: {op_name}"

                # create XModelNodeReduceMax object
                node = XModelNodeReduceMax(op_name)
                logger.debug(f"create XModelNodeReduceMax object: name: {node.op_name}")

                _, axis_id = bottom

                # axis
                assert axis_id in const_layer_dict
                const_layer = const_layer_dict.get(axis_id)
                node.axis = cls.__get_tensor(const_layer.attr["value"]).tolist()
                logger.debug(f"property: axis: {node.axis}")
                bottom.pop()

                # keep_dims
                keepdims = layer.attr["keep_dims"].b
                node.keep_dims = keepdims if keepdims is not None else False
                logger.debug(f"property: keep_dims: {node.keep_dims}")

            elif op_type == "exp":
                assert (
                    len(bottom) == 1
                ), f"[ERROR] TF exp op requires one input: actual: {len(bottom)}. Exp op name: {op_name}"

                # create XModelNodeElemExp object
                node = XModelNodeElemExp(op_name)
                logger.debug(f"create XModelNodeElemExp object: name: {node.op_name}")

            elif op_type == "realdiv":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF realdiv op requires two inputs: actual: {len(bottom)}. Realdiv op name: {op_name}"

                # create XModelNodeElemRealDiv object
                node = XModelNodeElemRealDiv(op_name)
                logger.debug(
                    f"create XModelNodeElemRealDiv object: name: {node.op_name}"
                )

            elif op_type == "sigmoid":
                assert (
                    len(bottom) == 1
                ), f"[ERROR] TF sigmoid requires one input: actual: {len(bottom)}. Sigmoid op name: {op_name}"

                # create XModelNodeSigmoid object
                node = XModelNodeSigmoid(op_name)
                logger.debug(f"create XModelNodeSigmoid object: name: {node.op_name}")

                if 'hard_sigmoid' in layer.attr:
                    assert(layer.attr['hard_sigmoid'].b in [True, False])
                    node.tmp_params['hard_sigmoid'] = layer.attr['hard_sigmoid'].b

            elif op_type == "square":
                assert (
                    len(bottom) == 1
                ), f"[ERROR] TF square requires one input: actual: {len(bottom)}. Square op name: {op_name}"

                # create XModelNodeElemSquare object
                node = XModelNodeElemSquare(op_name)
                logger.debug(
                    f"create XModelNodeElemSquare object: name: {node.op_name}"
                )

            elif op_type == "rsqrt":
                assert (
                    len(bottom) == 1
                ), f"[ERROR] TF rsqrt requires one input: actual: {len(bottom)}. RSqrt op name: {op_name}"

                # create XModelNodeElemRSqrt object
                node = XModelNodeElemRSqrt(op_name)
                logger.debug(f"create XModelNodeElemRSqrt object: name: {node.op_name}")

            elif op_type == "maximum":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF elemwise maximum requires two inputs: actual: {len(bottom)}. Maximum op name: {op_name}"

                # create XModelNodeElemMax object
                node = XModelNodeElemMax(op_name)
                logger.debug(f"create XModelNodeElemMax object: name: {node.op_name}")

            elif op_type == "depthtospace":
                assert (
                    len(bottom) == 1
                ), f"[ERROR] TF depth_to_space requires a single input: actual: {len(bottom)}. Depth_to_space op name: {op_name}"

                node = XModelNodeDepthToSpace(op_name)
                logger.debug(
                    f"create XModelNodeDepthToSpace object: name: {node.op_name}"
                )

                # block_size
                node.block_size = layer.attr["block_size"].i
                logger.debug(f"property: block_size: {node.block_size}")

            elif op_type == "minimum":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF elemwise minimum requires two inputs: actual: {len(bottom)}. Minimum op name: {op_name}"

                # create XModelNodeElemMin object
                node = XModelNodeElemMin(op_name)
                logger.debug(f"create XModelNodeElemMin object: name: {node.op_name}")

            elif op_type == "round":
                assert (
                    len(bottom) == 1
                ), f"[ERROR] TF elemwise round requires a single input: actual: {len(bottom)}. Round op name: {op_name}"

                # create XModelNodeElemRound object
                node = XModelNodeElemRound(op_name)
                logger.debug(f"create XModelNodeElemRound object: name: {node.op_name}")

                # mode
                node.round_mode = "py3_round"  # round_half_to_even

            elif op_type in ["cast", "bitcast"]:
                assert (
                    len(bottom) == 1
                ), f"[ERROR] TF cast requires a single input: actual: {len(bottom)}. Cast op name: {op_name}"

                # create XModelNodeTypeCast object
                node = XModelNodeTypeCast(op_name)
                logger.debug(f"create XModelNodeTypeCast object: name: {node.op_name}")

                if op_type == "cast":
                    # source dtype
                    src_dtype = DT_NAME.get(layer.attr["SrcT"].type)
                    if src_dtype == 'DT_BFLOAT16':
                        node.tmp_params['src_dtype'] = 'bfloat16'
                    node.src_dtype = TF_TO_NP.get(src_dtype)

                    # destination dtype
                    dst_dtype = DT_NAME.get(layer.attr["DstT"].type)
                    if dst_dtype == 'DT_BFLOAT16':
                        node.tmp_params['dtype'] = 'bfloat16'
                    node.dst_dtype = TF_TO_NP.get(dst_dtype)

                elif op_type == "bitcast":
                    # source dtype
                    node.src_dtype = TF_TO_NP.get(DT_NAME.get(layer.attr["T"].type))

                    # destination dtype
                    node.dst_dtype = TF_TO_NP.get(DT_NAME.get(layer.attr["type"].type))

            elif op_type == "randomstandardnormal":
                assert (
                    len(bottom) == 1
                ), f"[ERROR] TF RandomStandardNormal requires a single input: actual: {len(bottom)}. op name: {op_name}"

                # create XModelNodeRandomStandardNormal object
                node = XModelNodeRandomStandardNormal(op_name)
                logger.debug(
                    f"create XModelNodeRandomStandardNormal object: name: {node.op_name}"
                )

                # dtype
                node.dtype = TF_TO_NP.get(DT_NAME.get(layer.attr.get("dtype").type))

                # seed
                node.seed = layer.attr.get("seed").i

                # seed2
                node.seed2 = layer.attr.get("seed2").i

            elif op_type == "tanh":
                assert (
                    len(bottom) == 1
                ), f"[ERROR] TF {op_type} requires one input: actual: {len(bottom)}. Op name: {op_name}"

                # create XModelNodeElemTanh object
                node = XModelNodeElemTanh(op_name)
                logger.debug(f"create XModelNodeElemTanh object: name: {node.op_name}")

            elif op_type == "argmax":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF {op_type} requires two inputs: actual: {len(bottom)}. Op name: {op_name}"

                # create XModelNodeArgmax object
                node = XModelNodeArgmax(op_name)
                logger.debug(f"create XModelNodeArgmax object: name: {node.op_name}")

                # get axis
                axis_id = bottom.pop()
                assert axis_id in const_layer_dict
                const_layer = const_layer_dict.get(axis_id)
                tensor = cls.__get_tensor(const_layer.attr["value"])
                assert tensor.ndims == 1
                node.axis = tensor.tolist()[0]

                # output_type
                node.output_type = TF_TO_NP.get(
                    DT_NAME.get(layer.attr.get("output_type").type)
                )

            elif op_type == "clipbyvalue":
                assert (
                    len(bottom) == 1
                ), f"[ERROR] TF {op_type} requires one inputs: actual: {len(bottom)}. Op name: {op_name}"

                # create XModelNode object
                node = XModelNode(op_name, op_type="clip_by_value")
                logger.debug(f"create XModelNode object: name: {node.op_name}")

                for key in ["Minimum_y", "Maximum_y"]:
                    node.tmp_params[key] = layer.attr.get(key).f

            elif op_type == "expanddims":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF ExpandDims op requires two inputs: actual: {len(bottom)}. ExpandDims op name: {op_name}"

                # create XModelNodeExpandDims object
                node = XModelNodeExpandDims(op_name)
                logger.debug(f"create XModelNodeExpandDims object: name: {node.op_name}")

                for k, v in layer.attr.items():
                    if k in ['T', 'Tdim'] and v.HasField("type"):
                       val = TF_TO_NP[DT_NAME[v.type]].__name__
                    elif k == '_output_shapes':
                        val = cls.__get_output_shapes(layer.attr[k], layout)
                    node.tmp_params[k] = val
                node.tmp_params["kind"] = layer.op

            elif op_type == "pow":
                assert (
                    len(bottom) == 2
                ), f"[ERROR] TF pow op requires one input: actual: {len(bottom)}. Pow op name: {op_name}"

                # create XModelNodeElemRealPow object
                node = XModelNodeElemRealPow(op_name)
                logger.debug(f"create XModelNodePow object: name: {node.op_name}")

            elif op_type == "floor":
                assert (
                    len(bottom) == 1
                ), f"[ERROR] TF elemwise floor requires a single input: actual: {len(bottom)}. Round op name: {op_name}"

                # create XModelNodeElemFloor object
                node = XModelNodeElemFloor(op_name)
                logger.debug(f"create XModelNodeElemFloor object: name: {node.op_name}")

                # mode
                node.round_mode = "floor"
                node.tmp_params["name"] = op_name
                node.tmp_params["kind"] = layer.op

            elif op_type == "unsortedsegmentsum":
                assert (
                    len(bottom) == 3
                ), f"[ERROR] TF unsortedsegmentsum requires a single input: actual: {len(bottom)}. Round op name: {op_name}"

                # create XModelNodeUnsortedSegmentSum object
                node = XModelNodeUnsortedSegmentSum(op_name)

            elif op_type in ['split']:
                num = layer.attr['num_split'].i
                output_names = []
                output_name_diff_str = [':'+str(i) for i in range(1, num)]
                output_name_diff_str.append('')
                for _ in layers:
                    for input_name in _.input:
                        if input_name.startswith(op_name) and input_name.replace(op_name, '', 1) in output_name_diff_str and input_name not in output_names:
                            output_names.append(input_name)
                output_names.sort()

                input_layer = [_ for _ in layers if _.name == bottom[1]][0]
                if str(input_layer.attr['_output_shapes'])=='':
                    raise ValueError(
                        f"The layer {input_layer.name} must contain the attribute '_output_shapes'."
                    )
                input_shapes = cls.__get_output_shapes(input_layer.attr['_output_shapes'])
                if input_shapes[0] == -1:
                  input_shapes[0] = 1

                ndim = len(input_shapes)
                strides = ndim*[1]
                begin = ndim*[0]
                end = copy.deepcopy(input_shapes)
                axis_node = const_layer_dict.get(bottom[0])
                axis = axis_node.attr['value'].tensor.int_val[0]
                if axis < 0:
                  axis += ndim
                step = input_shapes[axis] / num
                for i in range(len(output_names)):
                  begin[axis] = int(i*step)
                  end[axis] = int((i+1)*step)
                  # create XModelNodeStridedSlice object
                  node = XModelNodeStridedSlice(
                      output_names[i], begin=begin, end=end, strides=strides
                  )
                  logger.debug(
                      f"create XModelNodeStridedSlice object: name: {node.op_name}"
                  )
                  # set layout
                  node.init_layout = node.layout = xmodel.layout
                  logger.debug(
                      f"property: init layout: {node.init_layout}, current layout: {node.layout}"
                  )
                  node.bottom = [bottom[1]]
                  xmodel.add_xnode(node)
                continue

            elif op_type in ['unpack']:
                #Since xir does not support multi-output op, replace multi-output unpack with multiple single-output strided_slice.
                axis = layer.attr["axis"].i
                logger.debug(f"property: axis: {axis}")
                num = layer.attr["num"].i
                logger.debug(f"property: axis: {axis}")

                op_names = []
                diff_str = [':'+str(i) for i in range(1, num)]
                diff_str.append('')
                for _ in layers:
                    for input_name in _.input:
                        if input_name.startswith(op_name) and input_name.replace(op_name, '', 1) in diff_str and input_name not in op_names:
                            op_names.append(input_name)
                op_names.sort()

                input_layer = [_ for _ in layers if _.name == bottom[0]][0]
                assert '_output_shapes' in input_layer.attr
                input_shapes = cls.__get_output_shapes(input_layer.attr['_output_shapes'])
                ndim = len(input_shapes)
                if axis < 0:
                  axis += ndim

                begin = ndim*[0]
                strides = ndim*[1]
                end = copy.deepcopy(input_shapes)

                for i in range(input_shapes[axis]):
                    begin[axis] = i
                    end[axis] = i+1

                    # create XModelNodeStridedSlice object
                    node = XModelNodeStridedSlice(
                        op_names[i]+'_stridedslice', begin=begin, end=end, strides=strides
                    )
                    logger.debug(
                        f"create XModelNodeStridedSlice object: name: {node.op_name}"
                    )

                    # set layout
                    node.init_layout = node.layout = xmodel.layout
                    logger.debug(
                        f"property: init layout: {node.init_layout}, current layout: {node.layout}"
                    )
                    node.bottom = bottom
                    xmodel.add_xnode(node)

                    #Due to xcompiler limitations, strided_slice only supports the parameters "begin" "end" and "strided", and the rest of the parameters are optimized.
                    # create XModelNodeSqueeze object
                    squeeze_node = XModelNodeSqueeze(op_names[i])
                    logger.debug(f"create XModelNodeSqueeze object: name: {squeeze_node.op_name}")

                    squeeze_node.axis = [axis]
                    logger.debug(f"property: axis: {squeeze_node.axis}")

                    squeeze_node.init_layout = squeeze_node.layout = xmodel.layout
                    logger.debug(
                        f"property: init layout: {squeeze_node.init_layout}, current layout: {squeeze_node.layout}"
                    )
                    squeeze_node.bottom = [node.op_name] # layer.input

                    # update xmodel
                    xmodel.add_xnode(squeeze_node)
                continue

            else:
                node = XModelNodeUnknown(op_name)
                logger.debug(f"create XModelNodeUnknown object: name: {node.op_name}")

                for k, v in layer.attr.items():

                    if v.HasField("i"):
                        val = v.i
                    elif v.HasField("f"):
                        val = v.f
                    elif v.HasField("b"):
                        val = v.b
                    elif v.HasField("type"):
                        val = v.type
                    elif v.HasField("s"):
                        val = v.s.decode("ascii")
                    elif v.HasField("shape"):
                        val = v.shape
                    elif v.HasField("list"):
                        if str(v.list) == '':
                            val = ''
                        elif len(v.list.i) != 0:
                            val = eval(str(v.list.i))
                            val = [x for x in val]
                        elif len(v.list.f) != 0:
                            val = eval(str(v.list.f))
                            val = [x for x in val]
                        elif len(v.list.shape) != 0:
                            if k == '_output_shapes':
                                val = cls.__get_output_shapes(layer.attr[k], layout)
                                k = 'shape'
                            else:
                                val = [x for x in v.list.shape]
                        else:
                            val = v
                    else:
                        raise ValueError(
                            f"[ERROR] Unsupported attribute value type in tf1 Unknown op. Op name: {op_name}, attribute name: {k}."
                        )
                    # {T : most op, Tidx : range Tparams : GatherV2}
                    if k in ['T', 'Tidx', 'Tparams'] and v.HasField("type"):
                        k = 'data_type'
                        val = TF_TO_NP[DT_NAME[v.type]].__name__

                    if not isinstance(val, (list, bool, int, float, str, bytes, map)):
                        val = str(val)
                        assert isinstance(val, (list, bool, int, float, str, bytes, map)), f"Unsupported data type!"

                    node.tmp_params[k] = val

                node.tmp_params["name"] = op_name
                node.tmp_params["kind"] = layer.op

            # process inputs
            for iname in layer.input:
                if iname in super_const_dict:
                    super_const = super_const_dict.get(iname)
                    super_const_name = iname
                    # step 1: create XModelNodeConst with tensor
                    # get tensor
                    sc_tensor = super_const.get("tensor")
                    assert (
                        sc_tensor is not None
                    ), f"Failed to get tensor from SuperConst layer: name: {super_const_name}"
                    # create an XModelNodeConst object
                    const_xnode = XModelNodeConst(op_name=iname + "_const")
                    const_xnode.init_layout = const_xnode.layout = xmodel.layout
                    const_xnode.tensor = sc_tensor

                    # update xmodel with const_node
                    xmodel.add_xnode(const_xnode)

                    # step 2: create XModelNodeFixNeuron with quantization info
                    quant_xnode = XModelNodeFixNeuron(op_name=iname)
                    quant_xnode.init_layout = quant_xnode.layout = xmodel.layout
                    quant_xnode.quant_in["bit_width"] = super_const.get("bit_width")
                    quant_xnode.quant_in["quantize_pos"] = super_const.get(
                        "quantize_pos"
                    )
                    quant_xnode.quant_out["bit_width"] = super_const.get(
                        "bit_width"
                    )
                    quant_xnode.quant_out["quantize_pos"] = super_const.get(
                        "quantize_pos"
                    )
                    quant_xnode.is_quantized = True
                    # set input
                    quant_xnode.bottom = [const_xnode.op_name]
                    # update xmodel with quant_xnode
                    xmodel.add_xnode(quant_xnode)

                elif iname in const_layer_dict:

                    const_layer = const_layer_dict.get(iname)
                    tensor = cls.__get_tensor(const_layer.attr["value"])
                    assert tensor is not None
                    # create XModelNodeConst object
                    const_xnode = XModelNodeConst(iname)
                    const_xnode.init_layout = const_xnode.layout = xmodel.layout
                    const_xnode.tensor = tensor

                    # update xmodel with quant_xnode
                    xmodel.add_xnode(const_xnode)

            # set layout
            node.init_layout = node.layout = xmodel.layout
            logger.debug(
                f"property: init layout: {node.init_layout}, current layout: {node.layout}"
            )
            # set bottom
            node.bottom = bottom  # layer.input
            # update xmodel
            xmodel.add_xnode(node)

        logger.info("* end: translate tensorflow nodes to xmodel nodes")

        return xmodel

    @classmethod
    def __get_ksize(cls, value, layout: Layout = Layout.NHWC) -> List[int]:
        ksize = value.list.i
        if layout == Layout.NHWC:
            ksize = [ksize[0]] + [ksize[3]] + ksize[1:3]
        return ksize[-2:]

    @classmethod
    def __get_padding(cls, value) -> str:
        return str(value.s, "utf-8")

    @classmethod
    def __get_shape(cls, value, layout: Layout = Layout.NHWC) -> List[int]:
        shape = None
        if hasattr(value, "shape"):
            if hasattr(value.shape, "dim") and len(value.shape.dim) > 0:
                shape = [d.size for d in value.shape.dim]
        return shape

    @classmethod
    def __get_output_shapes(cls, value, layout: Layout = Layout.NHWC) -> List[int]:
        if hasattr(value.list, "shape") and str(value.list) != '':
            if len(value.list.shape) ==1 and hasattr(value.list.shape[0], "dim") and len(value.list.shape[0].dim) > 0:
                return [dim.size for dim in value.list.shape[0].dim]
            if len(value.list.shape) > 1:
                _output_shapes = []
                for out_shape in value.list.shape:
                    if hasattr(out_shape, "dim") and len(out_shape.dim) > 0:
                        _shape = [dim.size for dim in out_shape.dim]
                        _output_shapes.append(_shape)
                if _output_shapes != []:
                    return _output_shapes

    @classmethod
    def __get_dtype(cls, value) -> np.dtype:
        """
        tensorflow dtype definition:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
        """
        tf_dtype = DT_NAME.get(value.tensor.dtype)
        assert tf_dtype is not None, "Unsupported tf dtype key: {0}".format(
            value.tensor.dtype
        )
        np_dtype = TF_TO_NP.get(tf_dtype)
        return np_dtype

    @classmethod
    def __get_dilations(cls, value, layout: Layout = Layout.NHWC) -> List[int]:
        dilations = value.list.i
        if layout == Layout.NHWC:
            dilations = [dilations[0]] + [dilations[3]] + dilations[1:3]
        return dilations

    @classmethod
    def __get_data_format(cls, value) -> str:
        return str(value.s, "utf-8")

    @classmethod
    def __get_strides(cls, value, layout: Layout = Layout.NHWC) -> List[int]:
        strides = value.list.i
        if layout == Layout.NHWC:
            strides = [strides[0]] + [strides[3]] + strides[1:3]
        return strides[-2:]

    @classmethod
    def __get_tensor(cls, value, node_name=None) -> XTensor:
        # get tensor dtype
        np_dtype = cls.__get_dtype(value)
        # get tensor shape
        shape = cls.__get_tensor_shape(value)
        if len(value.tensor.tensor_content) > 0:
            if shape==[]:
                shape = np.frombuffer(value.tensor.tensor_content, dtype=np_dtype).shape
            assert (
                shape is not None and len(shape) > 0
            ), f"[ERROR] Invalid shape info for a parsed tensor. The relevant node name: {node_name}."
            # create numpy ndarray with tensor content, dtype and shape
            data = np.frombuffer(value.tensor.tensor_content, dtype=np_dtype).reshape(
                shape
            )
        else:
            if np_dtype == np.int32:
                if len(shape) > 0:
                    data = np.full(shape, value.tensor.int_val[0], dtype=np.int32)
                else:
                    data = np.array(value.tensor.int_val, dtype=np.int32)

            elif np_dtype == np.float32:
                # TODO: (samshin) observed precision loss happened on tf_yolov3_voc model
                if len(shape) > 0:
                    data = np.full(shape, value.tensor.float_val[0], dtype=np.float32)
                else:
                    data = np.array(value.tensor.float_val, dtype=np.float32)

            else:
                error = "Unsupported Tensorflow tensor type"
                logger.debug(f"{error}")
                raise ValueError(f"{error}")

        return XTensor(data)

    @classmethod
    def __get_tensor_shape(cls, value) -> List[int]:
        shape = []
        if (
            hasattr(value, "tensor")
            and hasattr(value.tensor, "tensor_shape")
            and hasattr(value.tensor.tensor_shape, "dim")
            and len(value.tensor.tensor_shape.dim) > 0
        ):
            shape = [d.size for d in value.tensor.tensor_shape.dim]
            assert len(shape) > 0
        return shape

    @classmethod
    def __recover_dilated_conv_pass(
        cls, xmodel: XModel, xnode_dict: Dict[str, XModelNode]
    ) -> NoReturn:
        """Convert SpaceToBatchND + Conv2d (or similar op) with dr=1 + BatchToSpaceND into Conv2d with dr>1.

        Parameters
        ----------
        xmodel : XModel
            XModel object.
        xnode_dict : Dict[str, XModelNode]
            a dict object, in which key is name of XModelNode object, value: XModelNode object.

        Returns
        -------
        NoReturn
            No return.
        """
        # * For understanding SpaceToBatchND, reference https://blog.csdn.net/Murdock_C/article/details/87470248

        logger.info(
            "* start: special pass: convert SpaceToBatchND+Conv+BatchToSpaceND into DilatedConv"
        )
        # collect info of (SpaceToBatchND, Conv(dr=1), BatchToSpaceND)
        DilatedConv = namedtuple(
            "DilatedConv", ["space_to_batch_nd", "convolution", "batch_to_space_nd"]
        )
        # key: name of xnode of batchtospacend type
        # value: DilatedConv
        dilated_conv_dict: Dict[str, DilatedConv] = {}
        for xnode in xmodel.xnodes:
            if xnode.op_type == "batchtospacend":
                assert len(xnode.bottom) == 1
                pnode = xnode_dict.get(xnode.bottom[0])
                assert pnode is not None and pnode.op_type in [
                    "conv2d",
                    "depthwise_conv2d",
                ]
                assert len(pnode.bottom) == 1
                ppnode = xnode_dict.get(pnode.bottom[0])
                assert ppnode is not None and ppnode.op_type == "spacetobatchnd"
                dilated_conv_dict[xnode.op_name] = DilatedConv(ppnode, pnode, xnode)

        for _, dilated_conv in dilated_conv_dict.items():
            space_to_batch, conv, batch_to_space = (
                dilated_conv.space_to_batch_nd,
                dilated_conv.convolution,
                dilated_conv.batch_to_space_nd,
            )
            # update bottom info of conv
            pnames = space_to_batch.bottom
            assert len(conv.bottom) == 1
            conv.bottom = pnames
            # update dilation
            assert space_to_batch.block_shape == batch_to_space.block_shape
            conv.dilation = conv.dilation[:2] + space_to_batch.block_shape

            # for get true conv.pad_mode
            conv.tmp_params['recover_dilated_conv']=({'space_to_batch.paddings':space_to_batch.paddings},{'block_shape':space_to_batch.block_shape},{'batch_to_space.crops':batch_to_space.crops})

            # remove both space_to_batch and batch_to_space from xmodel
            xmodel.remove_xnode(space_to_batch)
            xmodel.remove_xnode(batch_to_space)

            # ! revised based on commit 31b8b5ca
            # if space_to_batch.paddings != [0, 0, 0, 0]:
            #     constant_values = [0] * 4
            #     assert constant_values is not None
            #     op_name = str(conv.op_name + "_pad")
            #     node = XModelNodePad(
            #         op_name, space_to_batch.paddings, "constant", constant_values
            #     )
            #     node.bottom = pnames
            #     node.top = [op_name]
            #     xnode_dict[op_name] = node
            #     xmodel.update_xnode(node)
            #     conv.bottom = [op_name]

        logger.info(
            "* end: special pass: convert SpaceToBatchND+Conv+BatchToSpaceND into DilatedConv"
        )

    @classmethod
    def __reduce_biasadd_pass(
        cls,
        xmodel: XModel,
        xnode_dict: Dict[str, XModelNode],
        bias_add_dict: Dict[str, Any],
        super_const_dict: Dict[str, Any],
    ) -> NoReturn:
        """Reduce BiasAdd op and merge into its parent node.

        Parameters
        ----------
        xmodel : XModel
            XModel object.
        xnode_dict : Dict[str, XModelNode]
            a dict object, in which key is name of XModelNode object, value: XModelNode object.
        bias_add_dict : Dict[str, Any]
            a dict object, in which key is name of BiasAdd layer, value is BiasAdd layer.
        super_const_dict: Dict[str, Any]
            a dict object, in which key is name of FixedNeuron layer, value is super_const which is a dict of properties of interest, including quantize_pos, bit_width, tensor, and etc.

        Returns
        -------
        NoReturn
            No return.
        """
        logger.info(
            "* start: special pass: reduce BiasAdd op and merge it into its parent node."
        )

        # create super_biasadd_dict
        # * key: name of BiasAdd layer, value: super_biasadd which is a dict of properties of interest, including input, quantize_pos, bit_width, tensor, and etc.
        super_bias_add_dict = {}
        for name, bias_add_layer in bias_add_dict.items():
            super_bias_add = {}
            # query SuperConst layer as input of current BiasAdd layer
            super_const_layer = None
            for iname in bias_add_layer.input:
                if iname in super_const_dict:
                    super_const_layer = super_const_dict.pop(iname)
                    bias_add_layer.input.remove(iname)
                    break
            assert (
                super_const_layer is not None
            ), f"Not found Super Const layer for current BiasAdd layer (name: {name})."
            super_bias_add["input"] = bias_add_layer.input
            super_bias_add["quantize_pos"] = super_const_layer.get("quantize_pos")
            super_bias_add["bit_width"] = super_const_layer.get("bit_width")
            super_bias_add["signed"] = True
            super_bias_add["round_mode"] = 0  # "STD_ROUND"
            super_bias_add["tensor"] = super_const_layer.get("tensor")
            # update super_bias_add_dict
            super_bias_add_dict[name] = super_bias_add

        # disconnect BiasAdd layers with their parent xnodes
        for name, super_bias_add in super_bias_add_dict.items():
            logger.debug(f"** name: {name}")
            # * According to the definition of BiasAdd layer in TensorFlow,
            # * each one only has TWO inputs, one is Conv2d layer, while
            # * the other Const layer (FixedNeuron layer in fixed point models)
            # * The following algorithm is based on such an assumption above.
            assert (
                len(super_bias_add.get("input")) == 1
            ), f"Current BiasAdd layer has more than 1 inputs. layer name: {name}."
            # get parent node
            conv2d = xnode_dict.get(super_bias_add.get("input")[0])
            if conv2d.op_type not in [
                "conv2d",
                "depthwise_conv2d",
                "matmul",
                "conv2d_transpose",
            ]:  # special cases
                if conv2d.op_type == "batchtospacend":
                    batch_to_space = xnode_dict.get(conv2d.op_name)
                    assert batch_to_space is not None
                    assert len(batch_to_space.bottom) == 1
                    conv2d = xnode_dict.get(batch_to_space.bottom[0])
                else:
                    raise NotImplementedError(
                        f"Unsupported parent node: type: {conv2d.op_type}, name: {conv2d.op_name}"
                    )
                assert conv2d.op_type in [
                    "conv2d",
                    "depthwise_conv2d",
                    "matmul",
                    "conv2d_transpose",
                ], f"Invalid op: type: {conv2d.op_type}, name: {conv2d.op_name}"

            assert (
                conv2d is not None
            ), f"Not found a node named as {super_bias_add.get('input')[0]}."
            if super_bias_add.get("tensor").ndims!=1:
                super_bias_add['tensor'] = super_bias_add.get('tensor').reshape([-1])
                assert super_bias_add.get("tensor").ndims==1, f"type: {conv2d.op_type}, name: {conv2d.op_name}. bias tensor shape's dim num is {super_bias_add.get('tensor').ndims}."
            # set bias_term
            conv2d.bias_term = True
            # set bias
            conv2d.bias = super_bias_add.get("tensor")

            # set quantization info
            conv2d.quant_bias["bit_width"] = super_bias_add.get("bit_width")
            conv2d.quant_bias["quantize_pos"] = super_bias_add.get("quantize_pos")
            conv2d.quant_bias["signed"] = super_bias_add.get("signed")
            conv2d.quant_bias["round_mode"] = super_bias_add.get("round_mode")
            logger.debug(f"*** conv2d: name: {conv2d.op_name}")
            logger.debug(f"*** property: bias_term: {conv2d.bias_term}")
            logger.debug(f"*** property: quant_bias: {conv2d.quant_bias}")

        # disconnect BiasAdd layers with their child xnodes
        for xnode in xmodel.xnodes:
            # filter input, which is BiasAdd layer
            for pname in xnode.bottom:
                if pname in super_bias_add_dict:
                    super_bias_add = super_bias_add_dict.get(pname)
                    iname = super_bias_add.get("input")[0]
                    inode = xnode_dict.get(iname)
                    assert inode.op_type != "fixneuron"
                    if inode.op_type == "batchtospacend":
                        batch_to_space = xnode_dict.get(iname)
                        assert batch_to_space is not None
                        assert len(batch_to_space.bottom) == 1
                        inode = xnode_dict.get(batch_to_space.bottom[0])
                        iname = inode.op_name
                    idx = xnode.bottom.index(pname)
                    xnode.bottom[idx] = iname

        logger.info(
            "* end: special pass: reduce BiasAdd op and merge it into its parent node."
        )

    @classmethod
    def __specialcase_replace_elemadd_with_biasadd(cls, xmodel: XModel) -> NoReturn:
        """Special Case: When deal with inception_resnet_v2 (Tensorflow quantized model), it is hard for
        compiler team to deal with Add op which follows Conv2d op. The solution is to replace Add op with
        BiasAdd op.

        Parameters
        ----------
        xmodel : XModel
            XModel object

        Returns
        -------
        NoReturn
            No return.
        """
        xnodes = [x for x in xmodel.xnodes]
        while len(xnodes) > 0:
            xnode = xnodes.pop(0)
            if xnode.op_type == "elemadd":
                if xnode.bottom is not None and len(xnode.bottom) > 0:
                    # for pname in xnode.bottom:
                    for j, pname in enumerate(xnode.bottom):
                        pnode = xmodel.get_xnode_by_name(pname)
                        assert pnode is not None
                        if (
                            pnode.op_type in ["conv2d", "depthwise_conv2d"]
                            and not pnode.bias_term
                        ):
                            qnode = xmodel.get_xnode_by_name(xnode.bottom[j ^ 1])
                            assert qnode is not None
                            if qnode.op_type == "fixneuron":
                                const_node = xmodel.get_xnode_by_name(qnode.bottom[0])
                                assert const_node is not None
                                if const_node.op_type == "const":
                                    # set quantization info
                                    pnode.quant_bias["bit_width"] = qnode.quant_in[
                                        "bit_width"
                                    ]
                                    pnode.quant_bias["quantize_pos"] = qnode.quant_in[
                                        "quantize_pos"
                                    ]
                                    pnode.quant_bias["signed"] = qnode.quant_in[
                                        "signed"
                                    ]
                                    pnode.quant_bias["round_mode"] = 0  # "STD_ROUND"

                                    # set bias
                                    if const_node.tensor.ndims!=1:
                                        const_node.tensor = const_node.tensor.reshape([-1])
                                        assert const_node.tensor.ndims==1, f"type: {pnode.op_type}, name: {pnode.op_name}. bias tensor shape's dim num is {const_node.tensor.ndims}."
                                    pnode.bias = const_node.tensor
                                    pnode.bias_term = True
                                    # remove const node from xmodel
                                    xmodel.remove_xnode(const_node)
                                    qnode.bottom = []

                                    # remove name of qnode
                                    xnode.bottom.pop(j ^ 1)

                                    # remove xnode from xmodel
                                    xmodel.remove_xnode(xnode)
                                    qnode.top = []
                                else:
                                  continue

                            elif qnode.op_type == "const":
                                # set bias
                                if qnode.tensor.ndims!=1:
                                    qnode.tensor = qnode.tensor.reshape([-1])
                                    assert qnode.tensor.ndims==1, f"type: {pnode.op_type}, name: {pnode.op_name}. bias tensor shape's dim num is {qnode.tensor.ndims}."
                                pnode.bias = qnode.tensor
                                pnode.bias_term = True

                            else:
                                raise ValueError(
                                    f"[ERROR] Unsupported op: type: {qnode.op_type}, name: {qnode.op_name}."
                                )

                            # remove qnode from xmodel
                            xmodel.remove_xnode(qnode)
                            break
        xmodel.topsort()

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
                        if xmodel.origin == "tensorflow":
                            xnode.quant_in["round_mode"] = 0  # "STD_ROUND"
                            xnode.quant_out["round_mode"] = 0
                        else:
                            xnode.quant_in["round_mode"] = 2  # "PY3_ROUND"
                            xnode.quant_out["round_mode"] = 2
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

    @classmethod
    def __remove_activation_linear(cls, xmodel: XModel) -> None:
        """
        Remove activation linear nodes.

        Parameters
        ----------
        xmodel : XModel
            XModel instance
        """
        assert xmodel is not None, "'xmodel' should not be None."

        for xnode in xmodel.xnodes:
            if xnode.op_type == "activation_linear":
                xmodel.remove_xnode(xnode)
