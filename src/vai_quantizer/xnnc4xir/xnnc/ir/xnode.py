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
import math
from typing import Any, Dict, List, NoReturn, Optional, Union

import numpy as np
from xnnc.ir.enums import *
from xnnc.ir.helper import SerdeFactory
from xnnc.proto.openir import openir
from xnnc.tensor.xtensor import XTensor, XTensorCompute, DataFormat

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create console handler and set level to DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(name)s - %(lineno)d - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


class XModelNode(object):
    """
    XModelNode protocol
    """

    def __init__(self, op_name: str, op_type: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        # host
        self.__host: "XModel" = None
        # name of node
        self.__op_name = op_name
        # type of node
        self.__op_type = op_type
        # store names of parent nodes
        self.__bottom = []
        # store names of child nodes
        self.__top = []
        # indicates if the op represented by XModelNode is an in-place operation.
        # This field is used in operator fusion of high-level optimization phase.
        self.__in_place = False
        # True if node is quantized; otherwise, False.
        self.__is_quantized = False
        # quantization info for input
        self.__qin: Dict[str, Optional[int]] = {
            "bit_width": None,
            "quantize_pos": None,
            "signed": True,
            "round_mode": None,
        }
        # quantization info for output
        self.__qout: Dict[str, Optional[int]] = {
            "bit_width": None,
            "quantize_pos": None,
            "signed": True,
            "round_mode": None,
        }
        # shape of input tensors
        self.__inputs_tensor_shape: List[List[int]] = []
        # shape of output tensors
        self.__outputs_tensor_shape: List[List[int]] = []
        # temporary param dict
        self.__tmp_param_dict: Dict[str, Any] = {}
        # init layout
        self.__init_layout: str = None
        # current layout
        self.__curr_layout: str = None
        # inputs tensor
        self.__inputs_tensor: List[XTensor] = []
        # outputs tensor
        self.__outputs_tensor: List[XTensor] = []
        # layout type: layout-insensitive (I), layout-tolerant (T), layout-dependent (D), layout-reconstructed (R)
        self.__layout_type: LayoutType = LayoutType.INSENSITIVE
        # dict for recording index of dims
        self.__idx_dims_dict: Dict[str, Dict[str, Optional[List[List[int]]]]] = {
            "NCHW": {"in": None, "out": None},
            "NHWC": {"in": None, "out": None},
        }

    @property
    def host(self) -> "XModel":
        return self.__host

    @host.setter
    def host(self, xmodel: "XModel") -> NoReturn:
        assert xmodel is not None, "'xmodel' should not be None."
        self.__host = xmodel

    @property
    def op_name(self) -> str:
        return self.__op_name

    @op_name.setter
    def op_name(self, name: str) -> NoReturn:
        assert name is not None, "'name' should not be None."
        assert isinstance(name, str), "'name' should be of str type."
        self.__op_name = name

    @property
    def op_type(self) -> str:
        return self.__op_type

    @property
    def bottom(self) -> List[str]:
        return self.__bottom

    @bottom.setter
    def bottom(self, bottom: List[str]) -> NoReturn:
        self.__bottom = bottom

    @property
    def top(self) -> List[str]:
        return self.__top

    @top.setter
    def top(self, top: List[str]) -> NoReturn:
        self.__top = top

    @property
    def indegree(self) -> int:
        """Get in-degree.

        Returns
        -------
        int
            indegree.
        """
        return len(self.__bottom)

    @property
    def outdegree(self) -> int:
        """Get out-degree.

        Returns
        -------
        int
            outdegree.
        """
        return len(self.__top)

    @property
    def is_quantized(self) -> bool:
        return self.__is_quantized

    @is_quantized.setter
    def is_quantized(self, flag: bool) -> NoReturn:
        self.__is_quantized = flag

    @property
    def is_inplace(self) -> bool:
        return self.__in_place

    @is_inplace.setter
    def is_inplace(self, flag: bool) -> NoReturn:
        self.__in_place = flag

    @property
    def quant_in(self) -> Dict[str, int]:
        return self.__qin

    @property
    def quant_out(self) -> Dict[str, int]:
        return self.__qout

    @property
    def tmp_params(self) -> Dict[str, Any]:
        return self.__tmp_param_dict

    @property
    def inputs_tensor(self) -> List[XTensor]:
        return self.__inputs_tensor

    @inputs_tensor.setter
    def inputs_tensor(self, inputs_tensor: List[XTensor]) -> NoReturn:
        assert inputs_tensor is not None, "'inputs_tensor' should not be None."
        if self.op_type not in ["input"]:
            assert len(inputs_tensor) == len(self.bottom)
        self.__inputs_tensor = inputs_tensor

    @property
    def outputs_tensor(self) -> List[XTensor]:
        return self.__outputs_tensor

    @outputs_tensor.setter
    def outputs_tensor(self, outputs_tensor: List[XTensor]) -> NoReturn:
        assert outputs_tensor is not None, "'outputs_tensor' should not be None."
        self.__outputs_tensor = outputs_tensor

    @property
    def init_layout(self) -> str:
        return self.__init_layout

    @init_layout.setter
    def init_layout(self, layout: str) -> NoReturn:
        assert layout is not None, "'layout' should not be None."
        assert isinstance(layout, str), "'layout' should be of str type."
        self.__init_layout = layout

    @property
    def layout(self) -> str:
        return self.__curr_layout

    @layout.setter
    def layout(self, layout: str) -> NoReturn:
        assert layout is not None, "'layout' should not be None."
        assert isinstance(layout, str), "'layout' should be of str type."
        self.__curr_layout = layout

    @property
    def layout_type(self) -> LayoutType:
        return self.__layout_type

    @layout_type.setter
    def layout_type(self, kind: LayoutType) -> NoReturn:
        assert kind is not None, "'kind' should not be None."
        assert isinstance(kind, LayoutType), "'kind' should be a LayoutType value."
        self.__layout_type = kind

    @property
    def sequence_dims(self) -> Dict[str, Dict[str, Optional[List[List[int]]]]]:
        return self.__idx_dims_dict

    def check_dtype(self) -> bool:
        raise NotImplementedError("virtual method")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."

        if self.op_type == "round_typecast":
            assert (
                self.inputs_tensor is not None and len(self.inputs_tensor) == 1
            ), f"[ERROR] xnnc {self.op_type} op requires one input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}."
            assert (
                self.inputs_tensor[0].data_format.name == layout.name
            ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."
            assert (
                self.inputs_tensor[0].dtype_str == self.tmp_params["src_dtype"].__name__
            ), f"[ERROR] xnnc {self.op_type} requires that the dtype of input tensor ({self.inputs_tensor[0].dtype_str}) should be same as the 'src_dtype' field ({self.tmp_params['src_dtype'].__name__}). Op name: {self.op_name}."

            # compute outputs_tensor
            self.outputs_tensor = [
                XTensor.zeros(
                    self.inputs_tensor[0].shape,
                    dtype=self.tmp_params["dst_dtype"],
                    format=self.inputs_tensor[0].data_format,
                )
            ]

            # update layout
            if self.layout != layout.name:
                self.layout = layout.name

            self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout][
                "in"
            ]

        elif self.op_type == "silence":
            assert layout is not None and isinstance(
                layout, Layout
            ), "'layout' should be a Layout enum value."
            assert (
                self.inputs_tensor is not None and len(self.inputs_tensor) == 1
            ), f"[ERROR] xnnc {self.op_type} requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
            assert (
                self.inputs_tensor[0].data_format.name == layout.name
            ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

            self.outputs_tensor = self.inputs_tensor

            # update layout
            if self.layout != layout.name:
                self.layout = layout.name

            self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout][
                "in"
            ]

        elif self.op_type == "clip_by_value":
            assert (
                self.inputs_tensor is not None and len(self.inputs_tensor) == 1
            ), f"[ERROR] xnnc {self.op_type} op requires one input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}."
            assert (
                self.inputs_tensor[0].data_format.name == layout.name
            ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

            # update layout
            if self.layout != layout.name:
                self.layout = layout.name

            self.outputs_tensor = self.inputs_tensor

            self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout][
                "in"
            ]

        else:
            raise NotImplementedError(
                "virtual method defined in XModelNode base class."
            )

    @staticmethod
    def serialize(xnode: "XModelNode", target: TargetType) -> openir.NodeProto:
        return SerdeFactory.serialize(xnode, target)

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        raise NotImplementedError("virtial method")

    @staticmethod
    def extract_quant_info(
        quant_info: openir.QuantConfig.QuantInfo, quant_dict: Dict[str, Any]
    ) -> None:
        assert quant_info is not None, "'quant_info' should not be None."
        assert quant_dict is not None and isinstance(quant_dict, dict)

        if (
            hasattr(quant_info, "bit_width")
            and hasattr(quant_info, "quant_pos")
            and quant_info.bit_width > 0
        ):
            quant_dict["bit_width"] = quant_info.bit_width
            quant_dict["quantize_pos"] = quant_info.quant_pos
            quant_dict["signed"] = quant_info.signed
            if quant_info.round_mode in [
                openir.RoundMode.CEIL,
                openir.RoundMode.ROUND_UP,
            ]:
                quant_dict["round_mode"] = 1  # DPU_ROUND
            elif quant_info.round_mode in [
                openir.RoundMode.STD_ROUND,
                openir.RoundMode.ROUND_AWAY_FROM_ZERO,
            ]:
                quant_dict["round_mode"] = 0  # STD_ROUND
            elif quant_info.round_mode in [
                openir.RoundMode.PYTHON3_ROUND,
                openir.RoundMode.ROUND_HALF_TO_EVEN,
            ]:
                quant_dict["round_mode"] = 2  # PY3_ROUND
            else:
                round_mode = openir.RoundMode.Name(quant_info.round_mode)
                raise ValueError(f"[ERROR] Unsupported round mode: {round_mode}")


class XModelNodeInput(XModelNode):
    """
    XModelNode Input Protocol
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "input")
        # shape of input image
        self.__shape: List[int] = None
        self.layout_type = LayoutType.INSENSITIVE

    @property
    def shape(self) -> List[int]:
        return self.__shape

    @shape.setter
    def shape(self, shape: List[int]):
        if len(shape) != 4:
            assert (
                len(shape) == 2
            ), f"[ERROR] Only support 2D and 4D input shape. Found an input layer with {len(shape)}-D input shape"

        self.__shape = shape
        self.inputs_tensor = [XTensor(np.zeros(tuple(self.__shape), dtype=np.float32))]

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert self.inputs_tensor is not None
        assert (
            len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Input requires one input: actual: {len(self.inputs_tensor)}."
        assert isinstance(layout, Layout), "'layout' should be a Layout enum value."

        if self.layout == layout.name:
            # update inputs_tensor
            self.inputs_tensor = [
                XTensor(
                    np.zeros(tuple(self.shape), dtype=np.float32),
                    format=DataFormat[self.layout],
                )
            ]

            # set index of dims of outputs
            curr_seq_dict = self.sequence_dims.get(self.layout)
            if len(self.shape) == 4:
                curr_seq_dict["in"] = (
                    [[0, 1, 2, 3]]
                    if self.layout == Layout.NCHW.name
                    else [[0, 2, 3, 1]]
                )
            else:
                # 2 dims
                curr_seq_dict["in"] = [[0, 1]]

            curr_seq_dict["out"] = curr_seq_dict["in"]

        else:
            # update the shape property
            # nhwc -> nchw
            if len(self.shape) == 4:
                if layout == Layout.NCHW:
                    N, H, W, C = self.shape
                    self.shape = [N, C, H, W]

                else:  # nchw -> nhwc
                    N, C, H, W = self.shape
                    self.shape = [N, H, W, C]

                # update inputs_tensor
                self.inputs_tensor = [
                    XTensor(
                        np.zeros(tuple(self.shape), dtype=np.float32),
                        format=DataFormat[layout.name],
                    )
                ]

            # update index of dims
            curr_seq_dict = self.sequence_dims.get(layout.name)
            curr_seq_in = curr_seq_dict.get("in")
            curr_seq_out = curr_seq_dict.get("out")
            if curr_seq_in is None and curr_seq_out is None:
                if len(self.shape) == 4:
                    if layout == Layout.NHWC:
                        curr_seq_dict["in"] = [[0, 2, 3, 1]]
                        curr_seq_dict["out"] = [[0, 2, 3, 1]]
                    else:
                        curr_seq_dict["in"] = [[0, 1, 2, 3]]
                        curr_seq_dict["out"] = [[0, 1, 2, 3]]

            elif curr_seq_in is None or curr_seq_out is None:
                raise ValueError(
                    f"[ERROR] {'in' if curr_seq_in is None else 'out'} sequence is None."
                )

            # update layout
            self.layout = layout.name

        # set outputs_tensor
        self.outputs_tensor = [
            XTensor(
                np.zeros(self.shape, dtype=np.float32), format=DataFormat[self.layout]
            )
        ]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeInput(node.name)
        xnode.shape = [x for x in node.input_param.shape.dim]

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.input_param.quant_config
            # quant_in
            XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            # quant_out
            XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeRelu(XModelNode):
    """
    XModelNode Relu and LeakyRelu Protocol

    Derived from:

    Caffe ReLU and Leaky-ReLU layer
    https://caffe.berkeleyvision.org/tutorial/layers/relu.html

    TensorFlow tf.nn.relu
    https://www.tensorflow.org/api_docs/python/tf/nn/relu

    TensorFlow tf.nn.leaky_relu
    https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu

    tf.keras.layers.LeakyReLU
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/LeakyReLU
    """

    def __init__(self, op_name: str, op_type: str = "relu"):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, op_type)
        # negative slope
        # For Caffe, default 0; for TensorFlow 1.x, default 0.2; for TensorFlow2 Keras, default 0.3
        self.__alpha = 0.0
        self.layout_type = LayoutType.INSENSITIVE

    @property
    def alpha(self) -> float:
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha: float):
        self.__alpha = alpha

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Relu requires one input: actual: {len(self.inputs_tensor)}."

        # update layout
        if self.layout != layout.name:
            assert (
                self.outputs_tensor is not None and len(self.outputs_tensor) == 1
            ), f"[ERROR] xnnc Relu should preform shape inference with {self.layout} first, then with {layout.name}."
            self.layout = layout.name

        # compute outputs_tensor
        self.outputs_tensor = self.inputs_tensor
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeRelu(node.name)
        xnode.alpha = node.relu_param.negative_slope

        if node.quantized:
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeRelu6(XModelNodeRelu):
    """
    XModelNode Relu6 Protocol

    min(max(features, 0), 6)

    Derived from:

    TensorFlow tf.nn.relu6
    https://www.tensorflow.org/api_docs/python/tf/nn/relu6
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNodeRelu.__init__(self, op_name, "relu6")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeRelu6(node.name)
        xnode.alpha = node.relu6_param.negative_slope

        if node.quantized:
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodePRelu(XModelNode):
    """
    XModelNode PRelu Protocol

    f(x) = alpha * x for x < 0
    f(x) = x for x >= 0

    Derived from:

    TensorFlow tf.keras.layers.PReLU
    https://keras.io/api/layers/activation_layers/prelu/
    """

    def __init__(self, op_name: str, op_type: str = "prelu"):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, op_type)
        self.__alpha = 0.0
        # quantization info for weights
        self.__qalpha: Dict[str, Optional[int]] = {
            "bit_width": None,
            "quantize_pos": None,
            "signed": True,
            "round_mode": None,
        }

    @property
    def alpha(self) -> XTensor:
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha: XTensor):
        assert alpha is not None, "'alpha' should not be None."
        assert isinstance(alpha, XTensor), "'alpha' should be an XTensor object."
        self.__alpha = alpha

    @property
    def quant_alpha(self) -> Dict[str, Optional[int]]:
        return self.__qalpha

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Relu requires one input: actual: {len(self.inputs_tensor)}."

        # update layout
        if self.layout != layout.name:
            assert (
                self.outputs_tensor is not None and len(self.outputs_tensor) == 1
            ), f"[ERROR] xnnc Relu should preform shape inference with {self.layout} first, then with {layout.name}."
            self.layout = layout.name

        # compute outputs_tensor
        self.outputs_tensor = self.inputs_tensor
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeConv2d(XModelNode):
    """
    XModelNode Conv2d Protocol

    Derived from:

    Caffe Convolution Layer
    https://caffe.berkeleyvision.org/tutorial/layers/convolution.html

    TensorFlow tf.nn.conv2d
    https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    """

    def __init__(self, op_name: str, ksize: List[int], op_type: str = "conv2d"):
        assert op_name is not None, "'op_name' should not be None."
        assert isinstance(op_name, str), "'op_name' should be of str type."
        assert ksize is not None, "'ksize' should not be None."
        assert (
            isinstance(ksize, list) and len(ksize) == 2
        ), "'kernel_size' should be a list of two positive integers."
        XModelNode.__init__(self, op_name, op_type)
        # [kernel_h, kernel_w]
        self.__ksize: List[int] = ksize
        # [N,C,H,W]
        self.__dilation: List[int] = [1, 1, 1, 1]
        # [stride_h, stride_w]
        self.__strides: List[int] = [1, 1]
        # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        self.__padding: List[int] = [0, 0, 0, 0]
        # "explicit" (caffe, pytorch), "same" or "valid" (tensorflow)
        self.__pad_mode: PadMode = PadMode.EXPLICIT
        # True: bias is present
        self.__bias_term: bool = False
        # round_mode
        # floor: ROUND_DOWN
        # ceil: ROUND_UP
        # std_round: ROUND_AWAY_FROM_ZERO
        # py3_round: ROUND_HALF_TO_EVEN
        self.__round_mode: RoundMode = RoundMode.FLOOR
        # layout: (H, W, in_ch, out_ch)
        self.__weights: XTensor = None
        self.__bias: XTensor = None
        # case1: group == 1: normal convolution
        # case2: group == in_channels: depthwise convolution
        # case3: group-wise convolution
        self.__group: int = 1
        # num of output
        self.__num_output = 0
        # quantization info for weights
        self.__qweights: Dict[str, Optional[int]] = {
            "bit_width": None,
            "quantize_pos": None,
            "signed": True,
            "round_mode": None,
        }
        # quantization info for bias
        self.__qbias: Dict[str, Optional[int]] = {
            "bit_width": None,
            "quantize_pos": None,
            "signed": True,
            "round_mode": None,
        }
        self.layout_type = LayoutType.TOLERANT

    @property
    def kernel_size(self) -> List[int]:
        return self.__ksize

    @property
    def dilation(self) -> List[int]:
        return self.__dilation

    @dilation.setter
    def dilation(self, dilation: List[int]):
        assert dilation is not None, "The argument 'dilation' must not be None."
        assert (
            len(dilation) == 4
        ), "The argument 'dilation' only accepts four positive integers in the form of [1, 1, dilation_h, dilation_w]."
        self.__dilation = dilation

    @property
    def strides(self) -> List[int]:
        return self.__strides

    @strides.setter
    def strides(self, strides: List[int]):
        assert strides is not None, "'strides' should not be None."
        assert (
            isinstance(strides, list) and len(strides) == 2
        ), "'strides' should be a list of two positive integers."
        self.__strides = strides

    @property
    def padding(self) -> List[int]:
        return self.__padding

    @padding.setter
    def padding(self, padding: List[int]):
        assert padding is not None, "The argument 'padding' must not be None."
        assert (
            len(padding) == 4
        ), "The argument 'padding' only accepts four unsigned integers in the form of [pad_top, pad_down, pad_left, pad_right]."
        self.__padding = padding

    @property
    def pad_mode(self) -> PadMode:
        return self.__pad_mode

    @pad_mode.setter
    def pad_mode(self, mode: PadMode):
        assert mode is not None and isinstance(
            mode, PadMode
        ), "'mode' should be of PadMode enum type."
        self.__pad_mode = mode

    @property
    def bias_term(self) -> bool:
        return self.__bias_term

    @bias_term.setter
    def bias_term(self, bias_term: bool):
        assert (
            type(bias_term) == bool
        ), "The argument 'bias_term' only accepts True or False."
        self.__bias_term = bias_term

    @property
    def round_mode(self) -> RoundMode:
        return self.__round_mode

    @round_mode.setter
    def round_mode(self, mode: RoundMode) -> NoReturn:
        assert mode is not None, "'mode' should not be None."
        assert isinstance(mode, RoundMode), "'mode' should be of RoundMode enum type."
        self.__round_mode = mode

    @property
    def weights(self) -> XTensor:
        return self.__weights

    @weights.setter
    def weights(self, weights: XTensor):
        assert weights is not None, "'weights' should not be None."
        assert isinstance(weights, XTensor), "'weights' should be an XTensor object."
        self.__weights = weights

    @property
    def bias(self) -> XTensor:
        return self.__bias

    @bias.setter
    def bias(self, bias: XTensor):
        assert bias is not None, "'bias' should not be None."
        assert isinstance(bias, XTensor), "'bias' should be an XTensor object."
        self.__bias = bias

    @property
    def group(self) -> int:
        return self.__group

    @group.setter
    def group(self, group: int):
        self.__group = group

    @property
    def num_output(self) -> int:
        return self.__num_output

    @num_output.setter
    def num_output(self, num: int) -> NoReturn:
        assert num is not None, "'num' should not be None."
        assert isinstance(num, int), "'num' should be a positive integer."
        assert num > 0, "'num' should be a positive integer."
        self.__num_output = num

    @property
    def quant_weights(self) -> Dict[str, Optional[int]]:
        return self.__qweights

    @property
    def quant_bias(self) -> Dict[str, Optional[int]]:
        return self.__qbias

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None
            and len(self.inputs_tensor) == 1
            and self.inputs_tensor[0].ndims == 4
        ), f"[ERROR] xnnc {self.op_type} op requires one input of 4 dimensions: actual: {len(self.inputs_tensor)}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        if not self.outputs_tensor or self.layout != layout:
            in_tensor = self.inputs_tensor[0]

            curr_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
            if not self.outputs_tensor:
                assert self.layout == in_tensor.data_format.name

                if self.layout == Layout.NCHW.name:
                    N, _, H, W = in_tensor.shape
                else:
                    N, H, W, _ = in_tensor.shape

            else:
                assert in_tensor.data_format.name == layout.name

                if layout == Layout.NHWC:
                    N, H, W, _ = in_tensor.shape

                else:
                    N, _, H, W = in_tensor.shape

            # set pad_mode  when diliation > 1
            if 'recover_dilated_conv' in self.tmp_params:
                [pad_top, pad_bottom, pad_left, pad_right] = self.tmp_params['recover_dilated_conv'][0]['space_to_batch.paddings']
                [block_shape_height, block_shape_width] = self.tmp_params['recover_dilated_conv'][1]['block_shape']
                [crop_top, crop_bottom, crop_left, crop_right] = self.tmp_params['recover_dilated_conv'][2]['batch_to_space.crops']
                height, width = H, W
                # step1: get space_to_batch out_height and out_width
                height_pad = pad_top + height + pad_bottom
                width_pad = pad_left + width + pad_right
                conv_height = height_pad/block_shape_height
                conv_width = width_pad/block_shape_width
                # step2: get conv pad is 'valid' and diliation=1  out_height and out_width
                [filter_height, filter_width, in_channels, out_channels] = self.weights.shape
                [stride_height, stride_width] = self.strides
                conv_out_height = math.ceil(conv_height - filter_height + 1) / stride_height
                conv_out_width = math.ceil(conv_width - filter_width + 1) / stride_width
                # step3: get batch_to_space out_height and out_width
                actual_out_height = int(conv_out_height * block_shape_height - crop_top - crop_bottom)
                actual_out_width = int(conv_out_width * block_shape_width - crop_left - crop_right)


                # computer when conv diliation>1  and pad is 'VALIDE'
                conv_out_valid_height = int(math.ceil(height - (filter_height - 1)*block_shape_height) / stride_height)
                conv_out_valid_width = int(math.ceil(width - (filter_width - 1)*block_shape_width) / stride_width)

                # computer when conv diliation>1  and pad is 'SAME'
                conv_out_same_height = int(math.ceil(height / stride_height))
                conv_out_same_width  = int(math.ceil(width / stride_width))

                # set ture pad_mode when diliation > 1
                if conv_out_valid_height == actual_out_height and conv_out_valid_width == actual_out_width:
                    self.pad_mode = PadMode.VALID
                elif conv_out_same_height == actual_out_height and conv_out_same_width == actual_out_width:
                    self.pad_mode = PadMode.SAME

            _, _, _, OC = self.weights.shape

            ksize_h, ksize_w = self.kernel_size
            stride_h, stride_w = self.strides
            pad_h_before, pad_h_after, pad_w_before, pad_w_after = self.padding
            _, _, dilation_h, dilation_w = self.dilation

            if self.pad_mode == PadMode.EXPLICIT:
                if self.round_mode in [RoundMode.FLOOR, RoundMode.ROUND_DOWN]:
                    round_func = math.floor
                elif self.round_mode in [RoundMode.CEIL, RoundMode.ROUND_UP]:
                    round_func = math.ceil
                else:
                    raise TypeError(
                        f"[ERROR] unsupported round mode in {self.op_type}: {self.round_mode}"
                    )
                OH = (
                    round_func(
                        1.0
                        * (
                            H
                            + pad_h_before
                            + pad_h_after
                            - ksize_h
                            - (ksize_h - 1) * (dilation_h - 1)
                        )
                        / stride_h
                    )
                    + 1
                )
                OW = (
                    round_func(
                        1.0
                        * (
                            W
                            + pad_w_before
                            + pad_w_after
                            - ksize_w
                            - (ksize_w - 1) * (dilation_w - 1)
                        )
                        / stride_w
                    )
                    + 1
                )

            elif self.pad_mode == PadMode.SAME:
                OH = math.ceil(1.0 * (H + pad_h_before + pad_h_after) / stride_h)
                OW = math.ceil(1.0 * (W + pad_w_before + pad_w_after) / stride_w)

            elif self.pad_mode == PadMode.VALID:
                OH = math.ceil(
                    1.0
                    * (H + pad_h_before + pad_h_after - (ksize_h - 1) * dilation_h)
                    / stride_h
                )
                OW = math.ceil(
                    1.0
                    * (W + pad_w_before + pad_w_after - (ksize_w - 1) * dilation_w)
                    / stride_w
                )

            else:
                raise ValueError(f"[ERROR] Unsupported pad mode: {self.pad_mode}.")

            # compute outputs_tensor
            out_shape = None
            if self.layout == layout.name:
                if layout == Layout.NCHW:
                    out_shape = [N, OC, OH, OW]
                else:
                    out_shape = [N, OH, OW, OC]

            else:
                if layout == Layout.NHWC:
                    out_shape = [N, OH, OW, OC]
                else:
                    out_shape = [N, OC, OH, OW]

            assert out_shape is not None

            # update layout
            if self.outputs_tensor:
                self.layout = layout.name

            out_tensor = XTensor.zeros(
                out_shape, dtype=in_tensor.dtype, format=in_tensor.data_format
            )
            self.outputs_tensor = [out_tensor]

            # compute the dimension sequence of the output
            self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout][
                "in"
            ]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeConv2d(node.name, list(node.conv2d_param.kernel))
        # strides
        xnode.strides = list(node.conv2d_param.strides)
        # pad_mode
        xnode.pad_mode = PadMode[openir.PadMode.Name(node.conv2d_param.pad_mode)]
        # padding
        if node.conv2d_param.pad_mode == openir.PadMode.EXPLICIT:
            xnode.padding = list(node.conv2d_param.padding)
        # dilation (n,c,h,w)
        xnode.dilation = [1, 1] + list(node.conv2d_param.dilation)
        # bias_term
        xnode.bias_term = node.conv2d_param.bias_term
        # round_mode
        round_mode = openir.RoundMode.Name(node.conv2d_param.round_mode)
        xnode.round_mode = RoundMode[round_mode]
        # group
        xnode.group = node.conv2d_param.group
        # weights
        xnode.weights = XTensor.deserialize(node.conv2d_param.weights)
        # bias
        if xnode.bias_term:
            xnode.bias = XTensor.deserialize(node.conv2d_param.bias)
        # num_output
        if node.conv2d_param.output_channels > 0:
            xnode.num_output = node.conv2d_param.output_channels
        else:
            xnode.num_output = xnode.weights.shape[-1]

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

            # quant_weights
            XModelNode.extract_quant_info(
                quant_config.quant_weight, xnode.quant_weights
            )
            # quant_bias
            if xnode.bias_term:
                XModelNode.extract_quant_info(quant_config.quant_bias, xnode.quant_bias)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeConv2dDepthwise(XModelNodeConv2d):
    """
    XModelNode Depthwise-Conv2d Protocol

    Derived from: Tensorflow
    Reference: https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d
    """

    def __init__(self, op_name: str, ksize: List[int]):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNodeConv2d.__init__(self, op_name, ksize, "depthwise_conv2d")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc {self.op_type} op requires one input: actual: {len(self.inputs_tensor)}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        in_tensor = self.inputs_tensor[0]

        # compute outputs_tensor
        if layout == Layout.NCHW:
            N, _, H, W = in_tensor.shape
        else:
            N, H, W, _ = in_tensor.shape

        # set pad_mode  when diliation > 1
        if 'recover_dilated_conv' in self.tmp_params:
            [pad_top, pad_bottom, pad_left, pad_right] = self.tmp_params['recover_dilated_conv'][0]['space_to_batch.paddings']
            [block_shape_height, block_shape_width] = self.tmp_params['recover_dilated_conv'][1]['block_shape']
            [crop_top, crop_bottom, crop_left, crop_right] = self.tmp_params['recover_dilated_conv'][2]['batch_to_space.crops']
            height, width = H, W
            # step1: get space_to_batch out_height and out_width
            height_pad = pad_top + height + pad_bottom
            width_pad = pad_left + width + pad_right
            conv_height = height_pad/block_shape_height
            conv_width = width_pad/block_shape_width
            # step2: get conv pad is 'valid' and diliation=1  out_height and out_width
            [filter_height, filter_width, in_channels, out_channels] = self.weights.shape
            [stride_height, stride_width] = self.strides
            conv_out_height = math.ceil(conv_height - filter_height + 1) / stride_height
            conv_out_width = math.ceil(conv_width - filter_width + 1) / stride_width
            # step3: get batch_to_space out_height and out_width
            actual_out_height = int(conv_out_height * block_shape_height - crop_top - crop_bottom)
            actual_out_width = int(conv_out_width * block_shape_width - crop_left - crop_right)


            # computer when conv diliation>1  and pad is 'VALIDE'
            conv_out_valid_height = int(math.ceil(height - (filter_height - 1)*block_shape_height) / stride_height)
            conv_out_valid_width = int(math.ceil(width - (filter_width - 1)*block_shape_width) / stride_width)

            # computer when conv diliation>1  and pad is 'SAME'
            conv_out_same_height = int(math.ceil(height / stride_height))
            conv_out_same_width  = int(math.ceil(width / stride_width))

            # set ture pad_mode when diliation > 1
            if conv_out_valid_height == actual_out_height and conv_out_valid_width == actual_out_width:
                self.pad_mode = PadMode.VALID
            elif conv_out_same_height == actual_out_height and conv_out_same_width == actual_out_width:
                self.pad_mode = PadMode.SAME

        OC = self.num_output
        ksize_h, ksize_w = self.kernel_size
        stride_h, stride_w = self.strides
        pad_h_before, pad_h_after, pad_w_before, pad_w_after = self.padding
        _, _, dilation_h, dilation_w = self.dilation

        if self.pad_mode == PadMode.EXPLICIT:
            if self.round_mode in [RoundMode.FLOOR, RoundMode.ROUND_DOWN]:
                round_func = math.floor
            elif self.round_mode in [RoundMode.CEIL, RoundMode.ROUND_UP]:
                round_func = math.ceil
            else:
                raise TypeError(
                    f"[ERROR] unsupported round mode in {self.op_type}: {self.round_mode}"
                )

            OH = (
                round_func(
                    1.0
                    * (
                        H
                        + pad_h_before
                        + pad_h_after
                        - ksize_h
                        - (ksize_h - 1) * (dilation_h - 1)
                    )
                    / stride_h
                )
                + 1
            )
            OW = (
                round_func(
                    1.0
                    * (
                        W
                        + pad_w_before
                        + pad_w_after
                        - ksize_w
                        - (ksize_w - 1) * (dilation_w - 1)
                    )
                    / stride_w
                )
                + 1
            )

        elif self.pad_mode == PadMode.SAME:
            OH = math.ceil(1.0 * (H + pad_h_before + pad_h_after) / stride_h)
            OW = math.ceil(1.0 * (W + pad_w_before + pad_w_after) / stride_w)

        elif self.pad_mode == PadMode.VALID:
            OH = math.ceil(
                1.0
                * (H + pad_h_before + pad_h_after - (ksize_h - 1) * dilation_h)
                / stride_h
            )
            OW = math.ceil(
                1.0
                * (W + pad_w_before + pad_w_after - (ksize_w - 1) * dilation_w)
                / stride_w
            )

        if layout == Layout.NHWC:
            out_tensor = XTensor(
                np.zeros([N, OH, OW, OC], dtype=in_tensor.dtype),
                format=in_tensor.data_format,
            )
        else:
            out_tensor = XTensor(
                np.zeros([N, OC, OH, OW], dtype=in_tensor.dtype),
                format=in_tensor.data_format,
            )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeConv2dDepthwise(
            node.name, list(node.depthwise_conv2d_param.kernel)
        )
        # strides
        xnode.strides = list(node.depthwise_conv2d_param.strides)
        # pad_mode
        xnode.pad_mode = PadMode[
            openir.PadMode.Name(node.depthwise_conv2d_param.pad_mode)
        ]
        # padding
        if node.depthwise_conv2d_param.pad_mode == openir.PadMode.EXPLICIT:
            xnode.padding = list(node.depthwise_conv2d_param.padding)
        # dilation: (n,c,h,w)
        xnode.dilation = [1, 1] + list(node.depthwise_conv2d_param.dilation)
        # bias_term
        xnode.bias_term = node.depthwise_conv2d_param.bias_term
        # round_mode
        round_mode = openir.RoundMode.Name(node.depthwise_conv2d_param.round_mode)
        xnode.round_mode = RoundMode[round_mode]
        # group
        xnode.group = node.depthwise_conv2d_param.group
        # weights
        xnode.weights = XTensor.deserialize(node.depthwise_conv2d_param.weights)
        # bias
        if xnode.bias_term:
            xnode.bias = XTensor.deserialize(node.depthwise_conv2d_param.bias)
        # num_output
        assert node.depthwise_conv2d_param.output_channels > 0
        xnode.num_output = node.depthwise_conv2d_param.output_channels

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

            # quant_weights
            XModelNode.extract_quant_info(
                quant_config.quant_weight, xnode.quant_weights
            )
            # quant_bias
            if xnode.bias_term:
                XModelNode.extract_quant_info(quant_config.quant_bias, xnode.quant_bias)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeDeconvolution(XModelNodeConv2d):
    """
    XModelNode Deconvolution Protocol

    Derived from: Caffe
    https://caffe.berkeleyvision.org/tutorial/layers/deconvolution.html
    """

    def __init__(self, op_name: str, ksize: List[int]):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNodeConv2d.__init__(self, op_name, ksize, "deconvolution")
        self.layout_type = LayoutType.TOLERANT

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc {self.op_type} requires one input: actual: {len(self.inputs_tensor)}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        in_tensor = self.inputs_tensor[0]

        if layout == Layout.NCHW:
            N, _, H, W = in_tensor.shape
        else:
            N, H, W, _ = in_tensor.shape

        _, _, _, OC = self.weights.shape

        ksize_h, ksize_w = self.kernel_size
        stride_h, stride_w = self.strides
        pad_h_before, pad_h_after, pad_w_before, pad_w_after = self.padding
        _, _, dilation_h, dilation_w = self.dilation

        assert (
            self.pad_mode == PadMode.EXPLICIT
            and self.padding is not None
            and len(self.padding) == 4
        ), f"[ERROR] xnnc {self.op_type} requires the 'padding' property must be set. op name: {self.op_name}"
        OH = (
            stride_h * (H - 1)
            + (dilation_h * (ksize_h - 1) + 1)
            - (pad_h_before + pad_h_after)
        )
        OW = (
            stride_w * (W - 1)
            + (dilation_w * (ksize_w - 1) + 1)
            - (pad_w_before + pad_w_after)
        )

        if layout == Layout.NHWC:
            out_tensor = XTensor.zeros(
                [N, OH, OW, OC], dtype=in_tensor.dtype, format=in_tensor.data_format
            )
        else:
            out_tensor = XTensor.zeros(
                [N, OC, OH, OW], dtype=in_tensor.dtype, format=in_tensor.data_format
            )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("deconvolution")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        raise NotImplementedError("deconvolution")


class XModelNodeDeconvolutionDepthwise(XModelNodeConv2d):
    """
    XModelNode DepthwiseDeconvolution Protocol

    Derived from:

    Caffe
    https://caffe.berkeleyvision.org/tutorial/layers/deconvolution.html
    """

    def __init__(self, op_name: str, ksize: List[int]):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNodeConv2d.__init__(self, op_name, ksize, "depthwise_deconvolution")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc depthwise_deconvolution op requires one inputs: actual: {len(self.inputs_tensor)}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        in_tensor = self.inputs_tensor[0]

        if layout == Layout.NCHW:
            N, _, H, W = in_tensor.shape
        else:
            N, H, W, _ = in_tensor.shape

        OC = self.num_output

        ksize_h, ksize_w = self.kernel_size
        stride_h, stride_w = self.strides
        pad_h_before, pad_h_after, pad_w_before, pad_w_after = self.padding
        _, _, dilation_h, dilation_w = self.dilation

        assert (
            self.pad_mode == PadMode.EXPLICIT
            and self.padding is not None
            and len(self.padding) == 4
        ), f"[ERROR] xnnc Deconvolution (depthwise) requires the 'padding' property must be set. op name: {self.op_name}"

        OH = (
            stride_h * (H - 1)
            + (dilation_h * (ksize_h - 1) + 1)
            - (pad_h_before + pad_h_after)
        )
        OW = (
            stride_w * (W - 1)
            + (dilation_w * (ksize_w - 1) + 1)
            - (pad_w_before + pad_w_after)
        )

        if layout == Layout.NHWC:
            out_tensor = XTensor.zeros(
                [N, OH, OW, OC], dtype=in_tensor.dtype, format=in_tensor.data_format
            )
        else:
            out_tensor = XTensor.zeros(
                [N, OC, OH, OW], dtype=in_tensor.dtype, format=in_tensor.data_format
            )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("depthwise_deconvolution")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        raise NotImplementedError("depthwise_deconvolution")


class XModelNodeConv2dTranspose(XModelNodeConv2d):
    """
    XModelNode Conv2dTranspose Protocol

    Notice: weights' layout: (h,w,ic,oc)

    Derived from:

    Caffe Deconvolution Layer
    https://caffe.berkeleyvision.org/tutorial/layers/deconvolution.html

    TensorFlow tf.nn.conv2d_transpose
    https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose

    TensorFlow tensorflow::ops::Conv2DBackpropInput
    https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/conv2-d-backprop-input

    TensorFlow2 tf.keras.layers.Conv2DTranspose
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
    """

    def __init__(self, op_name: str, ksize: List[int]):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNodeConv2d.__init__(self, op_name, ksize, "conv2d_transpose")
        # A 1-D Tensor representing the output shape of the deconvolution.
        self.__output_shape: List[int] = None

    @property
    def output_shape(self) -> Optional[List[int]]:
        return self.__output_shape

    @output_shape.setter
    def output_shape(self, shape: List[int]) -> NoReturn:
        assert shape is not None, "'shape' should not be None."
        assert isinstance(shape, list), "'shape' should be a list of integers."
        self.__output_shape = shape

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and 1 <= len(self.inputs_tensor) <= 2
        ), f"[ERROR] xnnc {self.op_type} op requires one or two inputs: actual: {len(self.inputs_tensor)}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        # set output_shape property and compute outputs_tensor
        out_tensor = None
        # tensorflow 1.x
        if self.host.origin == "tensorflow":
            if len(self.inputs_tensor) == 2:
                in_tensor, output_shape_tensor = self.inputs_tensor
                # set output_shape property
                self.output_shape = output_shape_tensor.tolist()
                out_tensor = XTensor(
                    np.zeros(self.output_shape, dtype=in_tensor.dtype),
                    format=self.inputs_tensor[0].data_format,
                )
            else:
                raise ValueError(
                    "[ERROR] Unsupported case in xnnc XModelNodeConv2dTranspose."
                )

        elif self.host.origin == "tensorflow2":
            assert len(self.inputs_tensor) == 1
            assert self.pad_mode in [PadMode.SAME, PadMode.VALID]
            # height and width of output
            if layout == Layout.NHWC:
                N, H, W, _ = self.inputs_tensor[0].shape
            else:
                N, _, H, W = self.inputs_tensor[0].shape
            stride_h, stride_w = self.strides
            if self.pad_mode == PadMode.VALID:
                # Get the dilated kernel size
                ksize_h, ksize_w = self.kernel_size
                dilation_h, dilation_w = self.dilation[-2:]
                dilated_ksize_h = ksize_h + (ksize_h - 1) * (dilation_h - 1)
                dilated_ksize_w = ksize_w + (ksize_w - 1) * (dilation_w - 1)
                OH = H * stride_h + max(dilated_ksize_h - stride_h, 0)
                OW = W * stride_w + max(dilated_ksize_w - stride_w, 0)
            else:
                OH = H * stride_h
                OW = W * stride_w

            _, _, OC, _ = self.weights.shape
            # set output_shape property
            self.output_shape = (
                [N, OH, OW, OC] if layout == Layout.NHWC else [N, OC, OH, OW]
            )
            out_tensor = XTensor(
                np.zeros(self.output_shape, dtype=self.inputs_tensor[0].dtype),
                format=self.inputs_tensor[0].data_format,
            )

        elif len(self.inputs_tensor) == 1 and self.output_shape is not None:
            in_tensor = self.inputs_tensor[0]
            out_tensor = XTensor(
                np.zeros(self.output_shape, dtype=in_tensor.dtype),
                format=self.inputs_tensor[0].data_format,
            )

        else:
            raise ValueError(
                f"[ERROR] Unsupported case for computing outputs_tensor: op type: {self.op_type}, op name: {self.op_name}."
            )

        assert (
            out_tensor is not None
        ), f"[ERROR] Failed to compute outputs_tensor: op type: {self.op_type}, op name: {self.op_name}"
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeConv2dTranspose(
            node.name, list(node.conv2d_transpose_param.kernel)
        )

        # strides
        xnode.strides = list(node.conv2d_transpose_param.strides)
        # pad_mode
        xnode.pad_mode = PadMode[
            openir.PadMode.Name(node.conv2d_transpose_param.pad_mode)
        ]
        # padding
        if node.conv2d_transpose_param.pad_mode == openir.PadMode.EXPLICIT:
            xnode.padding = list(node.conv2d_transpose_param.padding)
        # dilation (n,c,h,w)
        xnode.dilation = [1, 1] + list(node.conv2d_transpose_param.dilation)
        # bias_term
        xnode.bias_term = node.conv2d_transpose_param.bias_term
        # round_mode
        round_mode = openir.RoundMode.Name(node.conv2d_transpose_param.round_mode)
        xnode.round_mode = RoundMode[round_mode]
        # group
        xnode.group = node.conv2d_transpose_param.group
        # weights
        weights = XTensor.deserialize(node.conv2d_transpose_param.weights)
        xnode.weights = XTensorCompute.flip(weights, axis=(0, 1)).transpose(
            (0, 1, 3, 2)
        )
        # bias
        if xnode.bias_term:
            xnode.bias = XTensor.deserialize(node.conv2d_transpose_param.bias)
        # output_shape: (output_h, output_w)
        xnode.output_shape = list(node.conv2d_transpose_param.output_shape)

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

            # quant_weights
            XModelNode.extract_quant_info(
                quant_config.quant_weight, xnode.quant_weights
            )
            # quant_bias
            if xnode.bias_term:
                XModelNode.extract_quant_info(quant_config.quant_bias, xnode.quant_bias)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodePool2d(XModelNode):
    """
    Base class of MaxPool and AvgPool
    """

    def __init__(self, op_name: str, ksize: List[int], op_type: str):
        assert op_name is not None, "The argument 'op_name' should not be None."
        assert ksize is not None, "The argument 'ksize' should not be None."
        assert (
            len(ksize) == 2
        ), "kernel size only accepts two unsigned integers in the form of [kernel_h, kernel_w]."
        assert op_type is not None, "'op_type' should not be None."
        XModelNode.__init__(self, op_name, op_type)
        # [kernel_h, kernel_w]
        self.__ksize: List[int] = ksize
        # [stride_h, stride_w]
        self.__strides: List[int] = [1, 1]
        # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        self.__padding: List[int] = [0, 0, 0, 0]
        # "explicit", "same" or "valid"
        self.__pad_mode: PadMode = PadMode.EXPLICIT
        # round_mode
        # floor: ROUND_DOWN
        # ceil: ROUND_UP
        # std_round: ROUND_AWAY_FROM_ZERO
        # py3_round: ROUND_HALF_TO_EVEN
        self.__round_mode: RoundMode = RoundMode.FLOOR
        # True indicates global max/avgpool; otherwise, False
        self.__is_global: bool = False
        # [N,C,H,W]
        self.__dilation: List[int] = [1, 1, 1, 1]
        self.layout_type = LayoutType.TOLERANT

    @property
    def kernel_size(self) -> List[int]:
        return self.__ksize

    @property
    def strides(self) -> List[int]:
        return self.__strides

    @strides.setter
    def strides(self, strides: List[int]):
        assert (
            len(strides) == 2
        ), "strides only accepts two unsigned integers in the form of [stride_h, stride_w]."
        if self.is_global:
            self.__strides = [1, 1]
        else:
            self.__strides = strides

    @property
    def padding(self) -> List[int]:
        return self.__padding

    @padding.setter
    def padding(self, padding: List[int]):
        assert (
            len(padding) == 4
        ), "padding only accepts four unsigned integers in the form of [pad_top, pad_down, pad_left, pad_right]."
        if self.is_global:
            self.__padding = [0, 0, 0, 0]
        else:
            self.__padding = padding

    @property
    def pad_mode(self) -> PadMode:
        return self.__pad_mode

    @pad_mode.setter
    def pad_mode(self, mode: PadMode):
        assert mode is not None and isinstance(
            mode, PadMode
        ), "'mode' should be of PadMode enum type."
        self.__pad_mode = mode

    @property
    def round_mode(self) -> RoundMode:
        return self.__round_mode

    @round_mode.setter
    def round_mode(self, mode: RoundMode) -> NoReturn:
        assert mode is not None, "'mode' should not be None."
        assert isinstance(mode, RoundMode), "'mode' should be of RoundMode enum type."
        self.__round_mode = mode

    @property
    def is_global(self) -> bool:
        return self.__is_global

    @is_global.setter
    def is_global(self, is_global: bool):
        self.__is_global = is_global
        if is_global:
            self.padding = [0, 0, 0, 0]
            self.strides = [1, 1]

    @property
    def dilation(self) -> List[int]:
        return self.__dilation

    @dilation.setter
    def dilation(self, dilation: List[int]) -> NoReturn:
        assert dilation is not None, "'dilation' should not be None."
        assert len(dilation) == 4, "'dilation' should be of length 4."
        self.__dilation = dilation

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc {self.op_type} requires one input: actual: {len(self.inputs_tensor)}."

        in_tensor = self.inputs_tensor[0]
        assert (
            in_tensor.ndims == 4
        ), f"[ERROR] xnnc {self.op_type} requires the rank of input tensor is 4."
        assert (
            in_tensor.data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        if layout == Layout.NCHW:
            N, C, H, W = in_tensor.shape
        else:
            N, H, W, C = in_tensor.shape

        if self.is_global:
            if not self.kernel_size or self.kernel_size == [0, 0]:
                self.__ksize = [H, W]

            OH, OW = 1, 1
        else:
            ksize_h, ksize_w = self.kernel_size
            stride_h, stride_w = self.strides
            pad_h_before, pad_h_after, pad_w_before, pad_w_after = self.padding
            dilation_h, dilation_w = self.dilation[::-2]
            if self.pad_mode == PadMode.EXPLICIT:
                if self.round_mode in [RoundMode.FLOOR, RoundMode.ROUND_DOWN]:
                    round_func = math.floor
                elif self.round_mode in [RoundMode.CEIL, RoundMode.ROUND_UP]:
                    round_func = math.ceil
                else:
                    raise TypeError(
                        f"[ERROR] unsupported round mode in {self.op_type}: {self.round_mode}"
                    )

                OH = (
                    round_func(
                        1.0 * (H + pad_h_before + pad_h_after - ksize_h) / stride_h
                    )
                    + 1
                )
                OW = (
                    round_func(
                        1.0 * (W + pad_w_before + pad_w_after - ksize_w) / stride_w
                    )
                    + 1
                )

            elif self.pad_mode == PadMode.SAME:
                OH = math.ceil(
                    1.0 * (H + self.padding[0] + self.padding[1]) / self.strides[0]
                )
                OW = math.ceil(
                    1.0 * (W + self.padding[2] + self.padding[3]) / self.strides[1]
                )

            elif self.pad_mode == PadMode.VALID:
                OH = math.ceil(
                    1.0
                    * (
                        H
                        + self.padding[0]
                        + self.padding[1]
                        - (self.kernel_size[0] - 1) * self.dilation[2]
                    )
                    / self.strides[0]
                )
                OW = math.ceil(
                    1.0
                    * (
                        W
                        + self.padding[2]
                        + self.padding[3]
                        - (self.kernel_size[1] - 1) * self.dilation[3]
                    )
                    / self.strides[1]
                )

            else:
                raise ValueError(f"[ERROR] Unsupported pad mode: {self.pad_mode}.")

        if layout == Layout.NHWC:
            out_tensor = XTensor(
                np.zeros([N, OH, OW, C], dtype=in_tensor.dtype),
                format=in_tensor.data_format,
            )
        else:
            out_tensor = XTensor(
                np.zeros([N, C, OH, OW], dtype=in_tensor.dtype),
                format=in_tensor.data_format,
            )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        self.sequence_dims[self.layout]["out"] = [
            self.sequence_dims[self.layout]["in"][0]
        ]


class XModelNodeMaxPool(XModelNodePool2d):
    """
    XModelNode MaxPool Protocol


    Derived from:

    Caffe Pooling
    https://caffe.berkeleyvision.org/tutorial/layers/pooling.html

    TensorFlow tf.nn.max_pool2d
    https://www.tensorflow.org/api_docs/python/tf/nn/max_pool2d
    """

    def __init__(self, op_name: str, ksize: List[int]):
        assert op_name is not None, "The argument 'op_name' should not be None."
        assert ksize is not None, "The argument 'ksize' should not be None."
        assert (
            len(ksize) == 2
        ), "kernel size only accepts two unsigned integers in the form of [kernel_h, kernel_w]."
        XModelNodePool2d.__init__(self, op_name, ksize, "maxpool")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeMaxPool(node.name, list(node.maxpool2d_param.kernel))

        # is_global
        xnode.is_global = node.maxpool2d_param.is_global
        # strides
        xnode.strides = list(node.maxpool2d_param.strides)
        # pad_mode
        xnode.pad_mode = PadMode(node.maxpool2d_param.pad_mode)
        # padding
        if node.maxpool2d_param.pad_mode == openir.PadMode.EXPLICIT:
            xnode.padding = list(node.maxpool2d_param.padding)
        # dilation: (n,c,h,w)
        xnode.dilation = [1, 1] + list(node.maxpool2d_param.dilation)
        # round_mode
        xnode.round_mode = RoundMode(node.maxpool2d_param.round_mode)

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeAvgPool(XModelNodePool2d):
    """
    XModelNode AvgPool Protocol

    Derived from:

    Caffe Pooling
    https://caffe.berkeleyvision.org/tutorial/layers/pooling.html

    TensorFlow tf.nn.avg_pool2d
    https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool2d
    """

    def __init__(self, op_name: str, ksize: List[int]):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNodePool2d.__init__(self, op_name, ksize, "avgpool")
        # True indicates count data in the pad position; otherwise, False
        # For Caffe and PyTorch, set True; for TensorFlow, set False
        self.__zero_padding_included: bool = True
        # For PyTorch and TensorFlor, set True; for Caffe, set False
        self.__count_include_invalid: bool = True

    @property
    def zero_padding_included(self) -> bool:
        return self.__zero_padding_included

    @zero_padding_included.setter
    def zero_padding_included(self, flag: bool) -> NoReturn:
        assert flag is not None
        self.__zero_padding_included = flag

    @property
    def count_include_invalid(self) -> bool:
        return self.__count_include_invalid

    @count_include_invalid.setter
    def count_include_invalid(self, flag: bool) -> NoReturn:
        assert flag is not None and isinstance(flag, bool)
        self.__count_include_invalid = flag

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeAvgPool(node.name, list(node.avgpool2d_param.kernel))

        # is_global
        xnode.is_global = node.avgpool2d_param.is_global

        # zero_padding_included
        xnode.zero_padding_included = node.avgpool2d_param.zero_padding_included

        # count_include_invalid
        xnode.count_include_invalid = node.avgpool2d_param.count_include_invalid

        if not xnode.is_global:
            # strides
            xnode.strides = list(node.avgpool2d_param.strides)
            # pad_mode
            xnode.pad_mode = PadMode[openir.PadMode.Name(node.avgpool2d_param.pad_mode)]
            # padding
            if xnode.pad_mode == PadMode.EXPLICIT:
                xnode.padding = list(node.avgpool2d_param.padding)
            # dilation: (n,h,w,c)
            xnode.dilation = [1] + list(node.avgpool2d_param.dilation) + [1]
            # round_mode
            round_mode = openir.RoundMode.Name(node.avgpool2d_param.round_mode)
            xnode.round_mode = RoundMode[round_mode]

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeAdaptiveAvgPoolND(XModelNode):
    """
    XModelNode AdaptiveAvgPoolND Protocol

    Derived from: PyTorch 1.3.1 AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d
    Reference: https://pytorch.org/docs/stable/nn.html?highlight=adaptive%20avgpool#torch.nn.AdaptiveAvgPool2d
    """

    def __init__(self, op_name: str, out_size: List[int]):
        assert op_name is not None, "'op_name' should not be None."
        assert out_size is not None, "'out_size' should not be None."
        assert len(out_size) == 2, "'out_size' shoud specify the dimensions of H and W."
        XModelNode.__init__(self, op_name, "adaptiveavgpoolnd")
        # out_size: [H, W]
        self.__out_size = out_size

    @property
    def out_size(self) -> List[int]:
        return self.__out_size

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("adaptiveavgpoolnd")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        raise NotImplementedError("adaptiveavgpoolnd")


class XModelNodeElemOp(XModelNode):
    """
    Base class of elementwise ops
    """

    def __init__(self, op_name: str, op_type: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, op_type)
        self.layout_type = LayoutType.INSENSITIVE


class XModelNodeElemAdd(XModelNodeElemOp):
    """
    XModelNode ElemAdd Protocol

    Derived from:

    Caffe eltwise layer
    https://caffe.berkeleyvision.org/tutorial/layers/eltwise.html

    Tensorflow tf.math.add
    https://www.tensorflow.org/api_docs/python/tf/math/add
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNodeElemOp.__init__(self, op_name, "elemadd")
        # caffe coeff: alpha[i] * input[i]
        self.__alpha: List[float] = None
        self.layout_type = LayoutType.INSENSITIVE

    @property
    def alpha(self) -> List[float]:
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha: List[float]):
        self.__alpha = alpha

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert self.inputs_tensor is not None and len(self.inputs_tensor) >= 2
        assert all(
            [
                layout.name == in_tensor.data_format.name
                for in_tensor in self.inputs_tensor
            ]
        ), f"[ERROR] xnnc {self.op_type} requires that the layout of all inputs must be same as 'layout' argument: op name: {self.op_name}."

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # check dimenstion sequences of the inputs
        curr_seq_dims_in_a = None
        for curr_seq_dims_in_b in self.sequence_dims[self.layout]["in"]:
            if curr_seq_dims_in_a == None:
                curr_seq_dims_in_a = curr_seq_dims_in_b
                continue
            if len(curr_seq_dims_in_a) > 1 and len(curr_seq_dims_in_b) > 1:
                assert (
                    curr_seq_dims_in_a == curr_seq_dims_in_b
                ), f"[ERROR] Mis-matched dimension sequences of inputs (xnnc ElemMul): {self.sequence_dims[self.layout]['in']}. Op name: {self.op_name}."

        # compute outputs_tensor
        out = None
        for i in self.inputs_tensor:
            if out == None:
                out = i
                continue
            out = XTensorCompute.elem_add(out, i)
        self.outputs_tensor = [out]

        # compute the dimension sequence of the output
        if curr_seq_dims_in_a == curr_seq_dims_in_b:
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_in_a]
        elif len(curr_seq_dims_in_a) == 1:
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_in_b]
        elif len(curr_seq_dims_in_b) == 1:
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_in_a]
        else:
            raise ValueError(
                f"[ERROR] failed to compute the dimension sequence of the output (xnnc ElemAdd): {curr_seq_dims_in_a}, {curr_seq_dims_in_b}."
            )

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeElemAdd(node.name)
        xnode.alpha = list(node.elem_add_param.coeffs)

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeElemAddn(XModelNodeElemOp):
    """
    XModelNode ElemAddn Protocol

    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNodeElemOp.__init__(self, op_name, "elemadd")
        self.layout_type = LayoutType.INSENSITIVE


    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) >= 2
        ), f"[ERROR] xnnc ElemAdd requires two inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert all(
            [
                layout.name == in_tensor.data_format.name
                for in_tensor in self.inputs_tensor
            ]
        ), f"[ERROR] xnnc {self.op_type} requires that the layout of all inputs must be same as 'layout' argument: op name: {self.op_name}."

        # update layout
        if self.layout != layout.name:
            assert (
                layout.name
                == self.inputs_tensor[0].data_format.name
                == self.inputs_tensor[1].data_format.name
            )
            self.layout = layout.name

        # check dimenstion sequences of two inputs
        if all([len(x)>1 for x in self.sequence_dims[self.layout]["in"]]):
            assert 1 == len(set([len(x) for x in self.sequence_dims[self.layout]["in"]])) , f"[ERROR] Mis-matched dimension sequences of inputs (xnnc ElemAdd). Op name: {self.op_name}."

        # compute outputs_tensor
        c = self.inputs_tensor[0]
        for x in self.inputs_tensor[1:]:
            c = XTensorCompute.elem_add(c, x)
        self.outputs_tensor = [c]

        # compute the dimension sequence of the output
        if self.sequence_dims[self.layout]["in"][1:] == self.sequence_dims[self.layout]["in"][:-1]:
            self.sequence_dims[self.layout]["out"] = [self.sequence_dims[self.layout]["in"][0]]
        else:
            raise ValueError(
                f"[ERROR] failed to compute the dimension sequence of the output (xnnc ElemAdd)."
            )


class XModelNodeElemNegative(XModelNodeElemOp):
    """
    XModelNode ElemNegative Protocol

    Derived from:

    TensorFlow tf.math.negative
    https://www.tensorflow.org/api_docs/python/tf/math/negative
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNodeElemOp.__init__(self, op_name, "elemnegative")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc ElemNegative requires a single input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        self.outputs_tensor = self.inputs_tensor

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("elemnegative")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        raise NotImplementedError("elemnegative")


class XModelNodeElemMul(XModelNodeElemOp):
    """
    XModelNode ElemMul Protocol

    Derived from:

    Caffe Eltwise Layer
    https://caffe.berkeleyvision.org/tutorial/layers/eltwise.html

    TensorFlow tf.math.multiply
    https://www.tensorflow.org/api_docs/python/tf/math/multiply
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNodeElemOp.__init__(self, op_name, "elemmul")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert self.inputs_tensor is not None and len(self.inputs_tensor) >= 2
        assert all(
            [
                layout.name == in_tensor.data_format.name
                for in_tensor in self.inputs_tensor
            ]
        ), f"[ERROR] xnnc {self.op_type} requires that the layout of all inputs must be same as 'layout' argument: op name: {self.op_name}."

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # check dimenstion sequences of the inputs
        curr_seq_dims_in_a = None
        for curr_seq_dims_in_b in self.sequence_dims[self.layout]["in"]:
            if curr_seq_dims_in_a == None:
                curr_seq_dims_in_a = curr_seq_dims_in_b
                continue
            if len(curr_seq_dims_in_a) > 1 and len(curr_seq_dims_in_b) > 1:
                assert (
                    curr_seq_dims_in_a == curr_seq_dims_in_b
                ), f"[ERROR] Mis-matched dimension sequences of inputs (xnnc ElemMul): {self.sequence_dims[self.layout]['in']}. Op name: {self.op_name}."

        # compute outputs_tensor
        a = None
        for b in self.inputs_tensor:
            if a == None:
                a = b
                continue
            out = XTensorCompute.elem_mul(a, b)
        self.outputs_tensor = [out]

        # compute the dimension sequence of the output
        if a.ndims == out.ndims:
            self.sequence_dims[self.layout]["out"] = [[x for x in curr_seq_dims_in_a]]
        elif b.ndims == out.ndims:
            self.sequence_dims[self.layout]["out"] = [[x for x in curr_seq_dims_in_b]]
        else:
            raise ValueError(
                f"[ERROR] failed to compute the dimension sequence of the output (xnnc ElemMul): {curr_seq_dims_in_a}, {curr_seq_dims_in_b}."
            )

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeElemMul(node.name)

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeElemSub(XModelNodeElemOp):
    """
    XModelNode ElemSub Protocol

    Derived from:

    TensorFlow tf.math.subtract
    https://www.tensorflow.org/api_docs/python/tf/math/subtract
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNodeElemOp.__init__(self, op_name, "elemsub")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 2
        ), f"[ERROR] xnnc {self.op_type} requires two inputs: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}."
        assert all(
            [
                layout.name == in_tensor.data_format.name
                for in_tensor in self.inputs_tensor
            ]
        ), f"[ERROR] xnnc {self.op_type} requires that the layout of all inputs must be same as 'layout' argument: op name: {self.op_name}."

        # check dimenstion sequences of two inputs
        curr_seq_dims_in_a, curr_seq_dims_in_b = self.sequence_dims[layout.name]["in"]
        if len(curr_seq_dims_in_a) > 1 and len(curr_seq_dims_in_b) > 1:
            assert (
                curr_seq_dims_in_a == curr_seq_dims_in_b
            ), f"[ERROR] Mis-matched dimension sequences of inputs (xnnc ElemSub): {curr_seq_dims_in_a}, {curr_seq_dims_in_b}"

        a, b = self.inputs_tensor
        c = XTensorCompute.elem_sub(a, b)
        self.outputs_tensor = [c]

        # compute the dimension sequence of the output
        if curr_seq_dims_in_a == curr_seq_dims_in_b:
            self.sequence_dims[layout.name]["out"] = [curr_seq_dims_in_a]
        elif len(curr_seq_dims_in_a) == 1:
            self.sequence_dims[layout.name]["out"] = [curr_seq_dims_in_b]
        elif len(curr_seq_dims_in_b) == 1:
            self.sequence_dims[layout.name]["out"] = [curr_seq_dims_in_a]
        else:
            raise ValueError(
                f"[ERROR] failed to compute the dimension sequence of the output (xnnc ElemSub): {curr_seq_dims_in_a}, {curr_seq_dims_in_b}."
            )

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("elemsub")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        raise NotImplementedError("elemsub")


class XModelNodeElemExp(XModelNodeElemOp):
    """
    XModelNode ElemExp Protocol

    Exponential of x element-wise.

    Derived from:

    TensorFlow tf.math.exp
    https://www.tensorflow.org/api_docs/python/tf/math/exp
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "elemexp")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc ElemExp requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        out_tensor = XTensorCompute.elem_exp(self.inputs_tensor[0])
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("elemexp")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        raise NotImplementedError("elemexp")


class XModelNodeElemRealDiv(XModelNodeElemOp):
    """
    XModelNode ElemRealDiv Protocol

    x / y element-wise for real types.

    Derived from:

    TensorFlow tf.realdiv
    https://www.tensorflow.org/api_docs/python/tf/realdiv
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "elemrealdiv")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 2
        ), f"[ERROR] xnnc ElemRealDiv requires two inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert all(
            [
                layout.name == in_tensor.data_format.name
                for in_tensor in self.inputs_tensor
            ]
        ), f"[ERROR] xnnc {self.op_type} requires that the layout of all inputs must be same as 'layout' argument: op name: {self.op_name}."

        # check dimenstion sequences of two inputs
        curr_seq_dims_in_a, curr_seq_dims_in_b = self.sequence_dims[layout.name]["in"]
        if len(curr_seq_dims_in_a) > 1 and len(curr_seq_dims_in_b) > 1:
            assert (
                curr_seq_dims_in_a == curr_seq_dims_in_b
            ), f"[ERROR] Mis-matched dimension sequences of inputs (xnnc ElemRealDiv): {curr_seq_dims_in_a}, {curr_seq_dims_in_b}"

        a, b = self.inputs_tensor
        c = XTensorCompute.elem_true_divide(a, b)
        self.outputs_tensor = [c]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        if curr_seq_dims_in_a == curr_seq_dims_in_b:
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_in_a]
        elif len(curr_seq_dims_in_a) == 1:
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_in_b]
        elif len(curr_seq_dims_in_b) == 1:
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_in_a]
        else:
            raise ValueError(
                f"[ERROR] failed to compute the dimension sequence of the output (xnnc ElemRealDiv): {curr_seq_dims_in_a}, {curr_seq_dims_in_b}."
            )

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("elemrealdiv")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        raise NotImplementedError("elemrealdiv")


class XModelNodeElemSquare(XModelNodeElemOp):
    """
    XModelNode ElemSquare Protocol

    Derived from:

    TensorFlow tf.math.square
    https://www.tensorflow.org/api_docs/python/tf/math/square
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "elemsquare")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc ElemSquare requires a single input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        self.outputs_tensor = [self.inputs_tensor[0].copy()]
        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeElemRSqrt(XModelNodeElemOp):
    """
    XModelNode ElemRSqrt Protocol

    Derived from:

    TensorFlow tf.math.rsqrt
    https://www.tensorflow.org/api_docs/python/tf/math/rsqrt
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "elemrsqrt")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc ElemRSqrt requires a single input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        self.outputs_tensor = [self.inputs_tensor[0].copy()]
        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeElemMax(XModelNodeElemOp):
    """
    XModelNode ElemMax Protocol

    Derived from:

    TensorFlow tf.math.maximum
    https://www.tensorflow.org/api_docs/python/tf/math/maximum
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "elemmax")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 2
        ), f"[ERROR] xnnc ElemMax requires two inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert all(
            [
                layout.name == in_tensor.data_format.name
                for in_tensor in self.inputs_tensor
            ]
        ), f"[ERROR] xnnc {self.op_type} requires that the layout of all inputs must be same as 'layout' argument: op name: {self.op_name}."

        tensor1, tensor2 = self.inputs_tensor
        out_tensor = XTensorCompute.elem_max(tensor1, tensor2)
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        self.sequence_dims[self.layout]["out"] = [
            self.sequence_dims[self.layout]["in"][0]
            if len(self.sequence_dims[self.layout]["in"][0])
            >= len(self.sequence_dims[self.layout]["in"][1])
            else self.sequence_dims[self.layout]["in"][1]
        ]


class XModelNodeElemMin(XModelNodeElemOp):
    """
    XModelNode ElemMin Protocol

    Derived from:

    TensorFlow tf.math.minimum
    https://www.tensorflow.org/api_docs/python/tf/math/minimum?hl=en
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "elemmin")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 2
        ), f"[ERROR] xnnc ElemMax requires two inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert all(
            [
                layout.name == in_tensor.data_format.name
                for in_tensor in self.inputs_tensor
            ]
        ), f"[ERROR] xnnc {self.op_type} requires that the layout of all inputs must be same as 'layout' argument: op name: {self.op_name}."

        tensor1, tensor2 = self.inputs_tensor
        out_tensor = XTensorCompute.elem_min(tensor1, tensor2)
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        self.sequence_dims[self.layout]["out"] = [
            self.sequence_dims[self.layout]["in"][0]
            if len(self.sequence_dims[self.layout]["in"][0])
            >= len(self.sequence_dims[self.layout]["in"][1])
            else self.sequence_dims[self.layout]["in"][1]
        ]


class XModelNodeElemRound(XModelNodeElemOp):
    """
    XModelNode ElemRound Protocol

    Derived from:

    TensorFlow tf.math.round
    https://www.tensorflow.org/api_docs/python/tf/math/round
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "elemround")

        # round_mode
        self.__round_mode: str = "PY3_ROUND"

    @property
    def round_mode(self) -> str:
        return self.__round_mode

    @round_mode.setter
    def round_mode(self, mode: str) -> NoReturn:
        assert mode is not None, "'mode' should not be None."
        assert isinstance(mode, str), "'mode' should be of str type."
        assert mode in ["std_round", "ceil", "floor", "py3_round"]
        self.__round_mode = mode

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc ElemRound requires a single input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        self.outputs_tensor = [self.inputs_tensor[0].copy()]
        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeElemFloor(XModelNodeElemOp):
    """
    XModelNode ElemFloor Protocol

    Derived from:

    TensorFlow tf.math.floor
    https://www.tensorflow.org/api_docs/python/tf/math/floor
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "elemfloor")

        # round_mode
        self.__round_mode: str = "FLOOR"

    @property
    def round_mode(self) -> str:
        return self.__round_mode

    @round_mode.setter
    def round_mode(self, mode: str) -> NoReturn:
        assert mode is not None, "'mode' should not be None."
        assert isinstance(mode, str), "'mode' should be of str type."
        assert mode in ["std_round", "ceil", "floor", "py3_round"]
        self.__round_mode = mode

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc ElemFloor requires a single input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        self.outputs_tensor = [self.inputs_tensor[0].copy()]
        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeElemTanh(XModelNodeElemOp):
    """
    XModelNode ElemTanh Protocol

    Derived from:

    TensorFlow tf.math.tanh
    https://www.tensorflow.org/api_docs/python/tf/math/tanh
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "elemtanh")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc ElemRSqrt requires a single input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        self.outputs_tensor = [self.inputs_tensor[0].copy()]
        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeDot(XModelNode):
    """
    XModelNode Dot Protocol

    Derived from:

    Caffe Inner Product
    Reference: https://caffe.berkeleyvision.org/tutorial/layers/innerproduct.html
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "dot")
        # Specifies the first index to be lumped into a single inner product computation.
        self.__axis: int = None
        self.__weights: XTensor = None
        self.__bias: XTensor = None
        self.__bias_term: bool = False
        # True, the transpose of the weight matrix will be used in any operation;
        # otherwise, False.
        self.__transpose: bool = False
        # quantization info for weights
        self.__qweights: Dict[str, Optional[int]] = {
            "bit_width": None,
            "quantize_pos": None,
            "signed": True,
            "round_mode": None,
        }
        # quantization info for bias
        self.__qbias: Dict[str, Optional[int]] = {
            "bit_width": None,
            "quantize_pos": None,
            "signed": True,
            "round_mode": None,
        }
        # num of output channels
        self.__num_output: int = 0
        self.layout_type = LayoutType.TOLERANT

    @property
    def axis(self) -> int:
        return self.__axis

    @axis.setter
    def axis(self, axis: int) -> NoReturn:
        assert axis is not None and isinstance(
            axis, int
        ), "'axis' should be an positive integer."
        self.__axis = axis

    @property
    def num_output(self) -> int:
        return self.__num_output

    @num_output.setter
    def num_output(self, value: int) -> NoReturn:
        assert value is not None, "'value' should not be None."
        assert isinstance(value, int), "'value' should be of int type."
        assert value > 0, "'value' should be greater than 0."
        self.__num_output = value

    @property
    def weights(self) -> XTensor:
        return self.__weights

    @weights.setter
    def weights(self, weights: XTensor):
        assert weights is not None, "'weights' should not be None."
        assert isinstance(weights, XTensor), "'weights' should be an XTensor object."
        self.__weights = weights

    @property
    def bias(self) -> XTensor:
        return self.__bias

    @bias.setter
    def bias(self, bias: XTensor):
        assert bias is not None, "'bias' should not be None."
        assert isinstance(bias, XTensor), "'bias' should be an XTensor object."
        self.__bias = bias

    @property
    def bias_term(self) -> bool:
        return self.__bias_term

    @bias_term.setter
    def bias_term(self, bias_term: bool):
        self.__bias_term = bias_term

    @property
    def quant_weights(self) -> Dict[str, Optional[int]]:
        return self.__qweights

    @property
    def quant_bias(self) -> Dict[str, Optional[int]]:
        return self.__qbias

    @property
    def transpose(self) -> bool:
        return self.__transpose

    @transpose.setter
    def transpose(self, flag: bool) -> NoReturn:
        assert flag is not None, "'flag' should not be None."
        assert isinstance(flag, bool), "'flag' should be a bool value."
        self.__transpose = flag

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Dot requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        if self.host.origin == "caffe":
            assert self.inputs_tensor[0].ndims == 4
            N = self.inputs_tensor[0].shape[0]
            if layout == Layout.NCHW:
                out_tensor = XTensor(
                    np.zeros(
                        (N, self.num_output, 1, 1), dtype=self.inputs_tensor[0].dtype
                    ),
                    format=self.inputs_tensor[0].data_format,
                )
            else:
                out_tensor = XTensor(
                    np.zeros(
                        (N, 1, 1, self.num_output), dtype=self.inputs_tensor[0].dtype
                    ),
                    format=self.inputs_tensor[0].data_format,
                )
            self.outputs_tensor = [out_tensor]

            if self.layout != layout.name:
                # update layout
                self.layout = layout.name

                # update weights
                if layout == Layout.NHWC:
                    _, H, W, C = self.inputs_tensor[0].shape
                    N_w, _ = self.weights.shape
                    self.weights = (
                        self.weights.reshape([N_w, C, H, W])
                        .transpose([0, 2, 3, 1])
                        .reshape([N_w, -1])
                    )
                else:
                    _, C, H, W = self.inputs_tensor[0].shape
                    N_w, _ = self.weights.shape
                    self.weights = (
                        self.weights.reshape([N_w, H, W, C])
                        .transpose([0, 3, 1, 2])
                        .reshape([N_w, -1])
                    )
                self.weights.data_format = DataFormat[self.layout]

                # update bias
                if self.bias_term and self.bias:
                    self.bias.data_format = DataFormat[self.layout]

            self.sequence_dims[self.layout]["out"] = [
                self.sequence_dims[self.layout]["in"][0]
            ]

        elif self.host.origin == "tensorflow2":
            # update layout
            if self.layout != layout.name:
                self.layout = layout.name

            if self.layout == Layout.NCHW.name:
                if self.inputs_tensor[0].ndims == 4:
                    N, _, H, W = self.inputs_tensor[0].shape
                    out_tensor = XTensor(
                        np.zeros(
                            (N, self.num_output, H, W),
                            dtype=self.inputs_tensor[0].dtype,
                        ),
                        format=self.inputs_tensor[0].data_format,
                    )
                    self.sequence_dims[self.layout]["out"] = [
                        self.sequence_dims[self.layout]["in"][0]
                    ]

                elif self.inputs_tensor[0].ndims == 2:
                    N, _ = self.inputs_tensor[0].shape
                    out_tensor = XTensor(
                        np.zeros(
                            (N, self.num_output, 1, 1),
                            dtype=self.inputs_tensor[0].dtype,
                        ),
                        format=self.inputs_tensor[0].data_format,
                    )
                    self.sequence_dims[self.layout]["out"] = [[0, 2, 3, 1]]

                else:
                    raise ValueError(
                        f"[ERROR] Unsupported inputs tensor with ndims: {self.inputs_tensor[0].ndims}"
                    )

            else:
                if self.inputs_tensor[0].ndims == 4:
                    N, H, W, _ = self.inputs_tensor[0].shape
                    out_tensor = XTensor(
                        np.zeros(
                            (N, H, W, self.num_output),
                            dtype=self.inputs_tensor[0].dtype,
                        ),
                        format=self.inputs_tensor[0].data_format,
                    )
                    self.sequence_dims[self.layout]["out"] = [
                        self.sequence_dims[self.layout]["in"][0]
                    ]

                elif self.inputs_tensor[0].ndims == 2:
                    N, _ = self.inputs_tensor[0].shape
                    out_tensor = XTensor(
                        np.zeros(
                            (N, 1, 1, self.num_output),
                            dtype=self.inputs_tensor[0].dtype,
                        ),
                        format=self.inputs_tensor[0].data_format,
                    )
                    self.sequence_dims[self.layout]["out"] = [[0, 1, 2, 3]]

                else:
                    raise ValueError(
                        f"[ERROR] Unsupported inputs tensor with ndims: {self.inputs_tensor[0].ndims}"
                    )
            self.outputs_tensor = [out_tensor]

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("dot")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        raise NotImplementedError("dot")


class XModelNodeSoftmax(XModelNode):
    """
    XModelNode Softmax Protocol

    Derived from:

    Caffe Softmax Layer
    https://caffe.berkeleyvision.org/tutorial/layers/softmax.html

    TensorFlow tf.nn.softmax
    https://www.tensorflow.org/api_docs/python/tf/nn/softmax
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "softmax")
        # For Caffe, default 1; for TensorFlow, default -1.
        self._axis: int = 1
        self.layout_type = LayoutType.RECONSTRUCTED

    @property
    def axis(self) -> int:
        return self._axis

    @axis.setter
    def axis(self, axis: int):
        self._axis = axis

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Softmax requires one input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        in_tensor = self.inputs_tensor[0]

        if self.axis < 0:
            self.axis += in_tensor.ndims
        assert (
            self.axis < in_tensor.ndims
        ), f"[ERROR] Invalid 'axis' value in xnnc Softmax: expected: in range of [0, {in_tensor.ndims - 1}], actual: {self.axis}."

        if self.layout != layout.name:
            if not self.sequence_dims[layout.name]["out"]:
                # update axis
                seq = self.sequence_dims[layout.name]["in"][0]
                self.axis = seq.index(self.axis)
            else:
                # update axis
                out_seq_before_change = self.sequence_dims[layout.name]["out"][0]
                in_seq = self.sequence_dims[self.layout]["in"][0]
                self.axis = out_seq_before_change.index(in_seq[self.axis])
            # update layout
            self.layout = layout.name

        self.outputs_tensor = self.inputs_tensor
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeSoftmax(node.name)

        # axis
        xnode.axis = node.softmax_param.axis

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeConcat(XModelNode):
    """
    XModelNode Concat Protocol

    Only support 4-dimension input tensors with either "NCHW" or "NHWC" layout

    Derived from:

    Caffe concat layer
    https://caffe.berkeleyvision.org/tutorial/layers/concat.html

    TensorFlow tf.concat
    https://www.tensorflow.org/api_docs/python/tf/concat
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "concat")
        # the axis dimension along which to do concatenation
        self.__axis: int = None
        # internal field: records the dims expanded
        self.__expand_dims = []
        self.layout_type = LayoutType.DEPENDENT

    @property
    def axis(self) -> int:
        return self.__axis

    @axis.setter
    def axis(self, axis: int):
        self.__axis = axis

    def infer_shape(self, layout: Layout) -> NoReturn:
        """

        [Algorithm for layout change]

        if (there is a potential_axis, on which the dimensions are different):
            if self.axis != potential_axis:
                self.axis = potential axis
        else:
            if (caffe model):
                if in_tensors.ndims != 4:
                    raise Unsupported
                if (layout changes):
                    update axis according to the new layout

        """
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) > 1
        ), f"[ERROR] xnnc Concat requires two or more inputs: actual: {len(self.inputs_tensor)}."

        rank = self.inputs_tensor[0].ndims
        assert all(
            [in_tensor.ndims == rank for in_tensor in self.inputs_tensor]
        ), f"[ERROR] xnnc Concat requires that the rank of all inputs must be same: op name: {self.op_name}."

        assert all(
            [
                layout.name == in_tensor.data_format.name
                for in_tensor in self.inputs_tensor
            ]
        ), f"[ERROR] xnnc Concat requires that the layout of all inputs must be same as 'layout' argument: op name: {self.op_name}."

        assert self.axis is not None and isinstance(
            self.axis, int
        ), "[ERROR] xnnc concat requires that the axis should be a positive integer."

        # if axis is negative, convert it to be positive
        if self.axis < 0:
            self.axis += rank

        if self.layout != layout.name:
            if not self.sequence_dims[layout.name]["out"]:
                # update axis
                seq = self.sequence_dims[layout.name]["in"][0]
                self.axis = seq.index(self.axis)
            else:
                # update axis
                out_seq_before_change = self.sequence_dims[layout.name]["out"][0]
                in_seq = self.sequence_dims[self.layout]["in"][0]
                self.axis = out_seq_before_change.index(in_seq[self.axis])
            # update layout
            self.layout = layout.name

        # compute outputs_tensor
        out_tensor = XTensorCompute.concat(self.inputs_tensor, axis=self.axis)
        self.outputs_tensor = [out_tensor]

        # compute the dimension sequence of the output
        self.sequence_dims[self.layout]["out"] = [
            [x for x in self.sequence_dims[self.layout]["in"][0]]
        ]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeConcat(node.name)

        # axis
        xnode.axis = node.concat_param.axis

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodePad(XModelNode):
    """
    XModelNode Pad Protocol

    Derived from:

    TensorFlow tf.pad
    https://www.tensorflow.org/api_docs/python/tf/pad
    """

    def __init__(
        self,
        op_name: str,
        pad_num: List[int],
        mode: str = "constant",
        constant_values: List[Union[int, float]] = None,
    ):
        """
        Parameters:
            op_name: name of operator.
            pad_num: the number of values the target tensor is padded with before/after each dimension. The length of pad_num is two times of the dimensions of target tensor.
            mode: the padding mode, which should be one of ["constant", "reflect", "symmetric", "edge"]. The default value is 'constant'.
            constant_values: a list of constant values to pad, of which the length is 4. The default value is [0, 0, 0, 0]. This argument will be used if 'mode' is set to 'constant'.
        """
        assert op_name is not None, "The argument 'op_name' must be a string value."
        assert (
            pad_num is not None
        ), "The argument 'pad_num' should be a list of length 8 in the form of [pad_N_before, pad_N_after, pad_C_before, pad_C_after, pad_H_before, pad_H_after, pad_W_before, pad_W_after]."
        assert (
            len(pad_num) == 8
        ), "The argument 'pad_num' should be a list of length 8 in the form of [pad_N_before, pad_N_after, pad_C_before, pad_C_after, pad_H_before, pad_H_after, pad_W_before, pad_W_after]."
        mode = mode.lower()
        assert mode in [
            "constant",
            "reflect",
            "symmetric",
            "edge",
        ], f"[ERROR] Unsupported pad mode: {mode}. The expected value is 'constant', 'reflect', 'symmetric', or 'edge'."

        XModelNode.__init__(self, op_name, "pad")
        # padding layout: ((pad_N_before, pad_N_after), (pad_C_before, pad_C_after)), (pad_H_before, pad_H_after), (pad_W_before, pad_W_after)
        self.__padding: List[int] = pad_num
        # One of 'constant', 'reflect', 'symmetric', 'edge'
        self.__mode: str = mode
        self.__constant_values: List[Union[int, float]] = constant_values
        self.layout_type = LayoutType.TOLERANT

    @property
    def padding(self) -> List[int]:
        return self.__padding

    @padding.setter
    def padding(self, padding: List[int]) -> NoReturn:
        assert (
            padding is not None and isinstance(padding, list) and len(padding) == 8
        ), "'padding' should be a list of 8 positive integers. For 'NCHW' layout, it should be [pad_n_before, pad_n_after, pad_c_before, pad_c_after, pad_h_before, pad_h_after, pad_w_before], pad_w_after; for 'NHWC' layout, it should be [pad_n_before, pad_n_after, pad_h_before, pad_h_after, pad_w_before, pad_c_before, pad_c_after]."
        self.__padding = padding

    @property
    def pad_mode(self) -> str:
        return self.__mode

    @pad_mode.setter
    def pad_mode(self, mode: str):
        assert mode in [
            "constant",
            "reflect",
            "symmetric",
            "edge",
        ], "'mode' should be one of ['constant', 'reflect', 'symmetric', 'edge']."
        self.__mode = mode

    @property
    def constant_values(self) -> List[Union[int, float]]:
        return self.__constant_values

    @constant_values.setter
    def constant_values(self, constant_values: List[Union[int, float]]) -> NoReturn:
        self.__constant_values = constant_values

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Pad requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        in_tensor = self.inputs_tensor[0]

        # update padding
        if self.layout != layout.name:
            assert layout.name == in_tensor.data_format.name

            if self.layout == Layout.NHWC.name:
                self.padding = self.padding[:2] + self.padding[-2:] + self.padding[2:6]
                if self.pad_mode == "constant":
                    self.constant_values = (
                        self.constant_values[:2]
                        + self.constant_values[-2:]
                        + self.constant_values[2:6]
                    )
            else:
                self.padding = self.padding[:2] + self.padding[-4:] + self.padding[2:4]
                if self.pad_mode == "constant":
                    self.constant_values = (
                        self.constant_values[:2]
                        + self.constant_values[-4:]
                        + self.constant_values[2:4]
                    )

            # update layout
            self.layout = layout.name

        pad_width = tuple(
            [
                (self.padding[i], self.padding[i + 1])
                for i in range(0, len(self.padding), 2)
            ]
        )

        # compute outputs_tensor
        if self.pad_mode == "constant":
            assert self.constant_values is not None
            constant_values = tuple(
                [
                    (self.constant_values[i], self.constant_values[i + 1])
                    for i in range(0, len(self.constant_values), 2)
                ]
            )
            out_tensor = XTensorCompute.pad(
                in_tensor,
                pad_width,
                mode="constant",
                constant_values=constant_values,
            )
        else:
            out_tensor = XTensorCompute.pad(in_tensor, pad_width, mode=self.pad_mode)

        self.outputs_tensor = [out_tensor]
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        # padding: (pad_top, pad_bottom, pad_left, pad_right)
        padding = node.pad_param.padding
        # pad_mode
        pad_mode = node.pad_param.mode
        # constant_values
        constant_values = None
        if pad_mode == "constant":
            constant_values = list(node.pad_param.constant_values)

        xnode = XModelNodePad(node.name, padding, pad_mode, constant_values)

        # quantization
        if node.quantized:
            xnode.is_quantized = True
            quant_config = node.input_param.quant_config
            # quant_out
            XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeMean(XModelNode):
    """
    XModelNode Mean Protocol

    Derived from:

    tf.math.reduce_mean
    https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "mean")
        # The dimensions to reduce. If None, reduces all dimensions
        self.__axis = None
        # If True, retains reduced dimensions with length 1.
        self.__keep_dims: bool = False
        # inputs_tensor
        self.__inputs_tensor: List[np.dtype] = []
        # the value depends on keep_dims value
        # if keep_dims is True, it's LayoutType.DEPENDENT
        # otherwise, LayoutType.RECONSTRUCTED
        self.layout_type = LayoutType.RECONSTRUCTED

    @property
    def axis(self) -> List[int]:
        return self.__axis

    @axis.setter
    def axis(self, axis: List[int]):
        self.__axis = axis

    @property
    def keep_dims(self) -> bool:
        return self.__keep_dims

    @keep_dims.setter
    def keep_dims(self, keepdims: bool):
        self.__keep_dims = keepdims
        # set layout type
        self.layout_type = (
            LayoutType.DEPENDENT if self.__keep_dims else LayoutType.RECONSTRUCTED
        )

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Mean requires one input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        # update the axis property if layout changes
        if self.layout != layout.name:
            prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
            curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
            new_axis = []
            for i in self.axis:
                x = prev_seq_dims_in[i]
                new_axis.append(curr_seq_dims_in.index(x))
            self.axis = new_axis
            # update layout
            self.layout = layout.name

        # compute outputs_tensor
        in_tensor = self.inputs_tensor[0]
        out_tensor = XTensorCompute.mean(
            in_tensor, axis=self.axis, keepdims=self.keep_dims
        )
        self.outputs_tensor = [out_tensor]

        # compute the dimension sequence of the output
        if self.keep_dims:
            self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout][
                "in"
            ]
        else:
            curr_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
            curr_seq_dims_out = []
            for i, v in enumerate(curr_seq_dims_in):
                if i not in self.axis:
                    curr_seq_dims_out.append(v)
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_out]

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("mean")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeMean(node.name)

        # axis
        xnode.axis = list(node.mean_param.axis)
        # keep_dims
        xnode.keep_dims = node.mean_param.keep_dims

        if node.quantized:
            # quantization
            quant_config = node.quant_config
            # quant_out
            XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeSqueeze(XModelNode):
    """
    XModelNode Squeeze Protocol

    Derived from:

    TensorFlow tf.squeeze
    https://www.tensorflow.org/api_docs/python/tf/squeeze
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "squeeze")
        # the dimensions to be squeezed. The dimension index starts at 0.
        # If None, it removes all size 1 dimensions.
        self.__axis: List[int] = None
        self.layout_type = LayoutType.DEPENDENT

    @property
    def axis(self) -> List[int]:
        return self.__axis

    @axis.setter
    def axis(self, squeeze_dims: List[int]):
        """
        Parameters:
            squeeze_dims: list of dims to be squeezed.
        """
        self.__axis = squeeze_dims

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Squeeze requires only one input: actual: {len(self.inputs_tensor_shape)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the data format of input tensor should be same as the 'layout' argument: expected:{layout.name}, actual:{self.inputs_tensor[0].data_format.name}."

        # update axis
        if self.layout != layout.name:
            # curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
            # self.axis = [curr_seq_dims_in.index(dim) for dim in self.axis]

            curr_seq_dims_in = self.sequence_dims["NHWC"]["in"][0]
            if layout == Layout.NCHW:
                self.axis = [curr_seq_dims_in[dim] for dim in self.axis]
            else:
                self.axis = [curr_seq_dims_in.index(dim) for dim in self.axis]

        for index, item in enumerate(self.axis):
            if item < 0:
              assert abs(item) <= len(self.sequence_dims[self.layout]["in"][0])
              self.axis[index] = len(self.sequence_dims[self.layout]["in"][0])+item

        # compute outputs_tensor
        out_tensor = XTensorCompute.squeeze(
            self.inputs_tensor[0], axis=tuple(self.axis) if self.axis else None
        )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        self.sequence_dims[self.layout]["out"] = [[j for i, j in enumerate(self.sequence_dims[self.layout]["in"][0]) if i not in self.axis]]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeSqueeze(node.name)

        xnode.axis = list(node.squeeze_param.axis)

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeReshape(XModelNode):
    """
    XModelNode Reshape Protocol

    Derived from:

    Caffe Reshape Layer
    https://caffe.berkeleyvision.org/tutorial/layers/reshape.html

    TensorFlow tf.reshape
    https://www.tensorflow.org/api_docs/python/tf/reshape

    TensorFlow tf.keras.layers.Reshape
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "reshape")
        # new shape. Notice: for tensorflow2, it indicates 'target_shape' defined in keras.
        self.__shape: List[int] = None
        self.layout_type = LayoutType.RECONSTRUCTED

    @property
    def shape(self) -> List[int]:
        return self.__shape

    @shape.setter
    def shape(self, shape: Optional[List[int]]) -> NoReturn:
        assert shape is None or isinstance(
            shape, list
        ), "'shape' should be None or a list of integers."
        self.__shape = shape

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and 1 <= len(self.inputs_tensor) <= 2
        ), f"[ERROR] xnnc Reshape requires one or two inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        in_tensor = self.inputs_tensor[0]

        # ! DO NOT REMOVED: check dimension trace of inputs
        # if len(self.sequence_dims[self.layout]["in"]) > 1:
        #     self.sequence_dims[self.layout]["in"].pop(-1)

        # get the shape info from parent branch
        if self.shape is None or len(self.shape) == 0:
            _, shape_id = self.bottom
            pnode_shape = self.host.get_xnode_by_name(shape_id)
            assert (
                pnode_shape is not None
            ), f"[ERROR] Not found node for shape info: name: {shape_id}"

            if pnode_shape.op_type in ["stack", "shape"]:
                self.shape = self.inputs_tensor[1].tolist()
            elif pnode_shape.op_type == "const":
                # extract shape info from the branch reaching to Const node
                shape = pnode_shape.tensor.tolist()
                assert all(
                    [x >= 0 or x == -1 for x in shape]
                ), f"[ERROR] Invalid shape: {shape}, op type: {self.op_type}, name: {self.op_name}"
                for i in range(len(shape)):
                    if shape[i] == 0:
                        shape[i] = in_tensor.shape[i]
                self.shape = shape
        elif self.host.origin == "tensorflow2":
            # keras case: (batch_size,) + target_shape
            N = self.inputs_tensor[0].shape[0]
            if N != self.shape[0]:
                self.shape[0] = N

        if self.layout != layout.name:
            if in_tensor.ndims > len(self.shape):  # case 1: contract dimensions
                prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
                curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]

                if prev_seq_dims_in == curr_seq_dims_in:
                    self.sequence_dims[layout.name]["out"] = self.sequence_dims[
                        self.layout
                    ]["out"]
                else:
                    # * step 1: compute the mapping relations of dimension sequences of doing reshape
                    # in the original layout.

                    # get the shape info of the inputs in the original layout
                    curr_in_shape = in_tensor.shape
                    lookup = {}
                    for i in range(in_tensor.ndims):
                        lookup[curr_seq_dims_in[i]] = curr_in_shape[i]
                    prev_in_shape = []
                    for i in range(in_tensor.ndims):
                        prev_in_shape.append(lookup[i])

                    # get the dimension mapping from inputs to outputs in the original layout
                    i = j = 0
                    dims_mapping = {}
                    queue = []
                    while i < len(prev_in_shape) and j < len(self.shape):
                        if prev_in_shape[i] == self.shape[j]:
                            queue.append(i)
                            dims_mapping[tuple(queue)] = j
                            i += 1
                            j += 1
                            queue = []
                        else:
                            if i + 1 == len(prev_in_shape):
                                raise ValueError(
                                    f"[ERROR] Failed to perform dimension match between input shape {in_tensor.shape} and the shape property {self.shape}."
                                )
                            prev_in_shape[i + 1] *= prev_in_shape[i]
                            queue.append(i)
                            i += 1
                    assert i == len(prev_in_shape) and j == len(
                        self.shape
                    ), f"[ERROR] Failed to perform dimension match between input shape {in_tensor.shape} and the shape property {self.shape}."

                    # * step 2: based on the result in step1, with both the shape info of inputs tensor
                    # and the corresponding dimension sequence of the input in the new layout, compute
                    # the dimension sequence of the outputs tensor first, then update the values of
                    # the shape property.

                    # compute the dimension sequence of the shape property in the new layout
                    curr_seq_dims_out = []
                    keys = []
                    for i in range(in_tensor.ndims):
                        key = curr_seq_dims_in[i]
                        if key in dims_mapping:
                            curr_seq_dims_out.append(dims_mapping[key])
                        else:
                            keys.append(key)
                            if len(keys) > 1:
                                if keys in dims_mapping:
                                    curr_seq_dims_out.append(dims_mapping[keys])
                                    keys = []
                    assert len(curr_seq_dims_out) == len(self.shape)
                    self.sequence_dims[layout.name]["out"] = [curr_seq_dims_out]

                    # * step 3: update the shape property
                    prev_seq_dims_out = self.sequence_dims[self.layout]["out"][0]
                    idx_list = [prev_seq_dims_out.index(x) for x in curr_seq_dims_out]
                    self.shape = [self.shape[x] for x in idx_list]

            elif in_tensor.ndims < len(self.shape):  # case 2: expand dimensions
                prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
                curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]

                if prev_seq_dims_in == curr_seq_dims_in:
                    self.sequence_dims[layout.name]["out"] = self.sequence_dims[
                        self.layout
                    ]["out"]
                else:
                    # * step 1: compute the mapping from the inputs dimension sequence to
                    # the outputs dimension sequence in the original layout.

                    # get the shape info of the inputs and outputs in the original layout
                    prev_in_shape = []
                    prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
                    curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
                    for x in prev_seq_dims_in:
                        idx = curr_seq_dims_in.index(x)
                        prev_in_shape.append(in_tensor.shape[idx])
                    prev_out_shape = [x for x in self.shape]

                    # build the dimension mapping from inputs to outputs
                    mapping = {}
                    i = j = 0
                    queue = []
                    while i < len(prev_in_shape) and j < len(prev_out_shape):
                        if prev_in_shape[i] == prev_out_shape[j]:
                            queue.append(j)
                            mapping[i] = queue
                            i += 1
                            j += 1
                            queue = []
                        else:
                            if j + 1 == len(prev_out_shape):
                                raise ValueError(
                                    f"[ERROR] Failed to perform dimension match between input shape {prev_in_shape} and the shape property {self.shape}."
                                )
                            prev_out_shape[j + 1] *= prev_out_shape[j]
                            queue.append(j)
                            j += 1
                    assert i == len(prev_in_shape) and j == len(
                        prev_out_shape
                    ), f"[ERROR] Failed to perform dimension match between input shape {prev_in_shape} and the shape property {self.shape}."

                    # * step 2: with the mapping from step 1, compute the dimension sequence
                    # of the outputs in the new layout.
                    curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
                    curr_seq_dims_out = []
                    for x in curr_seq_dims_in:
                        curr_seq_dims_out += mapping[x]
                    assert len(curr_seq_dims_out) == len(self.shape)
                    self.sequence_dims[layout.name]["out"] = [curr_seq_dims_out]

                    # * step 3: based on the shape info of the outputs in the original layout,
                    # along with the dimension sequence of the outputs in the new layout,
                    # infer the shape info of the output in the new layout, and update the
                    # shape property.
                    prev_seq_dims_out = self.sequence_dims[self.layout]["out"][0]
                    idx_list = [prev_seq_dims_out.index(x) for x in curr_seq_dims_out]
                    self.shape = [self.shape[x] for x in idx_list]

            else:  # case 3: no change on number of dimensions
                prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
                curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]

                if prev_seq_dims_in == curr_seq_dims_in:
                    self.sequence_dims[layout.name]["out"] = self.sequence_dims[
                        self.layout
                    ]["out"]
                else:
                    # * step 1: compute current dimension sequence of the output
                    curr_seq_dims_out = self.sequence_dims[layout.name]["in"][0]
                    self.sequence_dims[layout.name]["out"] = [curr_seq_dims_out]

                    # * step 2: update the shape property
                    prev_seq_dims_out = self.sequence_dims[self.layout]["out"][0]
                    idx_list = [prev_seq_dims_out.index(x) for x in curr_seq_dims_out]
                    self.shape = [self.shape[x] for x in idx_list]

        out_tensor = XTensorCompute.reshape(in_tensor, tuple(self.shape))
        self.outputs_tensor = [out_tensor]

        # update shape info if there is any negative dimension value
        for v in self.shape:
            if v < 0:
                self.shape = self.outputs_tensor[0].shape
                break

        if (
            self.layout == layout.name
            and self.sequence_dims[layout.name]["out"] is None
        ):
            if out_tensor.ndims == 4 and layout == Layout.NHWC:
                curr_seq_dims_out = [0, 2, 3, 1]
            else:
                curr_seq_dims_out = [x for x in range(len(self.shape))]
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_out]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeReshape(node.name)

        if len(node.inputs) == 1:
            xnode.shape = list(node.reshape_param.new_shape)

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeConst(XModelNode):
    """
    XModelNode Const Protocol
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "const")
        self.__tensor: XTensor = None
        self.layout_type = LayoutType.INSENSITIVE

    @property
    def tensor(self) -> XTensor:
        return self.__tensor

    @tensor.setter
    def tensor(self, tensor: XTensor) -> NoReturn:
        assert tensor is not None, "'tensor' should not be None."
        assert isinstance(tensor, XTensor), "'tensor' should be an XTensor object."
        self.__tensor = tensor

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert self.inputs_tensor is None or len(self.inputs_tensor) == 0

        # update layout and tensor
        if self.layout != layout.name:
            # update tensor
            assert self.tensor.data_format.name != layout.name
            if self.tensor.ndims == 4:
                if layout == Layout.NHWC:
                    # nchw -> nhwc
                    self.tensor = XTensorCompute.transpose(self.tensor, (0, 2, 3, 1))
                else:
                    # nhwc -> nchw
                    self.tensor = XTensorCompute.transpose(self.tensor, (0, 3, 1, 2))
            self.tensor.data_format = DataFormat[layout.name]
            # update layout
            self.layout = layout.name

        # compute outputs_tensor
        out_tensor = self.tensor
        self.outputs_tensor = [out_tensor]

        # compute the dimension sequence of the output
        if self.tensor.ndims == 4:
            if layout == Layout.NCHW:
                self.sequence_dims[self.layout]["out"] = [[0, 1, 2, 3]]
            else:
                self.sequence_dims[self.layout]["out"] = [[0, 2, 3, 1]]
        else:
            curr_seq_dims_out = [x for x in range(out_tensor.ndims)]
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_out]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeConst(node.name)
        xnode.tensor = XTensor.deserialize(node.const_param.tensor)

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs and outputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeFixNeuron(XModelNode):
    """
    XModelNode FixNeuron Protocol

    Designed for XIR.
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "fixneuron")
        self.layout_type = LayoutType.INSENSITIVE

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc {self.op_type} requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the data format of input tensor should be same as the 'layout' argument: expected:{layout.name}, actual:{self.inputs_tensor[0].data_format.name}."

        if self.layout != layout.name:
            self.layout = layout.name

        self.outputs_tensor = self.inputs_tensor
        self.sequence_dims[self.layout]["out"] = [
            x for x in self.sequence_dims[self.layout]["in"]
        ]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeFixNeuron(node.name)
        xnode.is_quantized = True

        # quantization
        quant_config = node.quant_config
        # quant_in
        XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
        # quant_out
        XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeStridedSlice(XModelNode):
    """
    XModelNode StridedSlice Protocol

    Derived from:

    TensorFlow tf.strided_slice
    https://www.tensorflow.org/api_docs/python/tf/strided_slice
    """

    def __init__(
        self,
        op_name: str,
        begin: List[int],
        end: List[int],
        strides: Optional[List[int]] = None,
    ):
        assert op_name is not None, "'op_name' should not be None."
        assert begin is not None, "'begin' should not be None."
        assert len(begin) > 0, "'begin' should have one or more integers."
        assert end is not None, "'end' should not be None."
        assert len(end) > 0, "'end' should have one or more integers."
        assert len(begin) == len(
            end
        ), "'begin' and 'end' should have exactly same number of integers."
        if strides is not None:
            assert len(strides) == len(
                begin
            ), "'strides' should have exactly same number of integers as 'begin'."
        XModelNode.__init__(self, op_name, "stridedslice")
        # start location of slicing (included)
        self.__begin: List[int] = begin
        # end location of slicing (excluded)
        self.__end: List[int] = end
        # strides of slicing
        self.__strides: List[int] = None
        if strides is None:
            self.__strides = [1] * len(self.__begin)
        else:
            self.__strides = strides
        self.__begin_mask: int = 0
        self.__end_mask: int = 0
        self.__ellipsis_mask: int = 0
        self.__new_axis_mask: int = 0
        self.__shrink_axis_mask: int = 0
        self.layout_type = LayoutType.RECONSTRUCTED

    @property
    def begin(self) -> List[int]:
        return self.__begin

    @property
    def end(self) -> List[int]:
        return self.__end

    @property
    def strides(self) -> List[int]:
        return self.__strides

    @property
    def begin_mask(self) -> int:
        return self.__begin_mask

    @begin_mask.setter
    def begin_mask(self, mask: int) -> NoReturn:
        self.__begin_mask = mask

    @property
    def end_mask(self) -> int:
        return self.__end_mask

    @end_mask.setter
    def end_mask(self, mask: int) -> NoReturn:
        self.__end_mask = mask

    @property
    def ellipsis_mask(self) -> int:
        return self.__ellipsis_mask

    @ellipsis_mask.setter
    def ellipsis_mask(self, mask: int) -> NoReturn:
        self.__ellipsis_mask = mask

    @property
    def new_axis_mask(self) -> int:
        return self.__new_axis_mask

    @new_axis_mask.setter
    def new_axis_mask(self, mask: int) -> NoReturn:
        self.__new_axis_mask = mask

    @property
    def shrink_axis_mask(self) -> int:
        return self.__shrink_axis_mask

    @shrink_axis_mask.setter
    def shrink_axis_mask(self, mask: int) -> NoReturn:
        self.__shrink_axis_mask = mask

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc StridedSlice requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the data format of input tensor should be same as the 'layout' argument: expected:{layout.name}, actual:{self.inputs_tensor[0].data_format.name}."
        assert (
            len(self.begin) == len(self.end) == len(self.strides)
        ), f"[ERROR] 'begin', 'end' and 'strides' must have the same size: actual: begin: {len(self.begin)}, end:{len(self.eng)}, strides:{len(self.strides)}"
        assert all(
            [x != 0 for x in self.strides]
        ), f"[ERROR] 'strides' entries must be non-zero: actual: {self.strides}."

        if self.ellipsis_mask != 0 or self.new_axis_mask != 0:
            raise NotImplementedError(
                f"[ERROR] Not implemented: begin_mask:{self.begin_mask}, end_mask:{self.end_mask}, ellipsis_mask:{self.ellipsis_mask}, new_axis_mask:{self.new_axis_mask}, shrink_axis_mask:{self.shrink_axis_mask}."
            )

        for k, v in enumerate(self.end):
            if v < 0:
                assert abs(v) < self.inputs_tensor[0].shape[k], f"[ERROR] 'end' index out of 'begin' range, begin: {self.begin}, end: {self.end}"
                v += self.inputs_tensor[0].shape[k]
                self.end[k] = v

        # compute outputs_tensor
        out_tensor = XTensorCompute.slice(
            self.inputs_tensor[0],
            begin=self.begin,
            end=self.end,
            step=self.strides,
            begin_mask=self.begin_mask,
            end_mask=self.end_mask,
            shrink_axis_mask=self.shrink_axis_mask,
        )
        self.outputs_tensor = [out_tensor]

        # compute the dimension sequence of the output
        if out_tensor.ndims == 4:
            self.sequence_dims[self.layout]["out"] = [
                x for x in self.sequence_dims[self.layout]["in"]
            ]

        else:
            self.sequence_dims[self.layout]["out"] = [
                [x for x in range(out_tensor.ndims)]
            ]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("XModelNodeStridedSlice")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        begin = list(node.strided_slice_param.begin)
        end = list(node.strided_slice_param.end)
        strides = list(node.strided_slice_param.strides)

        xnode = XModelNodeStridedSlice(node.name, begin, end, strides)

        # begin_mask
        xnode.begin_mask = node.strided_slice_param.begin_mask
        # end_mask
        xnode.end_mask = node.strided_slice_param.end_mask
        # ellipsis_mask
        xnode.ellipsis_mask = node.strided_slice_param.ellipsis_mask
        # new_axis_mask
        xnode.new_axis_mask = node.strided_slice_param.new_axis_mask
        # shrink_axis_mask
        xnode.shrink_axis_mask = node.strided_slice_param.shrink_axis_mask

        if node.quantized:
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodePermute(XModelNode):
    """
    XModelNode Permute Protocol

    Derived from:

    Caffe Permute Layer
    Reference: https://github.com/intel/caffe/blob/master/src/caffe/layers/permute_layer.cpp

    TensorFlow tf.transpose
    https://www.tensorflow.org/api_docs/python/tf/transpose
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "permute")
        self.__order: List[int] = None
        # If conjugate is True and input tensor dtype is either complex64
        # or complex128 then the values of a are transposed first and then conjugated.
        # In tensorflow, tf.math.conj(tf.transpose(input))
        self.__conjugate: bool = False
        # inputs_tensor
        self.__inputs_tensor: List[np.dtype] = []
        self.layout_type = LayoutType.DEPENDENT

    @property
    def order(self) -> List[int]:
        return self.__order

    @order.setter
    def order(self, order: List[int]) -> NoReturn:
        assert order is not None, "'order' should not be None."
        self.__order = order

    @property
    def conjugate(self) -> bool:
        return self.__conjugate

    @conjugate.setter
    def conjugate(self, flag: bool) -> NoReturn:
        assert flag is not None, "'flag' should not be None."
        assert isinstance(flag, bool), "'flag' should be a bool."
        self.__conjugate = flag

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Permute requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        in_tensor = self.inputs_tensor[0]

        # update order
        if self.layout != layout.name:
            if layout == Layout.NHWC:
                self.order = [
                    self.sequence_dims[layout.name]["in"][0].index(x)
                    for x in self.order
                ]
            else:
                self.order = [
                    self.sequence_dims[self.layout]["in"][0][x] for x in self.order
                ]

        # compute outputs_tensor
        out_tensor = XTensorCompute.transpose(in_tensor, axes=self.order)
        out_tensor.data_format = DataFormat[layout.name]
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        self.sequence_dims[self.layout]["out"] = [
            [self.sequence_dims[self.layout]["in"][0][x] for x in self.order]
        ]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodePermute(node.name)

        # order
        xnode.order = list(node.permute_param.order)

        # ! hardcode
        xnode.conjugate = False

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodePriorBox(XModelNode):
    """
    XModelNode PriorBox Protocol

    Derived from:

    Caffe PriorBoxLayer
    https://github.com/intel/caffe/blob/master/src/caffe/layers/prior_box_layer.cpp
    https://github.com/intel/caffe/blob/master/include/caffe/layers/prior_box_layer.hpp
    """

    def __init__(
        self, op_name: str, min_sizes: List[float], max_sizes: List[float]
    ) -> NoReturn:
        assert op_name is not None, "The argument 'op_name' must be a string value."
        assert min_sizes is not None, "'min_sizes' should not be None."
        assert max_sizes is not None, "'max_sizes' should not be None."
        XModelNode.__init__(self, op_name, "priorbox")
        # minimum box size (in pixels)
        self.__min_sizes: List[float] = min_sizes
        # maximum box size (in pixels)
        self.__max_sizes: List[float] = max_sizes
        # various of aspect ratios, default 1.0
        self.__aspect_ratio: List[float] = [1.0]
        # if true, will flip each aspect ratio
        self.__flip: bool = True
        # if true, will clip the prior so that it is within [0, 1]
        self.__clip: bool = False
        # variance for adjusting the prior bboxes
        self.__variance: List[float] = []
        # step size or [step_h, step_w]
        self.__step: List[float] = []
        # offset to the top left corner of each cell, default 0.5
        self.__offset: float = 0.5
        self.layout_type = LayoutType.TOLERANT

    @property
    def min_sizes(self) -> List[float]:
        return self.__min_sizes

    @property
    def max_sizes(self) -> List[float]:
        return self.__max_sizes

    @property
    def aspect_ratio(self) -> List[float]:
        return self.__aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, aspect_ratio: List[float]) -> NoReturn:
        assert aspect_ratio is not None, "'aspect_ratio' should not be None."
        self.__aspect_ratio = aspect_ratio

    @property
    def flip(self) -> bool:
        return self.__flip

    @flip.setter
    def flip(self, flip: bool) -> NoReturn:
        assert flip is not None, "'flip' should not be None."
        self.__flip = flip

    @property
    def clip(self) -> bool:
        return self.__clip

    @clip.setter
    def clip(self, clip: bool) -> NoReturn:
        assert clip is not None, "'clip' should not be None."
        self.__clip = clip

    @property
    def variance(self) -> List[float]:
        return self.__variance

    @variance.setter
    def variance(self, variance: List[float]) -> NoReturn:
        assert variance is not None, "'variance' should not be None."
        self.__variance = variance

    @property
    def step(self) -> List[float]:
        return self.__step

    @step.setter
    def step(self, step: List[float]) -> NoReturn:
        assert step is not None, "'step' should not be None."
        self.__step = step

    @property
    def offset(self) -> float:
        return self.__offset

    @offset.setter
    def offset(self, offset: float) -> NoReturn:
        assert offset is not None, "'offset' should not be None"
        self.__offset = offset

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        # according to the assertion from Caffe PriorBox
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) > 1
        ), f"[ERROR] xnnc PriorBox requires more than one inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert self.inputs_tensor[0].ndims == 4
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        if layout == Layout.NCHW:
            N, _, H, W = self.inputs_tensor[0].shape
        else:
            N, H, W, _ = self.inputs_tensor[0].shape

        # * preprocess aspect ratios
        aspect_ratios_ = [1.0]
        for ar in self.aspect_ratio:
            if not any([math.isclose(ar, x, rel_tol=1e-06) for x in aspect_ratios_]):
                aspect_ratios_.append(ar)
                if self.flip:
                    aspect_ratios_.append(1.0 / ar)
        # * compute output shape
        num_priors = len(aspect_ratios_) * len(self.min_sizes) + len(self.max_sizes)

        """
        (By samshin)
        xir priorbox op does not make a distinguish on layout, meaning, it always
        places the channels (value is 2) in the second dimension, so xnnc priorbox
        has to follow the constaints so that the shape inference can work correctly
        """
        # compute output shape
        """ Notice: do not remove
        if layout == Layout.NCHW:
            # batch_size: Since all images in a batch has same height and width, we only need to generate one set of priors which can be shared across all images.
            # channels: 2. First channel stores the mean of each prior coordinate. Second channel stores the variance of each prior coordinate.
            out_shape = [N, 2, W * H * num_priors * 4]
        else:
            out_shape = [N, W * H * num_priors * 4, 2]
        """
        out_shape = [N, 2, W * H * num_priors * 4]

        # compute outputs_tensor
        out_tensor = XTensor.zeros(
            out_shape,
            dtype=self.inputs_tensor[0].dtype,
            format=self.inputs_tensor[0].data_format,
        )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        """ Notice: do not remove
        if layout == Layout.NCHW:
            self.sequence_dims[self.layout]["out"] = [[0, 1, 2]]
        else:
            self.sequence_dims[self.layout]["out"] = [[0, 2, 1]]
        """
        self.sequence_dims[self.layout]["out"] = [[0, 1, 2]]


class XModelNodeStack(XModelNode):
    """
    XModelNode Stack Protocol

    Derived from:

    TensorFlow tf.stack
    Reference: https://www.tensorflow.org/api_docs/python/tf/stack
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "stack")
        # The axis to stack along
        self.__axis: int = 0

    @property
    def axis(self) -> int:
        return self.__axis

    @axis.setter
    def axis(self, axis) -> NoReturn:
        assert axis is not None, "'axis' should not be None."
        self.__axis = axis

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) > 1
        ), f"[ERROR] xnnc Stack requires more than one inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert all(
            [layout.name == x.data_format.name for x in self.inputs_tensor]
        ), f"[ERROR] xnnc {self.op_type} requires the layout of inputs tensor should be same as 'layout' argument. Op name: {self.op_name}."

        if self.layout != layout.name:
            # self.axis = curr_seq_dims_in.index(self.axis)
            raise NotImplementedError(
                f"[ERROR] Not support layout change in xnnc {self.op_type}."
            )

        curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
        assert all(
            [curr_seq_dims_in == seq for seq in self.sequence_dims[layout.name]["in"]]
        ), f"[ERROR] the dimension sequences of the inputs (xnnc Stack) must be same: {self.sequence_dims[self.layout]['in']}."

        # compute outputs_tensor
        out_tensor = XTensorCompute.stack(self.inputs_tensor, axis=self.axis)
        self.outputs_tensor = [out_tensor]

        # compute the dimension sequence of the output
        self.sequence_dims[layout.name]["out"] = [
            self.sequence_dims[layout.name]["in"][0]
        ]

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("XModelNodeStack")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeStack(node.name)

        xnode.axis = node.stack_param.axis

        if node.quantized:
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeMatMul(XModelNode):
    """
    XModelNode MatMul Protocol

    Derived from:

    TensorFlow tf.linalg.matmul
    https://www.tensorflow.org/api_docs/python/tf/linalg/matmul

    TensorFlow2 tf.keras.layers.Dense
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "matmul")
        # If True, "a" is transposed before multiplication.
        self.__transpose_a: bool = False
        # If true, "b" is transposed before multiplication.
        self.__transpose_b: bool = False
        # bias is optional field
        self.__bias: XTensor = None
        # quantization info for bias
        self.__qbias: Dict[str, Optional[int]] = {
            "bit_width": None,
            "quantize_pos": None,
            "signed": True,
            "round_mode": None,
        }

    @property
    def transpose_a(self) -> bool:
        return self.__transpose_a

    @transpose_a.setter
    def transpose_a(self, transpose_a):
        assert transpose_a is not None, "'transpose_a' should not be None."
        self.__transpose_a = transpose_a

    @property
    def transpose_b(self) -> bool:
        return self.__transpose_b

    @transpose_b.setter
    def transpose_b(self, transpose_b) -> NoReturn:
        assert transpose_b is not None, "'transpose_b' should not be None."
        self.__transpose_b = transpose_b

    @property
    def bias(self) -> XTensor:
        return self.__bias

    @bias.setter
    def bias(self, bias: XTensor):
        assert bias is not None, "'bias' should not be None."
        assert isinstance(bias, XTensor), "'bias' should be an XTensor object."
        self.__bias = bias

    @property
    def quant_bias(self) -> Dict[str, Optional[int]]:
        return self.__qbias

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 2
        ), f"[ERROR] xnnc {self.op_type} requires two inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert all(
            [layout.name == x.data_format.name for x in self.inputs_tensor]
        ), f"[ERROR] xnnc {self.op_type} requires the layout of inputs tensor should be same as 'layout' argument. Op name: {self.op_name}."

        if layout == Layout.NCHW:
            assert all([2 <= x.ndims <= 4 for x in self.inputs_tensor])
            matrix1 = self.inputs_tensor[0]
            if self.inputs_tensor[0].ndims == 4:
                matrix1 = matrix1.transpose((0, 2, 3, 1))
            matrix1.data_format = DataFormat.NHWC
            matrix2 = self.inputs_tensor[1]
            if self.inputs_tensor[1].ndims == 4:
                matrix2 = matrix2.transpose((0, 2, 3, 1))
            matrix2.data_format = DataFormat.NHWC

        else:
            matrix1, matrix2 = self.inputs_tensor

        # if self.layout != layout.name:
        #     # update transpose_a if necessary
        #     prev_seq_dims_a = self.sequence_dims[self.layout]["in"][0]
        #     curr_seq_dims_a = self.sequence_dims[layout.name]["in"][0]
        #     if prev_seq_dims_a != curr_seq_dims_a:
        #         # update the transpose_a property
        #         self.transpose_a = not self.transpose_a
        #     # update transpose_b if necessary
        #     prev_seq_dims_b = self.sequence_dims[self.layout]["in"][1]
        #     curr_seq_dims_b = self.sequence_dims[layout.name]["in"][1]
        #     if prev_seq_dims_b != curr_seq_dims_b:
        #         # update the transpose_b property
        #         self.transpose_b = not self.transpose_b

        assert (
            matrix1.ndims >= 2
        ), f"[ERROR] the rank of tensor (input1) must be no less than 2: actual:{matrix1.shape}."
        assert (
            matrix2.ndims >= 2
        ), f"[ERROR] the rank of tensor (input2) must be no less than 2: actual:{matrix2.shape}."

        if self.transpose_a and matrix1.ndims > 1:
            axes = list(range(matrix1.ndims))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            matrix1 = XTensorCompute.transpose(matrix1, axes=axes)
        if self.transpose_b and matrix2.ndims > 1:
            axes = list(range(matrix2.ndims))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            matrix2 = XTensorCompute.transpose(matrix2, axes=axes)

        assert (
            matrix1.shape[-1] == matrix2.shape[-2]
        ), f"[ERROR] The last dimention of 'matrx1' should be equal to the first dimension of 'matrix2' : matrix1: {matrix1.shape}, matrix2: {matrix2.shape}"

        # compute outputs_tensor
        out_tensor = XTensorCompute.matmul(matrix1, matrix2)
        if layout == Layout.NCHW and out_tensor.ndims == 4:
            out_tensor = out_tensor.transpose((0, 3, 1, 2))
            out_tensor.data_format = DataFormat.NCHW
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        if matrix1.ndims >= matrix2.ndims:
            curr_seq_dims_out = self.sequence_dims[self.layout]["in"][0]
        elif matrix1.ndims < matrix2.ndims:
            curr_seq_dims_out = self.sequence_dims[self.layout]["in"][1]
        self.sequence_dims[self.layout]["out"] = [curr_seq_dims_out]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeMatMul(node.name)

        # bias_term
        bias_term = node.matmul_param.bias_term

        # bias
        if bias_term:
            xnode.bias = XTensor.deserialize(node.matmul_param.bias)

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

            # quant_bias
            if bias_term:
                XModelNode.extract_quant_info(quant_config.quant_bias, xnode.quant_bias)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeIdentity(XModelNode):
    """
    XModelNode Identity Protocol

    Derived from:

    TensorFlow tf.identity
    https://www.tensorflow.org/api_docs/python/tf/identity
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "identity")
        # inputs_tensor
        self.__inputs_tensor: List[np.dtype] = []
        self.layout_type = LayoutType.INSENSITIVE

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc {self.op_type} requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        self.outputs_tensor = self.inputs_tensor

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeIdentity(node.name)

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeGSTiling(XModelNode):
    """
    XModelNode GSTiling Protocol

    Derived from: Xilinx
    https://github.com/Xilinx/Edge-AI-Platform-Tutorials/blob/master/docs/Caffe-Segmentation/Segment/caffe-master/src/caffe/layers/gs_tiling_layer.cpp
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "gstiling")
        self.__reverse: bool = False
        self.__stride: int = 1
        # inputs_tensor
        self.__inputs_tensor: List[np.dtype] = []
        self.layout_type = LayoutType.TOLERANT

    @property
    def reverse(self) -> bool:
        return self.__reverse

    @reverse.setter
    def reverse(self, flag: bool) -> NoReturn:
        assert flag is not None, "'flag' should not be None."
        self.__reverse = flag

    @property
    def stride(self) -> int:
        return self.__stride

    @stride.setter
    def stride(self, stride: int) -> NoReturn:
        assert stride is not None, "'stride' should not be None."
        self.__stride = stride

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc GSTiling requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].ndims == 4
        ), f"[ERROR] the rank of the input tensor must be 4: actual: {self.inputs_tensor[0].ndims}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        in_tensor = self.inputs_tensor[0]

        if layout == Layout.NCHW:
            N, ic, ih, iw = in_tensor.shape
        else:
            N, ih, iw, ic = in_tensor.shape

        # compute output shape
        if self.reverse:
            oc = int(ic / (self.stride ** 2))
            ow = iw * self.stride
            oh = ih * self.stride
        else:
            oc = ic * (self.stride ** 2)
            ow = iw / self.stride
            oh = ih / self.stride

        if layout == Layout.NCHW:
            out_shape = [N, oc, oh, ow]
        else:
            out_shape = [N, oh, ow, oc]

        self.outputs_tensor = [
            XTensor.zeros(
                out_shape, dtype=in_tensor.dtype, format=in_tensor.data_format
            )
        ]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeReorg(XModelNode):
    """
    XModelNode Reorg Protocol

    Derived from: YOLOv2

    Algorithm: https://www.mathworks.com/help/vision/ref/nnet.cnn.layer.yolov2reorglayer.html

    The reorganization layer improves the performance of the YOLO v2 object detection network
    by facilitating feature concatenation from different layers. It reorganizes the dimension
    of a lower layer feature map so that it can be concatenated with the higher layer feature map.

    Consider an input feature map of size [H W C], where:
    H is the height of the feature map.
    W is the width of the feature map.
    C is the number of channels.

    The reorganization layer chooses feature map values from locations based on the step sizes
    in stride and adds those feature values to the third dimension C. The size of the reorganized
    feature map from the reorganization layer is [floor(H/stride(1)) floor(W/stride(2)) C×stride(1)×stride(2)].
    For feature concatenation, the height and width of the reorganized feature map must match
    with the height and width of the higher layer feature map.
    """

    def __init__(self, op_name: str, strides: List[int]) -> NoReturn:
        assert op_name is not None, "The argument 'op_name' must be a string value."
        assert strides is not None, "'strides' should not be None."
        XModelNode.__init__(self, op_name, "reorg")
        # [stride_h, stride_w]
        self.__strides: List[int] = strides
        self.__reverse: bool = False
        self.layout_type = LayoutType.TOLERANT

    @property
    def stride(self) -> List[int]:
        return self.__strides

    @property
    def reverse(self) -> bool:
        return self.__reverse

    @reverse.setter
    def reverse(self, value: bool) -> NoReturn:
        self.__reverse = value

    def infer_shape(self, layout: Layout) -> NoReturn:
        """[N,C,H,W] to [N,C//stride//stride,H*stride,W*stride]"""
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc {self.op_type} requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        if layout == Layout.NCHW:
            N, C, H, W = self.inputs_tensor[0].shape
        else:
            N, H, W, C = self.inputs_tensor[0].shape

        stride_h, stride_w = self.stride
        OC = C * stride_h * stride_w
        OH = math.floor(H / stride_h)
        OW = math.floor(W / stride_w)

        if layout == Layout.NCHW:
            out_tensor = XTensor.zeros(
                [N, OC, OH, OW],
                dtype=self.inputs_tensor[0].dtype,
                format=self.inputs_tensor[0].data_format,
            )
        else:
            out_tensor = XTensor.zeros(
                [N, OH, OW, OC],
                dtype=self.inputs_tensor[0].dtype,
                format=self.inputs_tensor[0].data_format,
            )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeDeephiResize(XModelNode):
    """
    XModelNode DeephiResize Protocol

    Derived from: Xilinx
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "deephiresize")
        # [scale_h, scale_w]
        self.__scale: List[float] = [2.0, 2.0]
        # bilinear (default) or nearest
        self.__mode: str = "bilinear"
        self.layout_type = LayoutType.TOLERANT

    @property
    def scale(self) -> List[float]:
        return self.__scale

    @scale.setter
    def scale(self, value: List[float]) -> NoReturn:
        self.__scale = value

    @property
    def mode(self) -> str:
        return self.__mode

    @mode.setter
    def mode(self, mode: str) -> NoReturn:
        assert mode is not None, "'mode' should not be None."
        self.__mode = mode

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc {self.op_type} requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        in_tensor = self.inputs_tensor[0]

        # compute output shape
        if layout == Layout.NCHW:
            N, IC, IH, IW = in_tensor.shape
        else:
            N, IH, IW, IC = in_tensor.shape

        OC = IC
        OH = int(IH * self.scale[0])
        OW = int(IW * self.scale[1])
        if layout == Layout.NCHW:
            out_shape = [N, OC, OH, OW]
        else:
            out_shape = [N, OH, OW, OC]

        # compute outputs_tensor
        out_tensor = XTensor.zeros(
            out_shape, dtype=in_tensor.dtype, format=in_tensor.data_format
        )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeResize(XModelNode):
    """
    XModelNode XModelNodeResize Protocol

    Derived from:

    tensorflow::ops::ResizeNearestNeighbor
    https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/resize-nearest-neighbor

    tf.compat.v1.image.resize_nearest_neighbor
    https://www.tensorflow.org/api_docs/python/tf/compat/v1/image/resize_nearest_neighbor

    tensorflow::ops::ResizeBiliear
    https://www.tensorflow.org/versions/r1.15/api_docs/cc/class/tensorflow/ops/resize-bilinear?hl=en

    tf.compat.v1.image.resize_bilinear
    https://www.tensorflow.org/api_docs/python/tf/compat/v1/image/resize_bilinear
    """

    def __init__(self, op_name: str, mode: str):
        assert op_name is not None and isinstance(
            op_name, str
        ), "'op_name' should be a string value."
        assert (
            mode is not None
            and isinstance(mode, str)
            and mode.lower() in ["bilinear", "nearest"]
        ), f"'mode' should be 'bilinear' or 'nearest'."
        mode = mode.lower()
        XModelNode.__init__(self, op_name, "resize")
        # If true, the centers of the 4 corner pixels of the input and
        # output tensors are aligned, preserving the values at the corner
        # pixels. Defaults to false.
        self.__align_corners: bool = False
        self.__half_pixel_centers = False
        # bilinear or nearest
        self.__mode: str = mode
        self.layout_type = LayoutType.TOLERANT

    @property
    def mode(self) -> str:
        return self.__mode

    @property
    def align_corners(self) -> bool:
        return self.__align_corners

    @align_corners.setter
    def align_corners(self, flag: bool) -> NoReturn:
        assert flag is not None, "'flag' should not be None."
        assert isinstance(flag, bool), "'flag' should be a bool value."
        self.__align_corners = flag

    @property
    def half_pixel_centers(self) -> bool:
        return self.__half_pixel_centers

    @half_pixel_centers.setter
    def half_pixel_centers(self, half_pixel_centers: bool) -> NoReturn:
        assert half_pixel_centers is not None and isinstance(
            half_pixel_centers, bool
        ), "'half_pixel_centers' should be a bool value."
        self.__half_pixel_centers = half_pixel_centers

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 2
        ), f"[ERROR] xnnc Resize requires two inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        in_tensor, out_size_tensor = self.inputs_tensor

        # compute output shape
        if layout == Layout.NCHW.name:
            N, IC, _, _ = in_tensor.shape
        else:
            N, _, _, IC = in_tensor.shape
        OC = IC

        if out_size_tensor.ndims == 4:
            out_size = out_size_tensor.shape
            OH, OW = out_size[-2:] if layout == Layout.NCHW else out_size[1:3]
        else:
            OH, OW = self.inputs_tensor[1].tolist()

        if layout == Layout.NCHW:
            out_shape = [N, OC, OH, OW]
        else:
            out_shape = [N, OH, OW, OC]

        # compute outputs_tensor
        out_tensor = XTensor.zeros(out_shape, in_tensor.dtype, in_tensor.data_format)
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        self.sequence_dims[self.layout]["out"] = [
            self.sequence_dims[self.layout]["in"][0]
        ]

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("XModelNodeResize")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        mode = None
        if node.resize_param.mode == openir.ResizeParameter.Mode.NEAREST:
            mode = "nearest"
        elif node.resize_param.mode == openir.ResizeParameter.Mode.BILINEAR:
            mode = "bilinear"
        else:
            raise ValueError(
                f"[ERROR] Unsupported resize mode: {openir.ResizeParameter.Mode.Name(node.resize_param.mode)}."
            )
        assert mode is not None
        xnode = XModelNodeResize(node.name, mode)

        # align_corners
        xnode.align_corners = node.resize_param.align_corners
        # half_pixel_centers
        xnode.half_pixel_centers = node.resize_param.half_pixel_centers

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeFlatten(XModelNode):
    """
    XModelNode Flatten Protocol

    Derived from:

    PyTorch 1.3.1
    Reference: https://pytorch.org/docs/stable/torch.html?highlight=flatten#torch.flatten

    Caffe
    Reference: https://caffe.berkeleyvision.org/tutorial/layers/flatten.html

    TensorFlow tf.keras.layers.Flatten
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
    """

    def __init__(self, op_name: str) -> NoReturn:
        super(XModelNodeFlatten, self).__init__(op_name, "flatten")
        # the first dim to flatten
        self.__start_dim: int = 0
        # the last dim to flatten
        self.__end_dim: int = -1
        # inputs_tensor
        self.__inputs_tensor: List[np.dtype] = []
        self.layout_type = LayoutType.RECONSTRUCTED

    @property
    def start_dim(self) -> int:
        return self.__start_dim

    @start_dim.setter
    def start_dim(self, value: int) -> NoReturn:
        assert value is not None, "'value' should not be None."
        self.__start_dim = value

    @property
    def end_dim(self) -> int:
        return self.__end_dim

    @end_dim.setter
    def end_dim(self, value: int) -> NoReturn:
        assert value is not None, "'value' should not be None."
        self.__end_dim = value

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Flatten requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."
        assert (
            self.start_dim > 0
        ), f"[ERROR] The start dim of xnnc Flatten must be greater than 0: actual: {self.start_dim}"

        in_tensor = self.inputs_tensor[0]

        if self.host.origin == "pytorch":
            if self.layout == Layout.NCHW.name and layout == Layout.NHWC:
                if self.start_dim == 1:
                    self.start_dim = 3
                self.end_dim -= 1
                if self.start_dim > self.end_dim:
                    self.start_dim, self.end_dim = self.end_dim, self.start_dim
            elif self.layout == Layout.NHWC.name and layout == Layout.NCHW:
                if self.end_dim == 3:
                    self.end_dim = 1
                self.start_dim += 1
                if self.start_dim < self.end_dim:
                    self.start_dim, self.end_dim = self.end_dim, self.start_dim

        # compute output shape
        in_shape = in_tensor.shape
        rank = in_tensor.ndims
        if self.end_dim < 0:
            self.end_dim += rank

        if self.layout != layout:
            prev_seq = sorted(
                self.sequence_dims[self.layout]["in"][0][
                    self.start_dim : self.end_dim + 1
                ]
            )

            idx = [self.sequence_dims[layout.name]["in"][0].index(x) for x in prev_seq]
            self.start_dim = min(idx)
            self.end_dim = max(idx)

            curr_seq = sorted(
                self.sequence_dims[layout.name]["in"][0][
                    self.start_dim : self.end_dim + 1
                ]
            )
            assert prev_seq == curr_seq

            # update layout
            self.layout = layout.name

        out_shape = (
            in_shape[: self.start_dim]
            + [np.prod(in_shape[self.start_dim : self.end_dim + 1])]
            + in_shape[self.end_dim + 1 :]
        )
        out_tensor = XTensor.zeros(
            out_shape, dtype=in_tensor.dtype, format=in_tensor.data_format
        )
        self.outputs_tensor = [out_tensor]

        seq_in = self.sequence_dims[self.layout]["in"][0]
        curr_seq_dims_out = (
            seq_in[: self.start_dim]
            + [min(seq_in[self.start_dim : self.end_dim + 1])]
            + seq_in[self.end_dim + 1 :]
        )
        self.sequence_dims[self.layout]["out"] = [curr_seq_dims_out]

        # ! experimental
        if len(self.host.flatten_stack) > 0:
            self.host.flatten_stack.pop()
        self.host.flatten_stack.append(self)

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("avgpool")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeFlatten(node.name)

        # start_dim
        xnode.start_dim = node.flatten_param.start_dim
        # end_dim
        xnode.end_dim = node.flatten_param.end_dim

        # quantization
        quant_config = node.flatten_param.quant_config
        # quant_in
        XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
        # quant_out
        XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs and outputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeSize(XModelNode):
    """
    XModelNode Shape Protocol

    Derived from: PyTorch 1.3.1
    Reference: https://pytorch.org/docs/1.3.1/tensors.html?highlight=size#torch.Tensor.size
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNode.__init__(self, op_name, "size")
        # dims
        self.__dims: List[int] = None
        # output data type: "int32" or "int64"
        self.__out_type: str = "int32"

    @property
    def dims(self) -> List[int]:
        return self.__dims

    @dims.setter
    def dims(self, dims: List[int]) -> NoReturn:
        assert dims is not None, "'dims' should not be None."
        self.__dims = dims

    @property
    def out_type(self) -> str:
        return self.__out_type

    @out_type.setter
    def out_type(self, dtype: str) -> NoReturn:
        assert dtype is not None, "'dtype' should not be None."
        self.__out_type = dtype


class XModelNodeUpsample(XModelNode):
    """
    XModelNode Upsample Protocol

    Derived from:

    PyTorch 1.3.1
    https://pytorch.org/docs/1.3.1/nn.html?highlight=upsample#torch.nn.Upsample

    Tensorflow2 tf.keras.layers.UpSampling2D
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling2D
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNode.__init__(self, op_name, "upsample")
        # scale: multiplier for spatial size (N,C,H,W)
        self.__scale: List[float] = None
        # mode: 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'. Default: 'nearest'
        self.__mode: str = "nearest"
        # align_corners: if True, the corner pixels of the input and output tensors
        # are aligned, and thus preserving the values at those pixels. This only has
        # effect when mode is 'linear', 'bilinear', or 'trilinear'. Default: False
        self.__align_corners: bool = False

    @property
    def scale(self) -> List[float]:
        return self.__scale

    @scale.setter
    def scale(self, values: List[float]) -> NoReturn:
        assert values is not None, "'values' should not be None."
        self.__scale = values

    @property
    def mode(self) -> str:
        return self.__mode

    @mode.setter
    def mode(self, mode: str) -> NoReturn:
        assert mode is not None, "'mode' should not be None."
        assert mode in [
            "nearest",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
        ], "'mode' should be one of 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'."
        self.__mode = mode

    @property
    def align_corners(self) -> bool:
        return self.__align_corners

    @align_corners.setter
    def align_corners(self, flag: bool) -> NoReturn:
        assert flag is not None, "'flag' should not be None."
        self.__align_corners = flag

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Upsample requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            len(self.inputs_tensor[0].shape) == 4
        ), f"[ERROR] xnnc Upsample requires an 4-D input tensor: actual: {len(self.inputs_tensor[0].shape)}. Node name: {self.op_name}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        if layout == Layout.NHWC:
            N, H, W, C = self.inputs_tensor[0].shape
        else:
            N, C, H, W = self.inputs_tensor[0].shape

        # out_shape
        if self.host.origin == "tensorflow2":
            scale_h, scale_w = self.scale[-2:]
            out_shape = (
                (N, H * scale_h, W * scale_w, C)
                if layout == Layout.NHWC
                else (N, C, H * scale_h, W * scale_w)
            )
        else:
            raise ValueError(f"[ERROR] Unsupported model type: {self.host.origin}")

        # out_tensor
        out_tensor = XTensor.zeros(
            out_shape,
            dtype=self.inputs_tensor[0].dtype,
            format=self.inputs_tensor[0].data_format,
        )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        self.sequence_dims[self.layout]["out"] = [
            self.sequence_dims[self.layout]["in"][0]
        ]


class XModelNodeShape(XModelNode):
    """
    XModelNode Shape Protocol

    Derived from:

    TensorFlow tf.shape
    Reference: https://www.tensorflow.org/api_docs/python/tf/shape
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNode.__init__(self, op_name, "shape")
        self.layout_type = LayoutType.INSENSITIVE

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Flatten requires one input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the data format of input tensor should be same as the 'layout' argument: expected:{layout.name}, actual:{self.inputs_tensor[0].data_format.name}."

        # compute outputs_tensor
        shape = self.inputs_tensor[0].shape
        out_tensor = XTensor(
            np.array(shape, dtype=np.int32), format=self.inputs_tensor[0].data_format
        )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("squeeze")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeShape(node.name)

        if node.quantized:
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeReduceOp(XModelNode):
    """
    Base class of reduce ops
    """

    def __init__(self, op_name: str, op_type: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, op_type)
        # Axis or axes along which a reduce operation is performed.
        # If None (the default), reduces all dimensions.
        self.__axis: List[int] = None
        # If true, retain reduced dimensions with length 1.
        self.__keep_dims: bool = False
        # the value depends on keep_dims value
        # if keep_dims is True, it's LayoutType.DEPENDENT
        # otherwise, LayoutType.RECONSTRUCTED
        self.layout_type = LayoutType.RECONSTRUCTED

    @property
    def axis(self) -> List[int]:
        return self.__axis

    @axis.setter
    def axis(self, axis: List[int]) -> NoReturn:
        assert axis is not None, "'axis' should not be None."
        assert isinstance(axis, list), "'axis' should be a list of integers."
        self.__axis = axis

    @property
    def keep_dims(self) -> bool:
        return self.__keep_dims

    @keep_dims.setter
    def keep_dims(self, flag: bool) -> NoReturn:
        assert flag is not None, "'flag' should not be None."
        assert isinstance(flag, bool), "'flag' should be a bool value."
        self.__keep_dims = flag
        # set layout type
        self.layout_type = (
            LayoutType.DEPENDENT if self.__keep_dims else LayoutType.RECONSTRUCTED
        )


class XModelNodeReduceProd(XModelNodeReduceOp):
    """
    XModelNode Prod Protocol

    The product of elements across dimensions of a tensor

    Derived from:

    TensorFlow tf.math.reduce_prod
    https://www.tensorflow.org/api_docs/python/tf/math/reduce_prod
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeReduceOp.__init__(self, op_name, "reduceprod")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert self.inputs_tensor is not None
        assert (
            len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc ElemReduceProd requires one input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        # preprocess negative axis
        for i, x in enumerate(self.axis):
            if x < 0:
                self.axis[i] += self.inputs_tensor[0].ndims

        # update the axis property if layout changes
        if self.layout != layout.name:
            prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
            curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
            new_axis = []
            for i in self.axis:
                x = prev_seq_dims_in[i]
                new_axis.append(curr_seq_dims_in.index(x))
            self.axis = new_axis
        else:
            prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
            for i, dim in enumerate(self.axis):
                if dim == -1:
                    self.axis[i] = prev_seq_dims_in[dim]

        # compute outputs_tensor
        out_tensor = XTensorCompute.reduce_prod(
            self.inputs_tensor[0], self.axis, self.keep_dims
        )
        self.outputs_tensor = [out_tensor]

        # compute the dimension sequence of the output
        if self.keep_dims:
            self.sequence_dims[layout.name]["out"] = self.sequence_dims[layout.name][
                "in"
            ]
        else:
            curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
            curr_seq_dims_out = []
            for i, v in enumerate(curr_seq_dims_in):
                if i not in self.axis:
                    curr_seq_dims_out.append(v)
            if len(curr_seq_dims_out) == 0:
                curr_seq_dims_out = [x for x in range(self.outputs_tensor[0].ndims)]
            self.sequence_dims[layout.name]["out"] = [curr_seq_dims_out]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name


class XModelNodeReduceSum(XModelNodeReduceOp):
    """
    XModelNode ReduceSum Protocol

    The sum of elements across dimensions of a tensor.

    Derived from:

    TensorFlow tf.math.reduce_sum
    https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeReduceOp.__init__(self, op_name, "reducesum")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc ElemReduceSum requires one input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        # preprocess negative axis
        for i, x in enumerate(self.axis):
            if x < 0:
                self.axis[i] += self.inputs_tensor[0].ndims

        # update the axis property if layout changes
        if self.layout != layout.name:
            prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
            curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
            new_axis = []
            for i in self.axis:
                x = prev_seq_dims_in[i]
                new_axis.append(curr_seq_dims_in.index(x))
            self.axis = new_axis
        else:
            prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
            for i, dim in enumerate(self.axis):
                if dim == -1:
                    self.axis[i] = prev_seq_dims_in[dim]

        # compute outputs_tensor
        out_tensor = XTensorCompute.reduce_sum(
            self.inputs_tensor[0], self.axis, self.keep_dims
        )
        self.outputs_tensor = [out_tensor]

        # compute the dimension sequence of the output
        if self.keep_dims:
            self.sequence_dims[layout.name]["out"] = self.sequence_dims[layout.name][
                "in"
            ]
        else:
            curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
            curr_seq_dims_out = []
            for i, v in enumerate(curr_seq_dims_in):
                if i not in self.axis:
                    curr_seq_dims_out.append(v)
            self.sequence_dims[layout.name]["out"] = [curr_seq_dims_out]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name


class XModelNodeReduceMax(XModelNodeReduceOp):
    """
    XModelNode ReduceMax Protocol

    The maximum of elements across dimensions of a tensor.

    Derived from:

    TensorFlow tf.math.reduce_max
    https://www.tensorflow.org/api_docs/python/tf/math/reduce_max
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeReduceOp.__init__(self, op_name, "reducemax")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc ElemReduceMax requires one input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        # preprocess negative axis
        for i, x in enumerate(self.axis):
            if x < 0:
                self.axis[i] += self.inputs_tensor[0].ndims

        # update the axis property if layout changes
        if self.layout != layout.name:
            prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
            curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
            new_axis = []
            for i in self.axis:
                x = prev_seq_dims_in[i]
                new_axis.append(curr_seq_dims_in.index(x))
            self.axis = new_axis
        else:
            prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
            for i, dim in enumerate(self.axis):
                if dim == -1:
                    self.axis[i] = prev_seq_dims_in[dim]

        # compute outputs_tensor
        out_tensor = XTensorCompute.reduce_max(
            self.inputs_tensor[0], self.axis, self.keep_dims
        )
        self.outputs_tensor = [out_tensor]

        # compute the dimension sequence of the output
        if self.keep_dims:
            self.sequence_dims[layout.name]["out"] = self.sequence_dims[layout.name][
                "in"
            ]
        else:
            curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
            curr_seq_dims_out = []
            for i, v in enumerate(curr_seq_dims_in):
                if i not in self.axis:
                    curr_seq_dims_out.append(v)
            self.sequence_dims[layout.name]["out"] = [curr_seq_dims_out]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name


class XModelNodeScale(XModelNode):
    """
    XModelNode Scale Protocol

    Derived from:

    Caffe Scale Layer
    https://caffe.berkeleyvision.org/tutorial/layers/scale.html
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNode.__init__(self, op_name, "scale")
        # axis
        self.__axis: int = 1
        # True, bias is present; otherwise, False.
        self.__bias_term: bool = False
        # bias
        self.__bias: XTensor = None
        # quantization info for bias
        self.__qbias: Dict[str, Optional[int]] = {
            "bit_width": None,
            "quantize_pos": None,
            "signed": True,
            "round_mode": None,
        }

    @property
    def axis(self) -> int:
        return self.__axis

    @axis.setter
    def axis(self, axis: int) -> NoReturn:
        assert axis is not None, "'axis' should not be None."
        assert isinstance(axis, int), "'axis' should be an integer."
        self.__axis = axis

    @property
    def bias_term(self) -> bool:
        return self.__bias_term

    @bias_term.setter
    def bias_term(self, bias_term: bool):
        assert (
            type(bias_term) == bool
        ), "The argument 'bias_term' only accepts True or False."
        self.__bias_term = bias_term

    @property
    def bias(self) -> XTensor:
        return self.__bias

    @bias.setter
    def bias(self, bias: XTensor):
        assert bias is not None, "'bias' should not be None."
        assert isinstance(bias, XTensor), "'bias' should be an XTensor object."
        self.__bias = bias

    @property
    def quant_bias(self) -> Dict[str, Optional[int]]:
        return self.__qbias

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 2
        ), f"[ERROR] xnnc Scale requires two inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        in_tensor = self.inputs_tensor[0]

        if self.axis < 0:
            self.axis += in_tensor.ndims

        # update axis if layout changes
        if self.layout != layout.name:
            prev_seq_dims_in = self.sequence_dims[self.layout]["in"][0]
            curr_seq_dims_in = self.sequence_dims[layout.name]["in"][0]
            x = prev_seq_dims_in[self.axis]
            self.axis = curr_seq_dims_in.index(x)

        # compute outputs_tensor
        self.outputs_tensor = [XTensor.zero_like(in_tensor)]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        self.sequence_dims[self.layout]["out"] = [
            self.sequence_dims[self.layout]["in"][0]
        ]


class XModelNodeSigmoid(XModelNode):
    """
    XModelNode Sigmoid Protocol

    Derived from:

    TensorFlow tf.math.sigmoid
    https://www.tensorflow.org/api_docs/python/tf/math/sigmoid

    TensorFlow2 Keras tf.keras.activations.sigmoid
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNode.__init__(self, op_name, "sigmoid")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc Scale requires one input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        # compute outputs_tensor
        self.outputs_tensor = self.inputs_tensor

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        xnode = XModelNodeSigmoid(node.name)

        if node.quantized:
            xnode.is_quantized = True
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeL2Normalize(XModelNode):
    """
    XModelNode L2Normalize Protocol

    Derived from:

    TensorFlow tf.math.l2_normalize
    https://www.tensorflow.org/api_docs/python/tf/math/l2_normalize
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "l2_normalize")
        # Axis or axes along which a reduce operation is performed.
        # If None (the default), reduces all dimensions.
        self.__axis: List[int] = None
        # If true, retain reduced dimensions with length 1.
        self.__keep_dims: bool = False
        # A lower bound value for the norm.
        self.__epsilon: float = 1e-12

    @property
    def axis(self) -> List[int]:
        return self.__axis

    @axis.setter
    def axis(self, axis: List[int]):
        self.__axis = axis

    @property
    def keep_dims(self) -> bool:
        return self.__keep_dims

    @keep_dims.setter
    def keep_dims(self, keepdims: bool):
        self.__keep_dims = keepdims
        # set layout type
        self.layout_type = (
            LayoutType.DEPENDENT if self.__keep_dims else LayoutType.RECONSTRUCTED
        )

    @property
    def epsilon(self) -> float:
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> NoReturn:
        assert value is not None, "'value' should not be None."
        assert isinstance(value, float), "'value' should be a float value."
        self.__epsilon = value

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc ElemMax requires a single input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        self.outputs_tensor = self.inputs_tensor

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeSlice(XModelNode):
    """
    XModelNode Slice Protocol

    Derived from:

    Caffe
    https://caffe.berkeleyvision.org/tutorial/layers/slice.html
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        super().__init__(op_name, "slice")
        # target axis along which to slice, default N dimension
        self.__axis: int = 0
        # indexes in the selected dimension, default None. None indicates slice evenly.
        self.__slice_points: Optional[List[int]] = None

    @property
    def axis(self) -> int:
        return self.__axis

    @axis.setter
    def axis(self, axis: int) -> NoReturn:
        assert axis is not None, "'axis' should not be None."
        assert isinstance(axis, int), "'axis' should be an integer."
        self.__axis = axis

    @property
    def slice_points(self) -> Optional[List[int]]:
        return self.__slice_points

    @slice_points.setter
    def slice_points(self, points: Optional[List[int]]) -> NoReturn:
        assert points is None or isinstance(
            points, list
        ), "'points' should be either a list of positive integer, or None."
        assert (
            len(points) == len(self.top) - 1
        ), "'points' should have a size only one smaller than the size of top."
        self.__slice_points = points

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc XModelNodeSlice requires a single input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        # if axis is negative, convert it to be positive
        if self.axis < 0:
            self.axis += self.inputs_tensor[0].ndims

        if self.layout != layout.name:
            # update axis
            seq = self.sequence_dims[layout.name]["in"][0]
            self.axis = seq.index(self.axis)

        in_shape = self.inputs_tensor[0].shape
        maxsize = in_shape[self.axis]
        dims = []
        start = 0
        for stop in self.slice_points:
            if start >= stop:
                raise ValueError(
                    f"[ERROR] start index should less than stop index: start:{start}, stop:{stop}."
                )
            dims.append(stop - start)
            start = stop
        dims.append(maxsize - start)

        # compute outputs_tensor
        self.outputs_tensor = []
        for dim in dims:
            in_shape[self.axis] = dim
            out_tensor = XTensor.zeros(
                in_shape,
                dtype=self.inputs_tensor[0].dtype,
                format=self.inputs_tensor[0].data_format,
            )
            self.outputs_tensor.append(out_tensor)

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        self.sequence_dims[self.layout]["out"] = [
            self.sequence_dims[self.layout]["in"][0]
        ] * len(self.outputs_tensor)


class XModelNodeDepthToSpace(XModelNode):
    """
    XModelNode DepthToSpace Protocol

    Derived from:

    tf.nn.depth_to_space
    https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        super().__init__(op_name, "depth_to_space")

        # block_size
        self.__block_size: int = 1

    @property
    def block_size(self) -> int:
        return self.__block_size

    @block_size.setter
    def block_size(self, size: int) -> NoReturn:
        assert size is not None, "'size' should not be None."
        assert (
            isinstance(size, int) and size >= 1
        ), "'size' should be a positive integer."

        self.__block_size = size

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc XModelNodeDepthToSpace requires a single input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        in_shape = self.inputs_tensor[0].shape
        assert (
            len(in_shape) == 4
        ), "[ERROR] The dimension of input of XModelNodeDepthToSpace must be 4."

        if layout == Layout.NHWC:
            N, H, W, C = in_shape
        else:
            N, C, H, W = in_shape

        assert (
            C % (self.block_size ** 2) == 0
        ), "[ERROR] XModelNodeDepthToSpace requires that the input channel must be divided by the square of block_size."

        OC = C // (self.block_size ** 2)
        OH = H * self.block_size
        OW = W * self.block_size

        out_shape = (N, OH, OW, OC) if layout == Layout.NHWC else (N, OC, OH, OW)

        out_tensor = XTensor.zeros(
            out_shape,
            dtype=self.inputs_tensor[0].dtype,
            format=self.inputs_tensor[0].data_format,
        )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name
        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeTypeCast(XModelNodeElemOp):
    """
    XModelNode TypeCast Protocol

    Derived from:

    TensorFlow tf.cast
    https://www.tensorflow.org/api_docs/python/tf/cast
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "type_cast")

        # source dtype
        self.__src_dtype: np.dtype = np.float32

        # destination dtype
        self.__dst_dtype: np.dtype = np.float32

    @property
    def src_dtype(self) -> str:
        return self.__src_dtype

    @src_dtype.setter
    def src_dtype(self, dtype: np.dtype) -> NoReturn:
        assert dtype is not None, "'dtype' should not be None."
        self.__src_dtype = dtype

    @property
    def dst_dtype(self) -> str:
        return self.__dst_dtype

    @dst_dtype.setter
    def dst_dtype(self, dtype: np.dtype) -> NoReturn:
        assert dtype is not None, "'dtype' should not be None."
        self.__dst_dtype = dtype

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc TypeCast requires a single input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        self.outputs_tensor = [self.inputs_tensor[0].copy().astype(self.dst_dtype)]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeRandomStandardNormal(XModelNode):
    """
    XModelNode RandomStandardNormal Protocol

    Derived from:

    TensorFlow tf.raw_ops.RandomStandardNormal
    https://www.tensorflow.org/api_docs/python/tf/raw_ops/RandomStandardNormal?hl=id
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNode.__init__(self, op_name, "random_standard_normal")

        # shape
        self.__shape: List[int] = None

        # dtype
        self.__dtype: np.dtype = np.float32

        # seed
        self.__seed: int = 0

        # seed2
        self.__seed2: int = 0

    @property
    def shape(self) -> List[int]:
        return self.__shape

    @shape.setter
    def shape(self, out_shape: List[int]) -> NoReturn:
        self.__shape = out_shape

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    @dtype.setter
    def dtype(self, dtype: np.dtype) -> NoReturn:
        assert dtype is not None, "'dtype' should not be None."
        self.__dtype = dtype

    @property
    def seed(self) -> int:
        return self.__seed

    @seed.setter
    def seed(self, seed: int) -> NoReturn:
        assert seed is not None and isinstance(
            seed, int
        ), f"'seed' is a positive integer or 0."
        self.__seed = seed

    @property
    def seed2(self) -> int:
        return self.__seed2

    @seed2.setter
    def seed2(self, seed: int) -> NoReturn:
        assert seed is not None and isinstance(
            seed, int
        ), f"'seed' is a positive integer or 0."
        self.__seed2 = seed

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc RandomStandardNormal requires a single input: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        if self.shape:
            out_shape = self.shape
        else:
            out_shape = self.inputs_tensor[0].tolist()
        out_tensor = XTensor.zeros(
            tuple(out_shape), dtype=self.dtype, format=self.inputs_tensor[0].data_format
        )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        assert out_tensor.ndims in [2, 4]
        if out_tensor.ndims == 4:
            curr_seq_dict = self.sequence_dims.get(layout.name)
            if layout == Layout.NHWC:
                curr_seq_dict["out"] = [[0, 2, 3, 1]]
            else:
                curr_seq_dict["out"] = [[0, 1, 2, 3]]

        else:
            self.sequence_dims[layout.name]["out"] = [
                [x for x in range(out_tensor.ndims)]
            ]


class XModelNodeSpaceToBatchND(XModelNode):
    """
    XModelNode SpaceToBatchND Protocol

    Derived from:

    TensorFlow tf.space_to_batch_nd
    https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd
    """

    def __init__(self, op_name: str, block_shape: List[int], paddings: List[int]):
        assert op_name is not None, "'op_name' should not be None."
        assert block_shape is not None, "'block_shape' should not be None."
        assert paddings is not None, "'paddings' should not be None."
        assert (
            all([x >= 1 for x in block_shape]) == True
        ), "all values specified by 'block_shape' must be >= 1."
        assert (
            all([x >= 0 for x in paddings]) == True
        ), "all values specified by 'paddings' must be >= 0."

        XModelNode.__init__(self, op_name, "spacetobatchnd")
        self.__block_shape: List[int] = block_shape
        # the padding of the input with zeros across the spatial dimensions
        # in order: [top, bottom, left, right]
        self.__paddings: List[int] = paddings
        # inputs_tensor
        self.__inputs_tensor: List[np.dtype] = []

    @property
    def block_shape(self) -> List[int]:
        return self.__block_shape

    @property
    def paddings(self) -> List[int]:
        return self.__paddings

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc {self.op_type} requires one input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        input_tensor = self.inputs_tensor[0]
        assert input_tensor.ndims == 4

        if layout == Layout.NHWC:
            N, H, W, C = input_tensor.shape
        else:
            N, C, H, W = input_tensor.shape

        # padded
        padded_h, padded_w = (
            self.paddings[0] + H + self.paddings[1],
            self.paddings[2] + W + self.paddings[3],
        )

        # output_h, output_w
        output_h, output_w = (
            padded_h // self.block_shape[0],
            padded_w // self.block_shape[1],
        )

        # output_n
        output_n = N * self.block_shape[0] * self.block_shape[1]

        if layout == Layout.NHWC:
            out_tensor = XTensor.zeros(
                (output_n, output_h, output_w, C),
                dtype=input_tensor.dtype,
                format=input_tensor.data_format,
            )
        else:
            out_tensor = XTensor.zeros(
                (output_n, C, output_h, output_w),
                dtype=input_tensor.dtype,
                format=input_tensor.data_format,
            )
        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("XModelNodeSpaceToBatchND")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        # block_shape
        block_shape = list(node.spacetobatch_param.block_shape)

        # paddings
        paddings = list(node.spacetobatch_param.paddings)

        xnode = XModelNodeSpaceToBatchND(node.name, block_shape, paddings)

        if node.quantized:
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeBatchToSpaceND(XModelNode):
    """
    XModelNode BatchToSpaceND Protocol

    Derived from:

    TensorFlow tf.batch_to_space
    https://www.tensorflow.org/api_docs/python/tf/batch_to_space
    """

    def __init__(self, op_name: str, block_shape: List[int], crops: List[int]):
        assert op_name is not None, "'op_name' should not be None."
        assert block_shape is not None, "'block_shape' should not be None."
        assert crops is not None, "'crops' should not be None."
        assert (
            all([x >= 1 for x in block_shape]) == True
        ), "all values specified by 'block_shape' must be >= 1."
        assert (
            all([x >= 0 for x in crops]) == True
        ), "all values specified by 'crops' must be >= 0."
        XModelNode.__init__(self, op_name, "batchtospacend")
        self.__block_shape: List[int] = block_shape
        self.__crops: List[int] = crops

    @property
    def block_shape(self) -> List[int]:
        return self.__block_shape

    @property
    def crops(self) -> List[int]:
        return self.__crops

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc {self.op_type} requires one input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        input_tensor = self.inputs_tensor[0]

        if layout == Layout.NHWC:
            N, H, W, C = input_tensor.shape
        else:
            N, C, H, W = input_tensor.shape

        # output_n
        output_n = N // (self.block_shape[0] * self.block_shape[1])

        # output_h, output_w
        output_h, output_w = (
            H * self.block_shape[0],
            W * self.block_shape[1],
        )

        # crops
        output_h, output_w = (
            output_h - self.crops[0] - self.crops[1],
            output_w - self.crops[2] - self.crops[3],
        )

        if layout == Layout.NHWC:
            out_tensor = XTensor.zeros(
                (output_n, output_h, output_w, C),
                dtype=input_tensor.dtype,
                format=input_tensor.data_format,
            )
        else:
            out_tensor = XTensor.zeros(
                (output_n, C, output_h, output_w),
                dtype=input_tensor.dtype,
                format=input_tensor.data_format,
            )

        self.outputs_tensor = [out_tensor]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]

    def serialize(self) -> openir.NodeProto:
        raise NotImplementedError("XModelNodeBatchToSpaceND")

    @staticmethod
    def deserialize(node: openir.NodeProto) -> "XModelNode":
        assert node is not None, "'node' should be not None."
        assert isinstance(
            node, openir.NodeProto
        ), "'node' should be of openir.NodeProto type."

        # block_shape
        block_shape = list(node.batchtospace_param.block_shape)

        # crops
        crops = list(node.batchtospace_param.crops)

        xnode = XModelNodeBatchToSpaceND(node.name, block_shape, crops)

        if node.quantized:
            # quantization
            quant_config = node.quant_config
            if node.quant_config.HasField("quant_in"):
                # quant_in
                XModelNode.extract_quant_info(quant_config.quant_in, xnode.quant_in)
            if node.quant_config.HasField("quant_out"):
                # quant_out
                XModelNode.extract_quant_info(quant_config.quant_out, xnode.quant_out)

        # inputs
        if len(node.inputs) > 0:
            xnode.bottom = [x for x in node.inputs]

        # outputs
        if len(node.outputs) > 0:
            xnode.top = [x for x in node.outputs]

        return xnode


class XModelNodeArgmax(XModelNode):
    """
    XModelNode Argmax Protocol

    Derived from:

    TensorFlow tf.math.argmax
    https://www.tensorflow.org/api_docs/python/tf/math/argmax
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "argmax")
        # the axis to reduce across
        self.__axis = 0
        # output_type
        self.__output_type = np.int64

    def axis(self) -> int:
        return self.__axis

    def axis(self, axis: int) -> NoReturn:
        assert axis is not None and isinstance(
            axis, int
        ), "'axis' should be an integer."
        self.__axis = axis

    def output_type(self) -> Union[np.int32, np.int64]:
        return self.__output_type

    def output_type(self, dtype: Union[np.int32, np.int64]) -> NoReturn:
        assert dtype is not None and dtype in [
            np.int32,
            np.int64,
        ], "'dtype' should be one of numpy.int32 and numpy.int64."
        self.__output_type = dtype

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 1
        ), f"[ERROR] xnnc {self.op_type} requires one input: actual: {len(self.inputs_tensor)}. Op name: {self.op_name}."
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        in_tensor = self.inputs_tensor[0]

        if self.axis < 0:
            self.axis = in_tensor.ndims + self.axis

        if self.layout != layout.name:
            if not self.sequence_dims[layout.name]["out"]:
                # update axis
                seq = self.sequence_dims[layout.name]["in"][0]
                self.axis = seq.index(self.axis)
            else:
                # update axis
                out_seq_before_change = self.sequence_dims[layout.name]["out"][0]
                in_seq = self.sequence_dims[self.layout]["in"][0]
                self.axis = out_seq_before_change.index(in_seq[self.axis])
            # update layout
            self.layout = layout.name

        out_tensor = XTensorCompute.argmax(in_tensor, self.axis)
        if out_tensor.dtype != self.output_type:
            out_tensor.asdtype(self.output_type)
        self.outputs_tensor = [out_tensor]

        self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeLinear(XModelNode):
    """
    XModelNode Linear Protocol

    Derived from: PyTorch 1.3.1
    Reference: https://pytorch.org/docs/stable/nn.html?highlight=linear#torch.nn.Linear
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNode.__init__(self, op_name, "linear")
        # (out_features,in_features)
        self.__weights: XTensor = None
        # (out_features)
        self.__bias: XTensor = None
        # quantization info for weights
        self.__qweights: Dict[str, Optional[int]] = {
            "bit_width": None,
            "quantize_pos": None,
            "signed": True,
            "round_mode": None,
        }
        # quantization info for bias
        self.__qbias: Dict[str, Optional[int]] = {
            "bit_width": None,
            "quantize_pos": None,
            "signed": True,
            "round_mode": None,
        }

    @property
    def weights(self) -> XTensor:
        return self.__weights

    @weights.setter
    def weights(self, weights: XTensor) -> NoReturn:
        assert weights is not None, "'weights' should not be None."
        assert isinstance(weights, XTensor), "'weights' should be an XTensor object."
        self.__weights = weights

    @property
    def bias(self) -> XTensor:
        return self.__bias

    @bias.setter
    def bias(self, bias: XTensor) -> NoReturn:
        assert bias is not None, "'bias' should not be None."
        assert isinstance(bias, XTensor), "'bias' should be an XTensor object."
        self.__bias = bias

    @property
    def quant_weights(self) -> Dict[str, Optional[int]]:
        return self.__qweights

    @property
    def quant_bias(self) -> Dict[str, Optional[int]]:
        return self.__qbias


class XModelNodeBatchNorm2d(XModelNode):
    """
    XModelNode BatchNorm2d Protocol

    Derived from: PyTorch 1.3.1
    Reference: https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d
    """

    def __init__(self, op_name: str, num_features: int) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        assert num_features is not None, "'num_features' should not be None."
        XModelNode.__init__(self, op_name, "batchnorm2d")
        # num_features
        self.__num_features: int = num_features
        # eps: a value added to the denominator for numerical stability.
        self.__eps: float = 1e-05
        # momentum: the value used for the running_mean and running_var
        self.__momentum: float = 0.1
        # affine: a boolean value that when set to ``True``, this module has learnable affine parameters.
        self.__affine: bool = True
        # track_running_stats
        self.__track_running_stats: bool = True
        # weights
        self.__weights: XTensor = None
        # bias
        self.__bias: XTensor = None
        # running_mean
        self.__running_mean: XTensor = None
        # running_var
        self.__running_var: XTensor = None

    @property
    def num_features(self) -> int:
        return self.__num_features

    @property
    def eps(self) -> float:
        return self.__eps

    @eps.setter
    def eps(self, eps: float) -> NoReturn:
        assert eps is not None, "'eps' should not be None."
        self.__eps = eps

    @property
    def momentum(self) -> float:
        return self.__momentum

    @momentum.setter
    def momentum(self, momentum: float) -> NoReturn:
        assert momentum is not None, "'momentum' should not be None."
        self.__momentum = momentum

    @property
    def affine(self) -> bool:
        return self.__affine

    @affine.setter
    def affine(self, affine) -> NoReturn:
        assert affine is not None, "'affine' should not be None."
        self.__affine = affine

    @property
    def track_running_stats(self) -> bool:
        return self.__track_running_stats

    @track_running_stats.setter
    def track_running_stats(self, track_running_stats: bool) -> NoReturn:
        assert (
            track_running_stats is not None
        ), "'track_running_stats' should not be None."
        self.__track_running_stats = track_running_stats

    @property
    def weights(self) -> XTensor:
        return self.__weights

    @weights.setter
    def weights(self, weights: XTensor) -> NoReturn:
        assert weights is not None, "'weights' should not be None."
        assert isinstance(weights, XTensor), "'weights' should be an XTensor object."
        self.__weights = weights

    @property
    def bias(self) -> XTensor:
        return self.__bias

    @bias.setter
    def bias(self, bias: XTensor) -> NoReturn:
        assert bias is not None, "'bias' should not be None."
        assert isinstance(bias, XTensor), "'bias' should be an XTensor object."
        self.__bias = bias

    @property
    def running_mean(self) -> XTensor:
        return self.__running_mean

    @running_mean.setter
    def running_mean(self, mean: XTensor) -> NoReturn:
        assert mean is not None, "'mean' should not be None."
        assert isinstance(mean, XTensor), "'mean' should be an XTensor object."
        self.__running_mean = mean

    @property
    def running_var(self) -> XTensor:
        return self.__running_var

    @running_var.setter
    def running_var(self, variance: XTensor) -> NoReturn:
        assert variance is not None, "'variance should not be None."
        assert isinstance(variance, XTensor), "'variance' should be an XTensor object."
        self.__running_var = variance


class XModelNodeUnknown(XModelNode):
    def __init__(self, op_name: str, op_type: str = "unknown"):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, op_type)

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert self.host.origin in ["tensorflow", "tensorflow2",], "[ERROR] XModelNodeUnknown only supports for tensorflow and tensorflow2 models."
        assert 'shape' in self.tmp_params, f"'{self.op_name}' is an unknown op that requires a specific shape. Please provide shape information."

        if self.tmp_params['shape'][0] > 1:
            raise ValueError(f"'{self.op_name}' is an unknown operation, the shape is {self.tmp_params['shape']}. However, the DPU only supports shapes with a batchsize of 1.")
        elif self.tmp_params['shape'][0] == -1: # batchsize is -1
            for dim in self.tmp_params['shape'][1:]:
                if dim < 1:
                    raise ValueError(f"'{self.op_name}' is an unknown operation, the shape is {self.tmp_params['shape']}.")
            self.tmp_params['shape'][0] = 1

        shape = self.tmp_params['shape']
        dtype = self.tmp_params['data_type']

        self.outputs_tensor = [
            XTensor(np.zeros(shape,dtype=dtype), format=DataFormat[self.layout])
        ]

        if len(shape) == 2:
            self.sequence_dims[self.layout]["out"] = [[0, 1]]
        else:
            self.sequence_dims[self.layout]["out"] = self.sequence_dims[self.layout]["in"]


class XModelNodeExpandDims(XModelNode):
    """
    XModelNode ExpandDims Protocol

    Derived from:

    TensorFlow tf.expand_dims
    https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
    """

    def __init__(self, op_name: str):
        assert op_name is not None, "The argument 'op_name' must be a string value."
        XModelNode.__init__(self, op_name, "expand_dims")
        self.__shape: List[int] = None
        self.layout_type = LayoutType.RECONSTRUCTED

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and 2 == len(self.inputs_tensor)
        ), f"[ERROR] xnnc Reshape requires one or two inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument."

        if 'tmp_axis' not in self.tmp_params:
            axis=self.host.get_xnode_by_name(self.bottom[1]).tensor.tolist()

            for k, v in enumerate(axis):
                if v < 0:
                    nums_len = len(self.inputs_tensor[0].to_numpy().shape)+len(axis)
                    v+=nums_len
                    axis[k]=v

            assert len(axis) == len(set(axis))
            self.tmp_params['tmp_axis'] = axis

        else:
            axis = self.tmp_params['tmp_axis']

        if '_output_shapes' in self.tmp_params:
            shape = self.tmp_params['_output_shapes']
            for i in shape:
                if i < 0:
                    tmp_out = np.expand_dims(self.inputs_tensor[0].to_numpy(), axis=axis)
                    shape = tmp_out.shape
        else:
            if type(axis)==list and len(axis)==1:
                axis=axis[0]
            shape = np.expand_dims(self.inputs_tensor[0].to_numpy(), axis=axis).shape

        dtype = self.tmp_params['T']
        out_tensor = XTensor(np.zeros(shape,dtype=dtype), format=DataFormat[self.layout])
        self.host.get_xnode_by_name(self.bottom[1]).tensor=XTensor(np.array(out_tensor.shape))

        self.outputs_tensor = [out_tensor]

        if out_tensor.ndims == 4:
            if layout == Layout.NHWC:
                self.sequence_dims[layout.name]["out"] = [[0, 2, 3, 1]]
            else:
                self.sequence_dims[layout.name]["out"] = [[0, 1, 2, 3]]
        else:
            self.sequence_dims[layout.name]["out"] = [
                [x for x in range(out_tensor.ndims)]
            ]


        # update layout
        if self.layout != layout.name:
            self.layout = layout.name


class XModelNodeElemRealPow(XModelNodeElemOp):
    """
    XModelNode ElemRealPow Protocol

    Computes the power of one value to another.

    Derived from:

    TensorFlow tf.pow
    https://www.tensorflow.org/api_docs/python/tf/math/pow
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNodeElemOp.__init__(self, op_name, "elemrealpow")

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 2
        ), f"[ERROR] xnnc ElemRealPow requires two inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert all(
            [
                layout.name == in_tensor.data_format.name
                for in_tensor in self.inputs_tensor
            ]
        ), f"[ERROR] xnnc {self.op_type} requires that the layout of all inputs must be same as 'layout' argument: op name: {self.op_name}."

        # check dimenstion sequences of two inputs
        curr_seq_dims_in_a, curr_seq_dims_in_b = self.sequence_dims[layout.name]["in"]
        if len(curr_seq_dims_in_a) > 1 and len(curr_seq_dims_in_b) > 1:
            assert (
                curr_seq_dims_in_a == curr_seq_dims_in_b
            ), f"[ERROR] Mis-matched dimension sequences of inputs (xnnc ElemRealPow): {curr_seq_dims_in_a}, {curr_seq_dims_in_b}"

        a, b = self.inputs_tensor
        c = XTensorCompute.elem_true_power(a, b)
        self.outputs_tensor = [c]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        if curr_seq_dims_in_a == curr_seq_dims_in_b:
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_in_a]
        elif len(curr_seq_dims_in_a) == 1:
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_in_b]
        elif len(curr_seq_dims_in_b) == 1:
            self.sequence_dims[self.layout]["out"] = [curr_seq_dims_in_a]
        else:
            raise ValueError(
                f"[ERROR] failed to compute the dimension sequence of the output (xnnc ElemRealDiv): {curr_seq_dims_in_a}, {curr_seq_dims_in_b}."
            )


class XModelNodeUnsortedSegmentSum(XModelNode):
    """
    XModelNode UnsortedSegmentSum Protocol

    Computes the sum along segments of a tensor.

    Derived from:

    TensorFlow UnsortedSegmentSum tf.math.unsorted_segment_sum
    https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_sum
    """

    def __init__(self, op_name: str) -> NoReturn:
        assert op_name is not None, "'op_name' should not be None."
        XModelNode.__init__(self, op_name, "unsortedsegmentsum")
        # segment_ids
        self.__segment_ids: XTensor = None
        # num_segments
        self.__num_segments: XTensor = None

    @property
    def segment_ids(self) -> XTensor:
        return self.__segment_ids

    @segment_ids.setter
    def segment_ids(self, segment_ids: XTensor) -> NoReturn:
        assert segment_ids is not None, "'segment_ids' should not be None."
        assert isinstance(segment_ids, XTensor), "'segment_ids' should be an XTensor."
        self.__segment_ids = segment_ids

    @property
    def num_segments(self) -> XTensor:
        return self.__num_segments

    @num_segments.setter
    def num_segments(self, num_segments: XTensor) -> NoReturn:
        assert num_segments is not None, "'num_segments' should not be None."
        assert isinstance(num_segments, XTensor), "'num_segments' should be an XTensor object."
        self.__num_segments = num_segments

    def infer_shape(self, layout: Layout) -> NoReturn:
        assert layout is not None and isinstance(
            layout, Layout
        ), "'layout' should be a Layout enum value."
        assert (
            self.inputs_tensor is not None and len(self.inputs_tensor) == 3
        ), f"[ERROR] xnnc unsortedsegmentsum requires three inputs: actual: {len(self.inputs_tensor)}. op name: {self.op_name}"
        assert (
            self.inputs_tensor[0].data_format.name == layout.name
        ), f"[ERROR] xnnc {self.op_type} requires the layout of input tensor should be same as 'layout' argument. Op name: {self.op_name}."

        in_tensor = self.inputs_tensor[0]
        segment_ids = self.inputs_tensor[1]
        num_segments = self.inputs_tensor[2]

        if self.segment_ids is None:
            self.segment_ids = segment_ids

        if self.num_segments is None:
            self.num_segments = num_segments

        # compute outputs_tensor
        shape = self.num_segments.tolist() + in_tensor.shape[1:]

        self.outputs_tensor = [
            XTensor(np.zeros(shape,dtype=in_tensor.dtype), format=DataFormat[self.layout])
        ]

        # update layout
        if self.layout != layout.name:
            self.layout = layout.name

        # compute the dimension sequence of the output
        self.sequence_dims[self.layout]["out"] = [
            self.sequence_dims[self.layout]["in"][0]
        ]


REGISTERED_OPS = {
    "input_param": XModelNodeInput,
    "relu_param": XModelNodeRelu,
    "relu6_param": XModelNodeRelu6,
    "conv2d_param": XModelNodeConv2d,
    "maxpool2d_param": XModelNodeMaxPool,
    "avgpool2d_param": XModelNodeAvgPool,
    "matmul_param": XModelNodeMatMul,
    "const_param": XModelNodeConst,
    "elem_add_param": XModelNodeElemAdd,
    "flatten_param": XModelNodeFlatten,
    "pad_param": XModelNodePad,
    "mean_param": XModelNodeMean,
    "softmax_param": XModelNodeSoftmax,
    "conv2d_transpose_param": XModelNodeConv2dTranspose,
    "concat_param": XModelNodeConcat,
    "elem_mul_param": XModelNodeElemMul,
    "squeeze_param": XModelNodeSqueeze,
    "shape_param": XModelNodeShape,
    "reshape_param": XModelNodeReshape,
    "depthwise_conv2d_param": XModelNodeConv2dDepthwise,
    "strided_slice_param": XModelNodeStridedSlice,
    "identity_param": XModelNodeIdentity,
    "stack_param": XModelNodeStack,
    "resize_param": XModelNodeResize,
    "permute_param": XModelNodePermute,
    "spacetobatch_param": XModelNodeSpaceToBatchND,
    "batchtospace_param": XModelNodeBatchToSpaceND,
    "sigmoid": XModelNodeSigmoid,
    "fixneuron": XModelNodeFixNeuron,
    "unknown": XModelNodeUnknown,
    "expanddims": XModelNodeExpandDims,
    "pow": XModelNodeElemRealPow,
    "unsortedsegmentsum": XModelNodeUnsortedSegmentSum,
    "prelu_param": XModelNodePRelu,
}
