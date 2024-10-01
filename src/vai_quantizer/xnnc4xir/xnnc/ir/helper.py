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

from typing import Any

from xnnc.ir.enums import TargetType
from xnnc.proto.openir import open_tensor, openir
from xnnc.tensor.xtensor import XTensor


class SerdeFactory(object):
    @classmethod
    def serialize(
        cls,
        xnode: "XModelNode",
        target: TargetType,
    ) -> Any:

        if target is TargetType.OPENIR:

            return cls.to_openir_node(xnode)

        elif target is TargetType.XIR:

            raise NotImplementedError("SerdeFacotry.serialize")

    @classmethod
    def to_openir_node(cls, xnode: "XModelNode") -> openir.NodeProto:

        # create an open node
        node = openir.NodeProto()
        node.name = xnode.op_name
        node.kind = xnode.op_type

        class_name = cls.class_name(xnode)

        if class_name == "XModelNodeInput":
            input_param = openir.InputParameter()
            input_param.shape.dim.extend(xnode.shape)
            node.input_param.CopyFrom(input_param)

        elif class_name == "XModelNodeConst":
            const_param = openir.ConstParameter()
            const_param.tensor.CopyFrom(cls.create_tensor_proto(xnode.tensor))
            node.const_param.CopyFrom(const_param)

        elif class_name == "XModelNodeFixNeuron":
            qi = openir.QuantConfig.QuantInfo()
            qi.bit_width = xnode.quant_in.get("bit_width")
            qi.quant_pos = xnode.quant_in.get("quantize_pos")
            qi.round_mode = cls.to_openir_round_mode(xnode.quant_in.get("round_mode"))
            qi.signed = xnode.quant_in.get("signed")

            qc = openir.QuantConfig()
            qc.quant_in.CopyFrom(qi)
            qc.quant_out.CopyFrom(qi)

            # set quantization info
            node.quant_config.CopyFrom(qc)
            node.quantized = True

        elif class_name == "XModelNodePad":
            pad_param = openir.PadParameter()
            pad_param.mode = xnode.pad_mode
            pad_param.padding.extend(xnode.padding)
            pad_param.constant_values.extend(xnode.constant_values)
            node.pad_param.CopyFrom(pad_param)

        elif class_name == "XModelNodeConv2d":
            conv2d_param = openir.Conv2dParameter()

            # kernel_size: (kernel_h, kernel_w)
            conv2d_param.kernel.extend(xnode.kernel_size)
            # strides: (stride_h, stride_w)
            conv2d_param.strides.extend(xnode.strides)
            # pad_mode
            conv2d_param.pad_mode = openir.PadMode.Value(xnode.pad_mode.name)
            # dilation: (dilation_h, dilation_w)
            conv2d_param.dilation.extend(xnode.dilation[1:3])
            # round_mode
            conv2d_param.round_mode = openir.RoundMode.Value(xnode.round_mode.name)
            # group
            conv2d_param.group = xnode.group
            # output_channels
            conv2d_param.output_channels = xnode.num_output
            # weights: (h, w, ic, oc)
            conv2d_param.weights.CopyFrom(cls.create_tensor_proto(xnode.weights))
            # bias_term
            conv2d_param.bias_term = xnode.bias_term
            # bias
            if conv2d_param.bias_term:
                conv2d_param.bias.CopyFrom(cls.create_tensor_proto(xnode.bias))

            # set conv2d_param
            node.conv2d_param.CopyFrom(conv2d_param)

        elif class_name == "XModelNodeRelu":
            relu_param = openir.ReluParameter()
            relu_param.negative_slope = xnode.alpha
            node.relu_param.CopyFrom(relu_param)

        elif class_name == "XModelNodeRelu6":
            relu6_param = openir.Relu6Parameter()
            relu6_param.negative_slope = xnode.alpha
            node.relu6_param.CopyFrom(relu6_param)

        elif class_name == "XModelNodeMaxPool":
            pool2d_param = openir.MaxPool2dParameter()
            # kernel: (kernel_h, kernel_w)
            pool2d_param.kernel.extend(xnode.kernel_size)
            # strides: (stride_h, stride_w)
            pool2d_param.strides.extend(xnode.strides)
            # pad_mode
            pool2d_param.pad_mode = openir.PadMode.Value(xnode.pad_mode.name)
            # dilation: (h,w)
            pool2d_param.dilation.extend(xnode.dilation[1:3])
            # round_mode
            pool2d_param.round_mode = openir.RoundMode.Value(xnode.round_mode.name)
            node.maxpool2d_param.CopyFrom(pool2d_param)

        elif class_name == "XModelNodeAvgPool":
            pool2d_param = openir.AvgPool2dParameter()
            # zero_padding_included
            pool2d_param.zero_padding_included = xnode.zero_padding_included
            # count_include_invalid
            pool2d_param.count_include_invalid = xnode.count_include_invalid
            # kernel: (kernel_h, kernel_w)
            pool2d_param.kernel.extend(xnode.kernel_size)
            # strides: (stride_h, stride_w)
            pool2d_param.strides.extend(xnode.strides)
            # pad_mode
            pool2d_param.pad_mode = openir.PadMode.Value(xnode.pad_mode.name)
            # dilation: (h,w)
            pool2d_param.dilation.extend(xnode.dilation[1:3])
            # round_mode
            pool2d_param.round_mode = openir.RoundMode.Value(xnode.round_mode.name)
            node.avgpool2d_param.CopyFrom(pool2d_param)

        elif class_name == "XModelNodeElemAdd":
            elemadd_param = openir.ElemAddParameter()

            # set coefficients
            elemadd_param.coeffs.extend(xnode.alpha)
            node.elem_add_param.CopyFrom(elemadd_param)

        elif class_name == "XModelNodeMatMul":
            matmul_param = openir.MatMulParameter()

            matmul_param.transpose_a = xnode.transpose_a
            matmul_param.transpose_b = xnode.transpose_b

            node.matmul_param.CopyFrom(matmul_param)

        elif class_name == "XModelNodeSoftmax":
            softmax_param = openir.SoftmaxParameter()
            softmax_param.axis = xnode.axis

            node.softmax_param.CopyFrom(softmax_param)

        elif class_name == "XModelNodeConcat":
            concat_param = openir.ConcatParameter()
            # axis
            concat_param.axis = xnode.axis
            node.concat_param.CopyFrom(concat_param)

        elif class_name == "XModelNodeConv2dTranspose":
            conv2d_transpose_param = openir.Conv2dTransposeParameter()

            # (kernel_h, kernel_w)
            conv2d_transpose_param.kernel.extend(xnode.kernel_size)
            # (stride_h, stride_w)
            conv2d_transpose_param.strides.extend(xnode.strides)
            # pad_mode
            conv2d_transpose_param.pad_mode = openir.PadMode.Value(xnode.pad_mode.name)
            # dilation: (dilation_h, dilation_w)
            conv2d_transpose_param.dilation.extend(xnode.dilation[1:3])
            # round_mode
            conv2d_transpose_param.round_mode = openir.RoundMode.Value(
                xnode.round_mode.name
            )
            # group
            conv2d_transpose_param.group = xnode.group
            # output_shape: (output_h, output_w)
            conv2d_transpose_param.output_shape.extend(xnode.output_shape)

            # weights: (h, w, ic, oc)
            conv2d_transpose_param.weights.CopyFrom(
                cls.create_tensor_proto(xnode.weights)
            )

            # bias
            conv2d_transpose_param.bias_term = xnode.bias_term
            if conv2d_transpose_param.bias_term:
                conv2d_transpose_param.bias.CopyFrom(
                    cls.create_tensor_proto(xnode.bias)
                )

            node.conv2d_transpose_param.CopyFrom(conv2d_transpose_param)

        elif class_name == "XModelNodeConv2dDepthwise":
            depthwise_conv2d_param = openir.DepthwiseConv2dParameter()

            # kernel: (kernel_h, kernel_2)
            depthwise_conv2d_param.kernel.extend(xnode.kernel_size)
            # round_mode
            depthwise_conv2d_param.round_mode = openir.RoundMode.Value(
                xnode.round_mode.name
            )
            # strides: (stride_h, stride_w)
            depthwise_conv2d_param.strides.extend(xnode.strides)
            # dilation: (dilation_h, dilation_w)
            depthwise_conv2d_param.dilation.extend(xnode.dilation[1:3])
            # pad_mode
            depthwise_conv2d_param.pad_mode = openir.PadMode.Value(xnode.pad_mode.name)

            # weights: (h, w, ic, cm))
            depthwise_conv2d_param.weights.CopyFrom(
                cls.create_tensor_proto(xnode.weights)
            )

            # bias_term
            depthwise_conv2d_param.bias_term = xnode.bias_term
            # bias
            if depthwise_conv2d_param.bias_term:
                depthwise_conv2d_param.bias.CopyFrom(
                    cls.create_tensor_proto(xnode.bias)
                )

            # group
            depthwise_conv2d_param.group = xnode.group
            # output_channels
            depthwise_conv2d_param.output_channels = xnode.num_output

            node.depthwise_conv2d_param.CopyFrom(depthwise_conv2d_param)

        elif class_name == "XModelNodeReshape":
            reshape_param = openir.ReshapeParameter()

            # new shape
            reshape_param.new_shape.extend(xnode.shape)

            node.reshape_param.CopyFrom(reshape_param)

        elif class_name == "XModelNodeSigmoid":
            pass

        elif class_name == "XModelNodeElemMul":
            elem_mul_param = openir.ElemMulParameter()
            node.elem_mul_param.CopyFrom(elem_mul_param)

        elif class_name == "XModelNodeSqueeze":
            squeeze_param = openir.SqueezeParameter()
            squeeze_param.axis.extend(xnode.axis)

            node.squeeze_param.CopyFrom(squeeze_param)

        elif class_name == "XModelNodeResizeNearestNeighbor":
            resize_param = openir.ResizeParameter()

            # mode
            resize_param.mode = openir.ResizeParameter.Mode.NEAREST

            # align_corners
            resize_param.align_corners = xnode.align_corners

            node.resize_param.CopyFrom(resize_param)

        elif class_name == "XModelNodePermute":
            permute_param = openir.PermuteParameter()
            # order
            permute_param.order.extend(xnode.order)

            node.permute_param.CopyFrom(permute_param)

        elif class_name == "XModelNodeIdentity":
            identity_param = openir.IdentityParameter()
            node.identity_param.CopyFrom(identity_param)

        elif class_name == "XModelNodeResizeBiliear":
            resize_param = openir.ResizeParameter()

            # mode
            resize_param.mode = openir.ResizeParameter.Mode.BILINEAR

            # align_corners
            resize_param.align_corners = xnode.align_corners

            node.resize_param.CopyFrom(resize_param)

        else:
            # print(class_name)
            raise NotImplementedError(
                f"[serde]: Not supported XModelNode type: {class_name}"
            )

        # quantization
        if xnode.is_quantized:
            qc = openir.QuantConfig()

            if xnode.quant_in.get("bit_width"):
                qi_in = openir.QuantConfig.QuantInfo()
                qi_in.bit_width = xnode.quant_in.get("bit_width")
                qi_in.quant_pos = xnode.quant_in.get("quantize_pos")
                qi_in.round_mode = cls.to_openir_round_mode(
                    xnode.quant_in.get("round_mode")
                )
                qi_in.signed = xnode.quant_in.get("signed")
                qc.quant_in.CopyFrom(qi_in)

            if xnode.quant_out.get("bit_width"):
                qi_out = openir.QuantConfig.QuantInfo()
                qi_out.bit_width = xnode.quant_out.get("bit_width")
                qi_out.quant_pos = xnode.quant_out.get("quantize_pos")
                qi_out.round_mode = cls.to_openir_round_mode(
                    xnode.quant_out.get("round_mode")
                )
                qi_out.signed = xnode.quant_out.get("signed")
                qc.quant_out.CopyFrom(qi_out)

            if hasattr(xnode, "quant_weights") and xnode.quant_weights.get("bit_width"):
                qi_w = openir.QuantConfig.QuantInfo()
                qi_w.bit_width = xnode.quant_weights.get("bit_width")
                qi_w.quant_pos = xnode.quant_weights.get("quantize_pos")
                qi_w.round_mode = cls.to_openir_round_mode(
                    xnode.quant_weights.get("round_mode")
                )
                qi_w.signed = xnode.quant_weights.get("signed")

                qc.quant_weight.CopyFrom(qi_w)

            if (
                hasattr(xnode, "bias_term")
                and xnode.bias_term
                and hasattr(xnode, "quant_bias")
                and xnode.quant_bias.get("bit_width")
            ):
                qi_b = openir.QuantConfig.QuantInfo()
                qi_b.bit_width = xnode.quant_bias.get("bit_width")
                qi_b.quant_pos = xnode.quant_bias.get("quantize_pos")
                qi_b.round_mode = cls.to_openir_round_mode(
                    xnode.quant_bias.get("round_mode")
                )
                qi_b.signed = xnode.quant_bias.get("signed")

                qc.quant_weight.CopyFrom(qi_b)

            # set quantization info
            node.quant_config.CopyFrom(qc)
            node.quantized = True

        # inputs
        if len(xnode.bottom) > 0:
            node.inputs.extend(xnode.bottom)

        # outputs
        if len(xnode.top) > 0:
            node.outputs.extend(xnode.top)

        return node

    @classmethod
    def class_name(cls, obj: Any) -> str:
        assert obj is not None, "'obj' should not be None."
        return obj.__class__.__name__

    @classmethod
    def create_tensor_proto(cls, tensor: XTensor) -> open_tensor.TensorProto:
        assert tensor is not None and isinstance(
            tensor, XTensor
        ), f"'tensor' should an XTensor object."

        tp = open_tensor.TensorProto()
        tp.dtype = open_tensor.TensorProto.DataType.Value(tensor.dtype.name.upper())
        tp.shape.dim.extend(list(tensor.shape))
        tp.content = tensor.tobytes()

        return tp

    @classmethod
    def const_proto(
        cls, name: str, tensor: open_tensor.TensorProto
    ) -> openir.NodeProto:
        assert tensor is not None and isinstance(
            tensor, open_tensor.TensorProto
        ), f"'tensor' should be of openir TensorProto."

        const_param = openir.ConstParameter()
        const_param.tensor.CopyFrom(tensor)

        node = openir.NodeProto()
        node.name = name
        node.kind = "const"
        node.const_param.CopyFrom(const_param)

        return node

    @classmethod
    def to_openir_round_mode(cls, mode: int) -> openir.RoundMode:

        if mode == 0:
            return openir.RoundMode.STD_ROUND

        elif mode == 1:
            return openir.RoundMode.CEIL

        elif mode == 2:
            return openir.RoundMode.PYTHON3_ROUND

        else:
            raise ValueError(f"[ERROR] Unsupported round mode value: {mode}")
