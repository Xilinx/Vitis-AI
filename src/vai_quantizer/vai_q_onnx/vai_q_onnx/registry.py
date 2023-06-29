#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from .operators.activation import QDQRemovableActivation, QLinearActivation
from onnxruntime.quantization.operators.argmax import QArgMax
from onnxruntime.quantization.operators.attention import AttentionQuant
from onnxruntime.quantization.operators.base_operator import QuantOperatorBase
from onnxruntime.quantization.operators.binary_op import QLinearBinaryOp
from onnxruntime.quantization.operators.concat import QLinearConcat
from onnxruntime.quantization.operators.conv import ConvInteger, QDQConv, QLinearConv
from onnxruntime.quantization.operators.direct_q8 import Direct8BitOp, QDQDirect8BitOp
from onnxruntime.quantization.operators.embed_layernorm import EmbedLayerNormalizationQuant
from onnxruntime.quantization.operators.gather import GatherQuant, QDQGather
from onnxruntime.quantization.operators.gavgpool import QGlobalAveragePool
from onnxruntime.quantization.operators.gemm import QDQGemm, QLinearGemm
from onnxruntime.quantization.operators.lstm import LSTMQuant
from onnxruntime.quantization.operators.matmul import MatMulInteger, QDQMatMul, QLinearMatMul
from onnxruntime.quantization.operators.maxpool import QDQMaxPool, QMaxPool
from onnxruntime.quantization.operators.pad import QPad
from onnxruntime.quantization.operators.pooling import QLinearPool
from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
from onnxruntime.quantization.operators.resize import QDQResize, QResize
from onnxruntime.quantization.operators.softmax import QDQSoftmax, QLinearSoftmax
from onnxruntime.quantization.operators.split import QDQSplit, QSplit
from onnxruntime.quantization.quant_utils import QuantizationMode

CommonOpsRegistry = {
    "Gather": GatherQuant,
    "Transpose": Direct8BitOp,
    "EmbedLayerNormalization": EmbedLayerNormalizationQuant,
}

IntegerOpsRegistry = {
    "Conv": ConvInteger,
    "MatMul": MatMulInteger,
    "Attention": AttentionQuant,
    "LSTM": LSTMQuant,
}
IntegerOpsRegistry.update(CommonOpsRegistry)

QLinearOpsRegistry = {
    "ArgMax": QArgMax,
    "Conv": QLinearConv,
    "Gemm": QLinearGemm,
    "MatMul": QLinearMatMul,
    "Add": QLinearBinaryOp,
    "Mul": QLinearBinaryOp,
    "Relu": QLinearActivation,
    "Clip": QLinearActivation,
    "LeakyRelu": QLinearActivation,
    "Sigmoid": QLinearActivation,
    "MaxPool": QMaxPool,
    "GlobalAveragePool": QGlobalAveragePool,
    "Split": QSplit,
    "Pad": QPad,
    "Reshape": Direct8BitOp,
    "Squeeze": Direct8BitOp,
    "Unsqueeze": Direct8BitOp,
    "Resize": QResize,
    "AveragePool": QLinearPool,
    "Concat": QLinearConcat,
    "Softmax": QLinearSoftmax,
}
QLinearOpsRegistry.update(CommonOpsRegistry)

QDQRegistry = {
    "Conv": QDQConv,
    "Gemm": QDQGemm,
    "Clip": QDQRemovableActivation,
    "Relu": QDQRemovableActivation,
    "Reshape": QDQDirect8BitOp,
    "Transpose": QDQDirect8BitOp,
    "Squeeze": QDQDirect8BitOp,
    "Unsqueeze": QDQDirect8BitOp,
    "Resize": QDQResize,
    "MaxPool": QDQMaxPool,
    "AveragePool": QDQDirect8BitOp,
    "MatMul": QDQMatMul,
    "Split": QDQSplit,
    "Gather": QDQGather,
    "Softmax": QDQSoftmax,
}


def CreateDefaultOpQuantizer(onnx_quantizer, node):
    return QuantOperatorBase(onnx_quantizer, node)


def CreateOpQuantizer(onnx_quantizer, node):
    registry = IntegerOpsRegistry if onnx_quantizer.mode == QuantizationMode.IntegerOps else QLinearOpsRegistry
    if node.op_type in registry.keys():
        op_quantizer = registry[node.op_type](onnx_quantizer, node)
        if op_quantizer.should_quantize():
            return op_quantizer
    return QuantOperatorBase(onnx_quantizer, node)


def CreateQDQQuantizer(onnx_quantizer, node):
    if node.op_type in QDQRegistry.keys():
        return QDQRegistry[node.op_type](onnx_quantizer, node)
    return QDQOperatorBase(onnx_quantizer, node)
