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

import sys
from pathlib import Path

curr_path = Path(__file__).resolve()
PRJ_DIR = curr_path.parents[1]
sys.path.append(str(PRJ_DIR.resolve()))

import unittest

import numpy as np
from xnnc.ir.enums import Layout
from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import *


class ShapeInferenceTestCase(unittest.TestCase):
    # True: stop all test cases; otherwise, start all.
    stop_all = True

    def setUp(self) -> NoReturn:
        super().setUp()
        self.xmodel = XModel("dummy", "caffe", Layout.NCHW.name)

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeInput(self):
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.layout = Layout.NHWC.name
        xnode_input.shape = [1, 416, 416, 3]

        xnode_input.infer_shape(Layout.NHWC)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 3])

        # nhwc -> nchw
        self.assertTrue(Layout.NHWC.name == xnode_input.layout)
        xnode_input.infer_shape(Layout.NCHW)
        self.assertTrue(Layout.NCHW.name == xnode_input.layout)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 3, 416, 416])

        # nchw -> nhwc
        self.assertTrue(Layout.NCHW.name == xnode_input.layout)
        xnode_input.infer_shape(Layout.NHWC)
        self.assertTrue(Layout.NHWC.name == xnode_input.layout)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 3])

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeRelu(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 3, 416, 416]
        # relu
        xnode_relu = XModelNodeRelu("xnode_relu")

        xnode_relu.bottom = [xnode_input.op_name]
        xnode_input.top = [xnode_relu.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_relu])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 3])
        # relu
        self.assertTrue(xnode_relu.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_relu.outputs_tensor) == 1)
        self.assertTrue(xnode_relu.outputs_tensor[0].shape == [1, 416, 416, 3])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 3, 416, 416])
        # relu
        self.assertTrue(xnode_relu.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_relu.outputs_tensor) == 1)
        self.assertTrue(xnode_relu.outputs_tensor[0].shape == [1, 3, 416, 416])

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeConv2d(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 3, 416, 416]
        # conv2d
        xnode_conv2d = XModelNodeConv2d("xnode_conv2d", ksize=[3, 3])
        xnode_conv2d.dilation = [1] * 4
        xnode_conv2d.pad_mode = PadMode.SAME
        xnode_conv2d.strides = [1] * 2
        xnode_conv2d.weights = XTensor.zeros((3, 3, 3, 32))

        xnode_conv2d.bottom = [xnode_input.op_name]
        xnode_input.top = [xnode_conv2d.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_conv2d])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 3])
        # conv2d
        self.assertTrue(xnode_conv2d.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_conv2d.outputs_tensor) == 1)
        self.assertTrue(xnode_conv2d.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_conv2d.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 3, 416, 416])
        # conv2d
        self.assertTrue(xnode_conv2d.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_conv2d.outputs_tensor) == 1)
        self.assertTrue(xnode_conv2d.outputs_tensor[0].shape == [1, 32, 416, 416])
        self.assertTrue(
            xnode_conv2d.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeConv2dDepthwise(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 96, 112, 112]
        # depthwise_conv2d
        xnode_depthwise_conv2d = XModelNodeConv2dDepthwise("xnode_conv2d", ksize=[3, 3])
        xnode_depthwise_conv2d.dilation = [1] * 4
        xnode_depthwise_conv2d.pad_mode = PadMode.EXPLICIT
        xnode_depthwise_conv2d.padding = [1] * 4
        xnode_depthwise_conv2d.strides = [2] * 2
        xnode_depthwise_conv2d.num_output = 96
        xnode_depthwise_conv2d.group = 96
        xnode_depthwise_conv2d.round_mode = RoundMode.FLOOR
        xnode_depthwise_conv2d.weights = XTensor.zeros(
            (3, 3, 1, 96), format=DataFormat[self.xmodel.layout]
        )
        xnode_depthwise_conv2d.bias_term = True
        xnode_depthwise_conv2d.bias = XTensor.zeros(
            (96,), format=DataFormat[self.xmodel.layout]
        )

        xnode_depthwise_conv2d.bottom = [xnode_input.op_name]
        xnode_input.top = [xnode_depthwise_conv2d.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_depthwise_conv2d])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 112, 112, 96])
        # depthwise_conv2d
        self.assertTrue(xnode_depthwise_conv2d.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_depthwise_conv2d.outputs_tensor) == 1)
        self.assertTrue(
            xnode_depthwise_conv2d.outputs_tensor[0].shape == [1, 56, 56, 96]
        )
        self.assertTrue(xnode_depthwise_conv2d.weights.shape == [3, 3, 1, 96])
        self.assertTrue(xnode_depthwise_conv2d.weights.data_format == DataFormat.NCHW)

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 96, 112, 112])
        # depthwise_conv2d
        self.assertTrue(xnode_depthwise_conv2d.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_depthwise_conv2d.outputs_tensor) == 1)
        self.assertTrue(
            xnode_depthwise_conv2d.outputs_tensor[0].shape == [1, 96, 56, 56]
        )
        self.assertTrue(xnode_depthwise_conv2d.weights.shape == [3, 3, 1, 96])
        self.assertTrue(xnode_depthwise_conv2d.weights.data_format == DataFormat.NCHW)

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 112, 112, 96])
        # depthwise_conv2d
        self.assertTrue(xnode_depthwise_conv2d.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_depthwise_conv2d.outputs_tensor) == 1)
        self.assertTrue(
            xnode_depthwise_conv2d.outputs_tensor[0].shape == [1, 56, 56, 96]
        )
        self.assertTrue(xnode_depthwise_conv2d.weights.shape == [3, 3, 1, 96])
        self.assertTrue(xnode_depthwise_conv2d.weights.data_format == DataFormat.NCHW)

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeDeconvolution(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 32, 8, 16]
        # deconvolution
        xnode_deconv = XModelNodeDeconvolution("xnode_deconv", ksize=[4, 4])
        xnode_deconv.dilation = [1] * 4
        xnode_deconv.pad_mode = PadMode.EXPLICIT
        xnode_deconv.padding = [1] * 4
        xnode_deconv.strides = [2] * 2
        xnode_deconv.weights = XTensor.zeros(
            (4, 4, 32, 32), dtype=np.float32, format=DataFormat[self.xmodel.layout]
        )
        xnode_deconv.num_output = 32

        xnode_deconv.bottom = [xnode_input.op_name]
        xnode_input.top = [xnode_deconv.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_deconv])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 8, 16, 32])
        # deconvolution
        self.assertTrue(xnode_deconv.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_deconv.outputs_tensor) == 1)
        self.assertTrue(xnode_deconv.outputs_tensor[0].shape == [1, 16, 32, 32])
        self.assertTrue(xnode_deconv.outputs_tensor[0].data_format == DataFormat.NHWC)

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 32, 8, 16])
        # deconvolution
        self.assertTrue(xnode_deconv.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_deconv.outputs_tensor) == 1)
        self.assertTrue(xnode_deconv.outputs_tensor[0].shape == [1, 32, 16, 32])
        self.assertTrue(xnode_deconv.outputs_tensor[0].data_format == DataFormat.NCHW)

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeConv2dTranspose(self):
        self.xmodel = XModel("dummy", "tensorflow2", Layout.NHWC.name)
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 8, 8, 512]
        # conv2d_transpose
        xnode_conv2d_transpose = XModelNodeConv2dTranspose(
            "xnode_conv2d_transpose", ksize=[2, 2]
        )
        xnode_conv2d_transpose.dilation = [1] * 4
        xnode_conv2d_transpose.pad_mode = PadMode.SAME
        xnode_conv2d_transpose.strides = [2] * 2
        xnode_conv2d_transpose.weights = XTensor.zeros((2, 2, 128, 512))
        xnode_conv2d_transpose.group = 1

        xnode_conv2d_transpose.bottom = [xnode_input.op_name]
        xnode_input.top = [xnode_conv2d_transpose.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_conv2d_transpose])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 512, 8, 8])
        # conv2d_transpose
        self.assertTrue(xnode_conv2d_transpose.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_conv2d_transpose.outputs_tensor) == 1)
        self.assertTrue(
            xnode_conv2d_transpose.outputs_tensor[0].shape == [1, 128, 16, 16]
        )
        self.assertTrue(
            xnode_conv2d_transpose.outputs_tensor[0].data_format.name
            == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 8, 8, 512])
        # conv2d_transpose
        self.assertTrue(xnode_conv2d_transpose.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_conv2d_transpose.outputs_tensor) == 1)
        self.assertTrue(
            xnode_conv2d_transpose.outputs_tensor[0].shape == [1, 16, 16, 128]
        )
        self.assertTrue(
            xnode_conv2d_transpose.outputs_tensor[0].data_format.name
            == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodePad(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 32, 416, 416]
        # pad
        xnode_pad = XModelNodePad(
            "xnode_pad",
            pad_num=[0, 0, 0, 0, 1, 0, 1, 0],
            mode="constant",
            constant_values=[0.0] * 8,
        )

        xnode_input.top = [xnode_pad.op_name]
        xnode_pad.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_pad])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 32])
        # pad
        self.assertTrue(xnode_pad.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_pad.outputs_tensor) == 1)
        self.assertTrue(xnode_pad.outputs_tensor[0].shape == [1, 417, 417, 32])

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeFixNeuron(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 3, 416, 416]
        # pad
        xnode_fix = XModelNodeFixNeuron("xnode_fix")

        xnode_input.top = [xnode_fix.op_name]
        xnode_fix.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_fix])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 3])
        # pad
        self.assertTrue(xnode_fix.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_fix.outputs_tensor) == 1)
        self.assertTrue(xnode_fix.outputs_tensor[0].shape == [1, 416, 416, 3])

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeElemAdd(self):
        # input_1
        xnode_input_1 = XModelNodeInput("xnode_input_1")
        xnode_input_1.shape = [1, 64, 208, 208]
        # input_2
        xnode_input_2 = XModelNodeInput("xnode_input_2")
        xnode_input_2.shape = [1, 64, 208, 208]
        # elem_add
        xnode_add = XModelNodeElemAdd("xnode_add")

        xnode_input_1.top = [xnode_add.op_name]
        xnode_input_2.top = [xnode_add.op_name]
        xnode_add.bottom = [xnode_input_1.op_name, xnode_input_2.op_name]
        self.xmodel.add_xnodes([xnode_input_1, xnode_input_2, xnode_add])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input_1
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 208, 208, 64])
        # input_2
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 208, 208, 64])
        # pad
        self.assertTrue(xnode_add.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_add.outputs_tensor) == 1)
        self.assertTrue(xnode_add.outputs_tensor[0].shape == [1, 208, 208, 64])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input_1
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 64, 208, 208])
        # input_2
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 64, 208, 208])
        # pad
        self.assertTrue(xnode_add.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_add.outputs_tensor) == 1)
        self.assertTrue(xnode_add.outputs_tensor[0].shape == [1, 64, 208, 208])

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeConst(self):
        xnode_const = XModelNodeConst("xnode_input")
        xnode_const.tensor = XTensor.zeros(
            [1, 3, 416, 416], format=DataFormat[self.xmodel.layout]
        )
        self.xmodel.add_xnode(xnode_const)

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # const
        self.assertTrue(xnode_const.layout == Layout.NHWC.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NHWC.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [1, 416, 416, 3])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # const
        self.assertTrue(xnode_const.layout == Layout.NCHW.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NCHW.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [1, 3, 416, 416])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # const
        self.assertTrue(xnode_const.layout == Layout.NHWC.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NHWC.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [1, 416, 416, 3])

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeResize_XModelNodeConst(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 256, 13, 13]
        # size
        xnode_const = XModelNodeConst("xnode_const")
        xnode_const.tensor = XTensor(
            np.array([26, 26], dtype=np.int32), format=DataFormat[self.xmodel.layout]
        )
        # resize
        xnode_resize = XModelNodeResize("xnode_resize_nearest", mode="nearest")
        xnode_resize.align_corners = False
        xnode_resize.half_pixel_centers = False

        xnode_input.top = [xnode_resize.op_name]
        xnode_const.top = [xnode_resize.op_name]
        xnode_resize.bottom = [xnode_input.op_name, xnode_const.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_const, xnode_resize])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 13, 13, 256])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # const
        self.assertTrue(xnode_const.layout == Layout.NHWC.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NHWC.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [2])
        self.assertTrue(
            xnode_const.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # resize
        self.assertTrue(xnode_resize.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_resize.outputs_tensor) == 1)
        self.assertTrue(xnode_resize.outputs_tensor[0].shape == [1, 26, 26, 256])
        self.assertTrue(
            xnode_resize.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeFlatten_XModelNodeReshape_XModelNodeConst(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 65744, 1, 1]
        # flatten
        xnode_flatten = XModelNodeFlatten("xnode_flatten")
        xnode_flatten.start_dim = 1
        xnode_flatten.end_dim = 3
        # shape
        xnode_const = XModelNodeConst("xnode_const")
        xnode_const.tensor = XTensor(
            np.array([1, 16436, -1], dtype=np.int32),
            format=DataFormat[self.xmodel.layout],
        )
        # reshape
        xnode_reshape = XModelNodeReshape("xnode_resize_nearest")

        xnode_input.top = [xnode_flatten.op_name]
        xnode_flatten.bottom = [xnode_input.op_name]
        xnode_flatten.top = [xnode_reshape.op_name]
        xnode_const.top = [xnode_reshape.op_name]
        xnode_reshape.bottom = [xnode_flatten.op_name, xnode_const.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_flatten, xnode_const, xnode_reshape])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1, 1, 65744])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # flatten
        self.assertTrue(xnode_flatten.layout == Layout.NHWC.name)
        self.assertTrue(xnode_flatten.start_dim == 1)
        self.assertTrue(xnode_flatten.end_dim == 3)
        self.assertTrue(len(xnode_flatten.outputs_tensor) == 1)
        self.assertTrue(xnode_flatten.outputs_tensor[0].shape == [1, 65744])
        self.assertTrue(
            xnode_flatten.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # const
        self.assertTrue(xnode_const.layout == Layout.NHWC.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NHWC.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [3])
        self.assertTrue(
            xnode_const.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # reshape
        self.assertTrue(xnode_reshape.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_reshape.outputs_tensor) == 1)
        self.assertTrue(xnode_reshape.outputs_tensor[0].shape == [1, 16436, 4])
        self.assertTrue(
            xnode_reshape.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 65744, 1, 1])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # flatten
        self.assertTrue(xnode_flatten.layout == Layout.NCHW.name)
        self.assertTrue(xnode_flatten.start_dim == 1)
        self.assertTrue(xnode_flatten.end_dim == 3)
        self.assertTrue(len(xnode_flatten.outputs_tensor) == 1)
        self.assertTrue(xnode_flatten.outputs_tensor[0].shape == [1, 65744])
        self.assertTrue(
            xnode_flatten.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # const
        self.assertTrue(xnode_const.layout == Layout.NCHW.name)
        self.assertTrue(xnode_const.tensor.data_format.name == Layout.NCHW.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [3])
        self.assertTrue(
            xnode_const.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # reshape
        self.assertTrue(xnode_reshape.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_reshape.outputs_tensor) == 1)
        self.assertTrue(xnode_reshape.outputs_tensor[0].shape == [1, 16436, 4])
        self.assertTrue(
            xnode_reshape.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeConcat(self):
        # input_1
        xnode_input_1 = XModelNodeInput("xnode_input_1")
        xnode_input_1.shape = [1, 512, 26, 26]
        # input_2
        xnode_input_2 = XModelNodeInput("xnode_input_2")
        xnode_input_2.shape = [1, 256, 26, 26]
        # concat
        xnode_concat = XModelNodeConcat("xnode_concat")
        xnode_concat.axis = 1

        xnode_input_1.top = [xnode_concat.op_name]
        xnode_input_2.top = [xnode_concat.op_name]
        xnode_concat.bottom = [xnode_input_1.op_name, xnode_input_2.op_name]
        self.xmodel.add_xnodes([xnode_input_1, xnode_input_2, xnode_concat])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(xnode_concat.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input_1
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 26, 26, 512])
        self.assertTrue(
            xnode_input_1.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # input_1
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 26, 26, 256])
        self.assertTrue(
            xnode_input_2.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # concat
        self.assertTrue(xnode_concat.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_concat.outputs_tensor) == 1)
        self.assertTrue(xnode_concat.outputs_tensor[0].shape == [1, 26, 26, 768])
        self.assertTrue(
            xnode_concat.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        self.assertTrue(
            xnode_concat.axis == 3, f"expected:{3}, actual:{xnode_concat.axis}"
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(xnode_concat.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input_1
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 512, 26, 26])
        self.assertTrue(
            xnode_input_1.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # input_1
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 256, 26, 26])
        self.assertTrue(
            xnode_input_2.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # concat
        self.assertTrue(xnode_concat.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_concat.outputs_tensor) == 1)
        self.assertTrue(xnode_concat.outputs_tensor[0].shape == [1, 768, 26, 26])
        self.assertTrue(
            xnode_concat.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        self.assertTrue(
            xnode_concat.axis == 1, f"expected:{1}, actual:{xnode_concat.axis}"
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(xnode_concat.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input_1
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 26, 26, 512])
        self.assertTrue(
            xnode_input_1.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # input_1
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 26, 26, 256])
        self.assertTrue(
            xnode_input_2.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # concat
        self.assertTrue(xnode_concat.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_concat.outputs_tensor) == 1)
        self.assertTrue(xnode_concat.outputs_tensor[0].shape == [1, 26, 26, 768])
        self.assertTrue(
            xnode_concat.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        self.assertTrue(
            xnode_concat.axis == 3, f"expected:{3}, actual:{xnode_concat.axis}"
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeMaxPool(self):
        # input
        xnode_input = XModelNodeInput("xnode_input_1")
        xnode_input.shape = [1, 64, 112, 112]
        # maxpool
        xnode_maxpool = XModelNodeMaxPool("xnode_maxpool", ksize=[3, 3])
        xnode_maxpool.strides = [2, 2]
        xnode_maxpool.round_mode = RoundMode.CEIL

        xnode_input.top = [xnode_maxpool.op_name]
        xnode_maxpool.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_maxpool])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_maxpool.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 112, 112, 64])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # maxpool
        self.assertTrue(xnode_maxpool.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_maxpool.outputs_tensor) == 1)
        self.assertTrue(xnode_maxpool.outputs_tensor[0].shape == [1, 56, 56, 64])
        self.assertTrue(
            xnode_maxpool.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_maxpool.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 64, 112, 112])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # maxpool
        self.assertTrue(xnode_maxpool.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_maxpool.outputs_tensor) == 1)
        self.assertTrue(xnode_maxpool.outputs_tensor[0].shape == [1, 64, 56, 56])
        self.assertTrue(
            xnode_maxpool.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeAvgPool(self):
        # input_1
        xnode_input = XModelNodeInput("xnode_input_1")
        xnode_input.shape = [1, 1024, 7, 7]
        # avgpool
        xnode_avgpool = XModelNodeAvgPool("xnode_avgpool", ksize=[7, 7])
        xnode_avgpool.strides = [1, 1]
        xnode_avgpool.round_mode = RoundMode.CEIL

        xnode_input.top = [xnode_avgpool.op_name]
        xnode_avgpool.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_avgpool])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_avgpool.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 7, 7, 1024])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # avgpool
        self.assertTrue(xnode_avgpool.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_avgpool.outputs_tensor) == 1)
        self.assertTrue(xnode_avgpool.outputs_tensor[0].shape == [1, 1, 1, 1024])
        self.assertTrue(
            xnode_avgpool.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_avgpool.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1024, 7, 7])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # avgpool
        self.assertTrue(xnode_avgpool.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_avgpool.outputs_tensor) == 1)
        self.assertTrue(xnode_avgpool.outputs_tensor[0].shape == [1, 1024, 1, 1])
        self.assertTrue(
            xnode_avgpool.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeDot(self):
        self.xmodel = XModel("dummy", "caffe", Layout.NCHW.name)
        # input
        xnode_input = XModelNodeInput("xnode_input_1")
        xnode_input.shape = [1, 1024, 1, 1]
        # dot
        xnode_dot = XModelNodeDot("xnode_dot")
        xnode_dot.axis = 1
        xnode_dot.num_output = 1000
        xnode_dot.transpose = False
        xnode_dot.weights = XTensor.ones(
            shape=(1000, 1024), format=DataFormat[self.xmodel.layout]
        )
        xnode_dot.bias_term = True
        xnode_dot.bias = XTensor.zeros(
            shape=(1000,), format=DataFormat[self.xmodel.layout]
        )

        xnode_input.top = [xnode_dot.op_name]
        xnode_dot.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_dot])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_dot.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1, 1, 1024])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # dot
        self.assertTrue(xnode_dot.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_dot.outputs_tensor) == 1)
        self.assertTrue(xnode_dot.outputs_tensor[0].shape == [1, 1, 1, 1000])
        self.assertTrue(
            xnode_dot.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        self.assertTrue(xnode_dot.weights.shape == [1000, 1024])
        self.assertTrue(xnode_dot.weights.data_format == DataFormat.NHWC)

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_dot.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1024, 1, 1])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # dot
        self.assertTrue(xnode_dot.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_dot.outputs_tensor) == 1)
        self.assertTrue(xnode_dot.outputs_tensor[0].shape == [1, 1000, 1, 1])
        self.assertTrue(
            xnode_dot.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        self.assertTrue(xnode_dot.weights.shape == [1000, 1024])
        self.assertTrue(xnode_dot.weights.data_format == DataFormat.NCHW)

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeSoftmax(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 1000, 1, 1]
        # softmax
        xnode_softmax = XModelNodeSoftmax("xnode_softmax")
        xnode_softmax.axis = 1

        xnode_input.top = [xnode_softmax.op_name]
        xnode_softmax.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_softmax])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_softmax.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1, 1, 1000])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # softmax
        self.assertTrue(xnode_softmax.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_softmax.outputs_tensor) == 1)
        self.assertTrue(xnode_softmax.outputs_tensor[0].shape == [1, 1, 1, 1000])
        self.assertTrue(
            xnode_softmax.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        self.assertTrue(
            xnode_softmax.axis == 3, f"expected:{3}, actual:{xnode_softmax.axis}"
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_softmax.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1000, 1, 1])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # softmax
        self.assertTrue(xnode_softmax.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_softmax.outputs_tensor) == 1)
        self.assertTrue(xnode_softmax.outputs_tensor[0].shape == [1, 1000, 1, 1])
        self.assertTrue(
            xnode_softmax.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        self.assertTrue(
            xnode_softmax.axis == 1, f"expected:{1}, actual:{xnode_softmax.axis}"
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_softmax.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1, 1, 1000])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # softmax
        self.assertTrue(xnode_softmax.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_softmax.outputs_tensor) == 1)
        self.assertTrue(xnode_softmax.outputs_tensor[0].shape == [1, 1, 1, 1000])
        self.assertTrue(
            xnode_softmax.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        self.assertTrue(
            xnode_softmax.axis == 3, f"expected:{3}, actual:{xnode_softmax.axis}"
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodePermute(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 88, 45, 60]
        # permute
        xnode_permute = XModelNodePermute("xnode_permute")
        xnode_permute.order = [0, 2, 3, 1]

        xnode_input.top = [xnode_permute.op_name]
        xnode_permute.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_permute])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_permute.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 45, 60, 88])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # permute
        self.assertTrue(xnode_permute.layout == Layout.NHWC.name)
        self.assertTrue(
            xnode_permute.order == [0, 1, 2, 3],
            f"expected:[0, 1, 2, 3], actual:{xnode_permute.order}",
        )
        self.assertTrue(len(xnode_permute.outputs_tensor) == 1)
        self.assertTrue(xnode_permute.outputs_tensor[0].shape == [1, 45, 60, 88])
        self.assertTrue(
            xnode_permute.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_permute.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 88, 45, 60])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # permute
        self.assertTrue(xnode_permute.layout == Layout.NCHW.name)
        self.assertTrue(
            xnode_permute.order == [0, 2, 3, 1],
            f"expected:[0, 2, 3, 1], actual:{xnode_permute.order}",
        )
        self.assertTrue(len(xnode_permute.outputs_tensor) == 1)
        self.assertTrue(xnode_permute.outputs_tensor[0].shape == [1, 45, 60, 88])
        self.assertTrue(
            xnode_permute.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_permute.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 45, 60, 88])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # permute
        self.assertTrue(xnode_permute.layout == Layout.NHWC.name)
        self.assertTrue(
            xnode_permute.order == [0, 1, 2, 3],
            f"expected:[0, 1, 2, 3], actual:{xnode_permute.order}",
        )
        self.assertTrue(len(xnode_permute.outputs_tensor) == 1)
        self.assertTrue(xnode_permute.outputs_tensor[0].shape == [1, 45, 60, 88])
        self.assertTrue(
            xnode_permute.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeDeephiResize(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 26, 8, 16]
        # deephi_resize
        xnode_deephi_resize = XModelNodeDeephiResize("xnode_deephi_resize")
        xnode_deephi_resize.scale = [2, 2]
        xnode_deephi_resize.mode = "bilinear"

        xnode_input.top = [xnode_deephi_resize.op_name]
        xnode_deephi_resize.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_deephi_resize])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_deephi_resize.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 8, 16, 26])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # deephi_resize
        self.assertTrue(xnode_deephi_resize.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_deephi_resize.outputs_tensor) == 1)
        self.assertTrue(xnode_deephi_resize.outputs_tensor[0].shape == [1, 16, 32, 26])
        self.assertTrue(
            xnode_deephi_resize.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_deephi_resize.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 26, 8, 16])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # deephi_resize
        self.assertTrue(xnode_deephi_resize.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_deephi_resize.outputs_tensor) == 1)
        self.assertTrue(xnode_deephi_resize.outputs_tensor[0].shape == [1, 26, 16, 32])
        self.assertTrue(
            xnode_deephi_resize.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_deephi_resize.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 8, 16, 26])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # deephi_resize
        self.assertTrue(xnode_deephi_resize.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_deephi_resize.outputs_tensor) == 1)
        self.assertTrue(xnode_deephi_resize.outputs_tensor[0].shape == [1, 16, 32, 26])
        self.assertTrue(
            xnode_deephi_resize.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeReorg(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 64, 28, 28]
        # reorg
        xnode_reorg = XModelNodeReorg("xnode_reorg", [2, 2])
        xnode_reorg.reverse = False

        xnode_input.top = [xnode_reorg.op_name]
        xnode_reorg.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_reorg])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_reorg.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 28, 28, 64])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # reorg
        self.assertTrue(xnode_reorg.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_reorg.outputs_tensor) == 1)
        self.assertTrue(xnode_reorg.outputs_tensor[0].shape == [1, 14, 14, 256])
        self.assertTrue(
            xnode_reorg.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_reorg.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 64, 28, 28])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # reorg
        self.assertTrue(xnode_reorg.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_reorg.outputs_tensor) == 1)
        self.assertTrue(xnode_reorg.outputs_tensor[0].shape == [1, 256, 14, 14])
        self.assertTrue(
            xnode_reorg.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_reorg.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 28, 28, 64])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # reorg
        self.assertTrue(xnode_reorg.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_reorg.outputs_tensor) == 1)
        self.assertTrue(xnode_reorg.outputs_tensor[0].shape == [1, 14, 14, 256])
        self.assertTrue(
            xnode_reorg.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeGSTiling(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 256, 15, 20]
        # gstiling
        xnode_gstiling = XModelNodeGSTiling("xnode_gstiling")
        xnode_gstiling.reverse = True
        xnode_gstiling.stride = 8

        xnode_input.top = [xnode_gstiling.op_name]
        xnode_gstiling.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_gstiling])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_gstiling.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 15, 20, 256])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # gstiling
        self.assertTrue(xnode_gstiling.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_gstiling.outputs_tensor) == 1)
        self.assertTrue(xnode_gstiling.outputs_tensor[0].shape == [1, 120, 160, 4])
        self.assertTrue(
            xnode_gstiling.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_gstiling.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 256, 15, 20])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # gstiling
        self.assertTrue(xnode_gstiling.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_gstiling.outputs_tensor) == 1)
        self.assertTrue(xnode_gstiling.outputs_tensor[0].shape == [1, 4, 120, 160])
        self.assertTrue(
            xnode_gstiling.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_gstiling.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 15, 20, 256])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # gstiling
        self.assertTrue(xnode_gstiling.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_gstiling.outputs_tensor) == 1)
        self.assertTrue(xnode_gstiling.outputs_tensor[0].shape == [1, 120, 160, 4])
        self.assertTrue(
            xnode_gstiling.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodePriorBox(self):
        # input
        xnode_input_1 = XModelNodeInput("xnode_input_1")
        xnode_input_1.shape = [1, 154, 45, 60]
        xnode_input_2 = XModelNodeInput("xnode_input_2")
        xnode_input_2.shape = [1, 3, 360, 480]
        # priorbox
        xnode_priorbox = XModelNodePriorBox(
            "xnode_priorbox", min_sizes=[21.0], max_sizes=[45.0]
        )
        xnode_priorbox.aspect_ratio = [2.0]
        xnode_priorbox.flip = True
        xnode_priorbox.clip = False
        xnode_priorbox.variance = [0.1, 0.1, 0.2, 0.2]
        xnode_priorbox.step = [8.0]
        xnode_priorbox.offset = 0.5

        xnode_input_1.top = [xnode_priorbox.op_name]
        xnode_input_2.top = [xnode_priorbox.op_name]
        xnode_priorbox.bottom = [xnode_input_1.op_name, xnode_input_2.op_name]
        self.xmodel.add_xnodes([xnode_input_1, xnode_input_2, xnode_priorbox])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(xnode_priorbox.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 45, 60, 154])
        self.assertTrue(
            xnode_input_1.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # input
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 360, 480, 3])
        self.assertTrue(
            xnode_input_2.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # priorbox
        self.assertTrue(xnode_priorbox.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_priorbox.outputs_tensor) == 1)
        self.assertTrue(xnode_priorbox.outputs_tensor[0].shape == [1, 43200, 2])
        self.assertTrue(
            xnode_priorbox.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(xnode_priorbox.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 154, 45, 60])
        self.assertTrue(
            xnode_input_1.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # input
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 3, 360, 480])
        self.assertTrue(
            xnode_input_2.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # priorbox
        self.assertTrue(xnode_priorbox.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_priorbox.outputs_tensor) == 1)
        self.assertTrue(xnode_priorbox.outputs_tensor[0].shape == [1, 2, 43200])
        self.assertTrue(
            xnode_priorbox.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(xnode_priorbox.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 45, 60, 154])
        self.assertTrue(
            xnode_input_1.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # input
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 360, 480, 3])
        self.assertTrue(
            xnode_input_2.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # priorbox
        self.assertTrue(xnode_priorbox.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_priorbox.outputs_tensor) == 1)
        self.assertTrue(xnode_priorbox.outputs_tensor[0].shape == [1, 43200, 2])
        self.assertTrue(
            xnode_priorbox.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeScale(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 32, 256, 512]
        # const
        xnode_const = XModelNodeConst("xnode_const")
        xnode_const.tensor = XTensor.zeros(
            (32,), dtype=np.float32, format=DataFormat.NCHW
        )
        # scale
        xnode_scale = XModelNodeScale("xnode_scale")
        xnode_scale.axis = 1

        xnode_input.top = [xnode_scale.op_name]
        xnode_const.top = [xnode_scale.op_name]
        xnode_scale.bottom = [xnode_input.op_name, xnode_const.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_const, xnode_scale])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_const.layout == Layout.NCHW.name)
        self.assertTrue(xnode_scale.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 256, 512, 32])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # const
        self.assertTrue(xnode_const.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [32])
        self.assertTrue(
            xnode_const.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # scale
        self.assertTrue(xnode_scale.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_scale.outputs_tensor) == 1)
        self.assertTrue(xnode_scale.outputs_tensor[0].shape == [1, 256, 512, 32])
        self.assertTrue(
            xnode_scale.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_const.layout == Layout.NHWC.name)
        self.assertTrue(xnode_scale.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 32, 256, 512])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # const
        self.assertTrue(xnode_const.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [32])
        self.assertTrue(
            xnode_const.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # scale
        self.assertTrue(xnode_scale.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_scale.outputs_tensor) == 1)
        self.assertTrue(xnode_scale.outputs_tensor[0].shape == [1, 32, 256, 512])
        self.assertTrue(
            xnode_scale.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_const.layout == Layout.NCHW.name)
        self.assertTrue(xnode_scale.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 256, 512, 32])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # const
        self.assertTrue(xnode_const.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [32])
        self.assertTrue(
            xnode_const.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # scale
        self.assertTrue(xnode_scale.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_scale.outputs_tensor) == 1)
        self.assertTrue(xnode_scale.outputs_tensor[0].shape == [1, 256, 512, 32])
        self.assertTrue(
            xnode_scale.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeMatMul(self):
        self.xmodel = XModel("dummy", "tensorflow2", Layout.NHWC.name)
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 1, 1, 2048]
        # const
        xnode_const = XModelNodeConst("xnode_const")
        xnode_const.tensor = XTensor.zeros(
            (2048, 1000), dtype=np.float32, format=DataFormat[self.xmodel.layout]
        )
        # matmul
        xnode_matmul = XModelNodeMatMul("xnode_matmul")

        xnode_input.top = [xnode_matmul.op_name]
        xnode_const.top = [xnode_matmul.op_name]
        xnode_matmul.bottom = [xnode_input.op_name, xnode_const.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_const, xnode_matmul])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_const.layout == Layout.NHWC.name)
        self.assertTrue(xnode_matmul.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 2048, 1, 1])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # const
        self.assertTrue(xnode_const.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [2048, 1000])
        self.assertTrue(
            xnode_const.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # matmul
        self.assertTrue(xnode_matmul.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_matmul.outputs_tensor) == 1)
        self.assertTrue(xnode_matmul.outputs_tensor[0].shape == [1, 1000, 1, 1])
        self.assertTrue(
            xnode_matmul.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeUpsample(self):
        self.xmodel = XModel("dummy", "tensorflow2", Layout.NHWC.name)
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 36, 36, 512]
        # upsample
        xnode_upsample = XModelNodeUpsample("xnode_scale")
        xnode_upsample.scale = [1, 1, 2, 2]
        xnode_upsample.mode = "nearest"

        xnode_input.top = [xnode_upsample.op_name]
        xnode_upsample.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_upsample])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_upsample.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 512, 36, 36])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # upsample
        self.assertTrue(xnode_upsample.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_upsample.outputs_tensor) == 1)
        self.assertTrue(xnode_upsample.outputs_tensor[0].shape == [1, 512, 72, 72])
        self.assertTrue(
            xnode_upsample.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_upsample.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 36, 36, 512])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # upsample
        self.assertTrue(xnode_upsample.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_upsample.outputs_tensor) == 1)
        self.assertTrue(xnode_upsample.outputs_tensor[0].shape == [1, 72, 72, 512])
        self.assertTrue(
            xnode_upsample.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeSigmoid(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 1, 576, 576]
        # sigmoid
        xnode_sigmoid = XModelNodeSigmoid("xnode_sigmoid")

        xnode_input.top = [xnode_sigmoid.op_name]
        xnode_sigmoid.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_sigmoid])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_sigmoid.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 576, 576, 1])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # sigmoid
        self.assertTrue(xnode_sigmoid.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_sigmoid.outputs_tensor) == 1)
        self.assertTrue(xnode_sigmoid.outputs_tensor[0].shape == [1, 576, 576, 1])
        self.assertTrue(
            xnode_sigmoid.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_sigmoid.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1, 576, 576])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # sigmoid
        self.assertTrue(xnode_sigmoid.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_sigmoid.outputs_tensor) == 1)
        self.assertTrue(xnode_sigmoid.outputs_tensor[0].shape == [1, 1, 576, 576])
        self.assertTrue(
            xnode_sigmoid.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeShape(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 1, 576, 576]
        # shape
        xnode_shape = XModelNodeShape("xnode_shape")

        xnode_input.top = [xnode_shape.op_name]
        xnode_shape.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_shape])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_shape.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 576, 576, 1])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # shape
        self.assertTrue(xnode_shape.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_shape.outputs_tensor) == 1)
        self.assertTrue(xnode_shape.outputs_tensor[0].tolist() == [1, 576, 576, 1])
        self.assertTrue(xnode_shape.outputs_tensor[0].shape == [4])
        self.assertTrue(
            xnode_shape.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_shape.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1, 576, 576])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # shape
        self.assertTrue(xnode_shape.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_shape.outputs_tensor) == 1)
        self.assertTrue(xnode_shape.outputs_tensor[0].tolist() == [1, 1, 576, 576])
        self.assertTrue(xnode_shape.outputs_tensor[0].shape == [4])
        self.assertTrue(
            xnode_shape.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeElemMul(self):
        self.xmodel = XModel("dummy", "tensorflow2", Layout.NHWC.name)
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 1, 1, 2048]
        # const
        xnode_const = XModelNodeConst("xnode_const")
        xnode_const.tensor = XTensor(
            np.array([1.0048828125], dtype=np.float32),
            format=DataFormat[self.xmodel.layout],
        )
        # elemmul
        xnode_elemmul = XModelNodeElemMul("xnode_elemmul")

        xnode_input.top = [xnode_elemmul.op_name]
        xnode_const.top = [xnode_elemmul.op_name]
        xnode_elemmul.bottom = [xnode_input.op_name, xnode_const.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_const, xnode_elemmul])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_const.layout == Layout.NHWC.name)
        self.assertTrue(xnode_elemmul.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 2048, 1, 1])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # const
        self.assertTrue(xnode_const.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [1])
        self.assertTrue(xnode_const.outputs_tensor[0].tolist() == [1.0048828125])
        self.assertTrue(
            xnode_const.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # matmul
        self.assertTrue(xnode_elemmul.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_elemmul.outputs_tensor) == 1)
        self.assertTrue(xnode_elemmul.outputs_tensor[0].shape == [1, 2048, 1, 1])
        self.assertTrue(
            xnode_elemmul.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_const.layout == Layout.NCHW.name)
        self.assertTrue(xnode_elemmul.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1, 1, 2048])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # const
        self.assertTrue(xnode_const.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_const.outputs_tensor) == 1)
        self.assertTrue(xnode_const.outputs_tensor[0].shape == [1])
        self.assertTrue(xnode_const.outputs_tensor[0].tolist() == [1.0048828125])
        self.assertTrue(
            xnode_const.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # matmul
        self.assertTrue(xnode_elemmul.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_elemmul.outputs_tensor) == 1)
        self.assertTrue(xnode_elemmul.outputs_tensor[0].shape == [1, 1, 1, 2048])
        self.assertTrue(
            xnode_elemmul.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeSqueeze(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 1000, 1, 1]
        # squeeze
        xnode_squeeze = XModelNodeSqueeze("xnode_squeeze")
        xnode_squeeze.axis = [2, 3]

        xnode_input.top = [xnode_squeeze.op_name]
        xnode_squeeze.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_squeeze])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_squeeze.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1, 1, 1000])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # squeeze
        self.assertTrue(xnode_squeeze.layout == Layout.NHWC.name)
        self.assertTrue(xnode_squeeze.axis == [1, 2])
        self.assertTrue(len(xnode_squeeze.outputs_tensor) == 1)
        self.assertTrue(xnode_squeeze.outputs_tensor[0].shape == [1, 1000])
        self.assertTrue(
            xnode_squeeze.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_squeeze.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1000, 1, 1])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # squeeze
        self.assertTrue(xnode_squeeze.layout == Layout.NCHW.name)
        self.assertTrue(xnode_squeeze.axis == [2, 3])
        self.assertTrue(len(xnode_squeeze.outputs_tensor) == 1)
        self.assertTrue(xnode_squeeze.outputs_tensor[0].shape == [1, 1000])
        self.assertTrue(
            xnode_squeeze.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_squeeze.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1, 1, 1000])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # squeeze
        self.assertTrue(xnode_squeeze.layout == Layout.NHWC.name)
        self.assertTrue(xnode_squeeze.axis == [1, 2])
        self.assertTrue(len(xnode_squeeze.outputs_tensor) == 1)
        self.assertTrue(xnode_squeeze.outputs_tensor[0].shape == [1, 1000])
        self.assertTrue(
            xnode_squeeze.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeElemNegative(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 32, 416, 416]
        # elem_negative
        xnode_elem_negative = XModelNodeElemNegative("xnode_negative")

        xnode_input.top = [xnode_elem_negative.op_name]
        xnode_elem_negative.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_elem_negative])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_elem_negative.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elem_negative
        self.assertTrue(xnode_elem_negative.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_elem_negative.outputs_tensor) == 1)
        self.assertTrue(
            xnode_elem_negative.outputs_tensor[0].shape == [1, 416, 416, 32]
        )
        self.assertTrue(
            xnode_elem_negative.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_elem_negative.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 32, 416, 416])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # elem_negative
        self.assertTrue(xnode_elem_negative.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_elem_negative.outputs_tensor) == 1)
        self.assertTrue(
            xnode_elem_negative.outputs_tensor[0].shape == [1, 32, 416, 416]
        )
        self.assertTrue(
            xnode_elem_negative.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_elem_negative.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elem_negative
        self.assertTrue(xnode_elem_negative.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_elem_negative.outputs_tensor) == 1)
        self.assertTrue(
            xnode_elem_negative.outputs_tensor[0].shape == [1, 416, 416, 32]
        )
        self.assertTrue(
            xnode_elem_negative.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeElemSub(self):
        # input_1
        xnode_input_1 = XModelNodeInput("xnode_input_1")
        xnode_input_1.shape = [1, 32, 416, 416]
        # input_2
        xnode_input_2 = XModelNodeInput("xnode_input_2")
        xnode_input_2.shape = [1, 32, 416, 416]
        # elem_sub
        xnode_sub = XModelNodeElemSub("xnode_sub")

        xnode_input_1.top = [xnode_sub.op_name]
        xnode_input_2.top = [xnode_sub.op_name]
        xnode_sub.bottom = [xnode_input_1.op_name, xnode_input_2.op_name]
        self.xmodel.add_xnodes([xnode_input_1, xnode_input_2, xnode_sub])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input_1
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 416, 416, 32])
        # input_2
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 416, 416, 32])
        # elem_sub
        self.assertTrue(xnode_sub.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_sub.outputs_tensor) == 1)
        self.assertTrue(xnode_sub.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_sub.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input_1
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 32, 416, 416])
        # input_2
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 32, 416, 416])
        # elem_sub
        self.assertTrue(xnode_sub.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_sub.outputs_tensor) == 1)
        self.assertTrue(xnode_sub.outputs_tensor[0].shape == [1, 32, 416, 416])
        self.assertTrue(
            xnode_sub.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeElemSquare(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 32, 416, 416]
        # elem_square
        xnode_elem_square = XModelNodeElemSquare("xnode_square")

        xnode_input.top = [xnode_elem_square.op_name]
        xnode_elem_square.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_elem_square])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_elem_square.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elem_square
        self.assertTrue(xnode_elem_square.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_elem_square.outputs_tensor) == 1)
        self.assertTrue(xnode_elem_square.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_elem_square.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_elem_square.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 32, 416, 416])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # elem_square
        self.assertTrue(xnode_elem_square.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_elem_square.outputs_tensor) == 1)
        self.assertTrue(xnode_elem_square.outputs_tensor[0].shape == [1, 32, 416, 416])
        self.assertTrue(
            xnode_elem_square.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_elem_square.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elem_square
        self.assertTrue(xnode_elem_square.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_elem_square.outputs_tensor) == 1)
        self.assertTrue(xnode_elem_square.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_elem_square.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeElemRSqrt(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 32, 416, 416]
        # elem_rsquare
        xnode_elem_square = XModelNodeElemRSqrt("xnode_rsquare")

        xnode_input.top = [xnode_elem_square.op_name]
        xnode_elem_square.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_elem_square])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_elem_square.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elem_rsquare
        self.assertTrue(xnode_elem_square.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_elem_square.outputs_tensor) == 1)
        self.assertTrue(xnode_elem_square.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_elem_square.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_elem_square.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 32, 416, 416])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # elem_rsquare
        self.assertTrue(xnode_elem_square.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_elem_square.outputs_tensor) == 1)
        self.assertTrue(xnode_elem_square.outputs_tensor[0].shape == [1, 32, 416, 416])
        self.assertTrue(
            xnode_elem_square.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_elem_square.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elem_rsquare
        self.assertTrue(xnode_elem_square.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_elem_square.outputs_tensor) == 1)
        self.assertTrue(xnode_elem_square.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_elem_square.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeElemMax(self):
        self.xmodel = XModel("dummy", "tensorflow2", Layout.NHWC.name)
        # input_1
        xnode_input_1 = XModelNodeInput("xnode_input_1")
        xnode_input_1.shape = [1, 1, 1, 2048]
        # input_2
        xnode_input_2 = XModelNodeInput("xnode_input_2")
        xnode_input_2.shape = [1, 1, 1, 2048]
        # elemmax
        xnode_max = XModelNodeElemMax("xnode_max")

        xnode_input_1.top = [xnode_max.op_name]
        xnode_input_2.top = [xnode_max.op_name]
        xnode_max.bottom = [xnode_input_1.op_name, xnode_input_2.op_name]
        self.xmodel.add_xnodes([xnode_input_1, xnode_input_2, xnode_max])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(xnode_max.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input_1
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 2048, 1, 1])
        self.assertTrue(
            xnode_input_1.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # input_2
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 2048, 1, 1])
        self.assertTrue(
            xnode_input_2.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # elemmax
        self.assertTrue(xnode_max.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_max.outputs_tensor) == 1)
        self.assertTrue(xnode_max.outputs_tensor[0].shape == [1, 2048, 1, 1])
        self.assertTrue(
            xnode_max.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(xnode_max.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input_1
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 1, 1, 2048])
        self.assertTrue(
            xnode_input_1.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # input_2
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 1, 1, 2048])
        self.assertTrue(
            xnode_input_2.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elemmax
        self.assertTrue(xnode_max.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_max.outputs_tensor) == 1)
        self.assertTrue(xnode_max.outputs_tensor[0].shape == [1, 1, 1, 2048])
        self.assertTrue(
            xnode_max.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeElemMin(self):
        self.xmodel = XModel("dummy", "tensorflow2", Layout.NHWC.name)
        # input_1
        xnode_input_1 = XModelNodeInput("xnode_input_1")
        xnode_input_1.shape = [1, 1, 1, 2048]
        # input_2
        xnode_input_2 = XModelNodeInput("xnode_input_2")
        xnode_input_2.shape = [1, 1, 1, 2048]
        # elemmin
        xnode_min = XModelNodeElemMin("xnode_min")

        xnode_input_1.top = [xnode_min.op_name]
        xnode_input_2.top = [xnode_min.op_name]
        xnode_min.bottom = [xnode_input_1.op_name, xnode_input_2.op_name]
        self.xmodel.add_xnodes([xnode_input_1, xnode_input_2, xnode_min])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(xnode_min.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input_1
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 2048, 1, 1])
        self.assertTrue(
            xnode_input_1.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # input_2
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 2048, 1, 1])
        self.assertTrue(
            xnode_input_2.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # elemmin
        self.assertTrue(xnode_min.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_min.outputs_tensor) == 1)
        self.assertTrue(xnode_min.outputs_tensor[0].shape == [1, 2048, 1, 1])
        self.assertTrue(
            xnode_min.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_1.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input_2.layout == Layout.NCHW.name)
        self.assertTrue(xnode_min.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input_1
        self.assertTrue(xnode_input_1.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_1.outputs_tensor) == 1)
        self.assertTrue(xnode_input_1.outputs_tensor[0].shape == [1, 1, 1, 2048])
        self.assertTrue(
            xnode_input_1.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # input_2
        self.assertTrue(xnode_input_2.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input_2.outputs_tensor) == 1)
        self.assertTrue(xnode_input_2.outputs_tensor[0].shape == [1, 1, 1, 2048])
        self.assertTrue(
            xnode_input_2.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elemmin
        self.assertTrue(xnode_min.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_min.outputs_tensor) == 1)
        self.assertTrue(xnode_min.outputs_tensor[0].shape == [1, 1, 1, 2048])
        self.assertTrue(
            xnode_min.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeElemRound(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 32, 416, 416]
        # elem_round
        xnode_elem_round = XModelNodeElemRound("xnode_round")

        xnode_input.top = [xnode_elem_round.op_name]
        xnode_elem_round.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_elem_round])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_elem_round.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elem_round
        self.assertTrue(xnode_elem_round.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_elem_round.outputs_tensor) == 1)
        self.assertTrue(xnode_elem_round.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_elem_round.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_elem_round.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 32, 416, 416])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # elem_round
        self.assertTrue(xnode_elem_round.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_elem_round.outputs_tensor) == 1)
        self.assertTrue(xnode_elem_round.outputs_tensor[0].shape == [1, 32, 416, 416])
        self.assertTrue(
            xnode_elem_round.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_elem_round.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elem_round
        self.assertTrue(xnode_elem_round.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_elem_round.outputs_tensor) == 1)
        self.assertTrue(xnode_elem_round.outputs_tensor[0].shape == [1, 416, 416, 32])
        self.assertTrue(
            xnode_elem_round.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeDepthToSpace(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 256, 48, 48]
        # depth_to_space
        xnode_depth2space = XModelNodeDepthToSpace("xnode_depth2space")
        xnode_depth2space.block_size = 2

        xnode_input.top = [xnode_depth2space.op_name]
        xnode_depth2space.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_depth2space])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_depth2space.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 48, 48, 256])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elem_round
        self.assertTrue(xnode_depth2space.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_depth2space.outputs_tensor) == 1)
        self.assertTrue(xnode_depth2space.outputs_tensor[0].shape == [1, 96, 96, 64])
        self.assertTrue(
            xnode_depth2space.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_depth2space.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 256, 48, 48])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # elem_round
        self.assertTrue(xnode_depth2space.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_depth2space.outputs_tensor) == 1)
        self.assertTrue(xnode_depth2space.outputs_tensor[0].shape == [1, 64, 96, 96])
        self.assertTrue(
            xnode_depth2space.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_depth2space.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 48, 48, 256])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # elem_round
        self.assertTrue(xnode_depth2space.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_depth2space.outputs_tensor) == 1)
        self.assertTrue(xnode_depth2space.outputs_tensor[0].shape == [1, 96, 96, 64])
        self.assertTrue(
            xnode_depth2space.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeTypeCast(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 3, 96, 96]
        # type_cast
        xnode_type_cast = XModelNodeTypeCast("xnode_depth2space")
        xnode_type_cast.src_dtype = "uint8"
        xnode_type_cast.dst_dtype = "float32"

        xnode_input.top = [xnode_type_cast.op_name]
        xnode_type_cast.bottom = [xnode_input.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_type_cast])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_type_cast.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 96, 96, 3])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # type_cast
        self.assertTrue(xnode_type_cast.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_type_cast.outputs_tensor) == 1)
        self.assertTrue(xnode_type_cast.outputs_tensor[0].shape == [1, 96, 96, 3])
        self.assertTrue(
            xnode_type_cast.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_type_cast.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 3, 96, 96])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # type_cast
        self.assertTrue(xnode_type_cast.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_type_cast.outputs_tensor) == 1)
        self.assertTrue(xnode_type_cast.outputs_tensor[0].shape == [1, 3, 96, 96])
        self.assertTrue(
            xnode_type_cast.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_type_cast.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 96, 96, 3])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # type_cast
        self.assertTrue(xnode_type_cast.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_type_cast.outputs_tensor) == 1)
        self.assertTrue(xnode_type_cast.outputs_tensor[0].shape == [1, 96, 96, 3])
        self.assertTrue(
            xnode_type_cast.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeRandomStandardNormal(self):
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 128, 1, 1]
        # shape
        xnode_shape = XModelNodeShape("xnode_shape")
        # random
        xnode_random = XModelNodeRandomStandardNormal("xnode_random")
        xnode_random.shape = [1, 128, 1, 1]
        xnode_random.dtype = np.float32
        xnode_random.seed = 87654321
        xnode_random.seed2 = 5832756

        xnode_input.top = [xnode_shape.op_name]
        xnode_shape.bottom = [xnode_input.op_name]
        xnode_shape.top = [xnode_random.op_name]
        xnode_random.bottom = [xnode_shape.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_shape, xnode_random])

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_shape.layout == Layout.NCHW.name)
        self.assertTrue(xnode_random.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1, 1, 128])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # shape
        self.assertTrue(xnode_shape.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_shape.outputs_tensor) == 1)
        self.assertTrue(xnode_shape.outputs_tensor[0].shape == [4])
        self.assertTrue(
            xnode_shape.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # random
        self.assertTrue(xnode_random.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_random.outputs_tensor) == 1)
        self.assertTrue(xnode_random.outputs_tensor[0].shape == [1, 1, 1, 128])
        self.assertTrue(
            xnode_random.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(xnode_shape.layout == Layout.NHWC.name)
        self.assertTrue(xnode_random.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 128, 1, 1])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # shape
        self.assertTrue(xnode_shape.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_shape.outputs_tensor) == 1)
        self.assertTrue(xnode_shape.outputs_tensor[0].shape == [4])
        self.assertTrue(
            xnode_shape.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )
        # random
        self.assertTrue(xnode_random.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_random.outputs_tensor) == 1)
        self.assertTrue(xnode_random.outputs_tensor[0].shape == [1, 128, 1, 1])
        self.assertTrue(
            xnode_random.outputs_tensor[0].data_format.name == Layout.NCHW.name
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(xnode_shape.layout == Layout.NCHW.name)
        self.assertTrue(xnode_random.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 1, 1, 128])
        self.assertTrue(
            xnode_input.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # shape
        self.assertTrue(xnode_shape.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_shape.outputs_tensor) == 1)
        self.assertTrue(xnode_shape.outputs_tensor[0].shape == [4])
        self.assertTrue(
            xnode_shape.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )
        # random
        self.assertTrue(xnode_random.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_random.outputs_tensor) == 1)
        self.assertTrue(xnode_random.outputs_tensor[0].shape == [1, 1, 1, 128])
        self.assertTrue(
            xnode_random.outputs_tensor[0].data_format.name == Layout.NHWC.name
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeReduceMax(self):
        self.xmodel = XModel("dummy", "tensorflow", Layout.NHWC.name)
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 224, 224, 12]
        # reduce_max
        xnode_reduce_max = XModelNodeReduceMax("xnode_reduce_max")
        xnode_reduce_max.axis = [-1]
        xnode_reduce_max.keep_dims = True

        xnode_reduce_max.bottom = [xnode_input.op_name]
        xnode_input.top = [xnode_reduce_max.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_reduce_max])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 12, 224, 224])
        # reduce_max
        self.assertTrue(xnode_reduce_max.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_reduce_max.outputs_tensor) == 1)
        self.assertTrue(
            xnode_reduce_max.outputs_tensor[0].shape == [1, 1, 224, 224,]
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 224, 224, 12])
        # reduce_max
        self.assertTrue(xnode_reduce_max.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_reduce_max.outputs_tensor) == 1)
        self.assertTrue(xnode_reduce_max.outputs_tensor[0].shape == [1, 224, 224, 1])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 12, 224, 224])
        # reduce_max
        self.assertTrue(xnode_reduce_max.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_reduce_max.outputs_tensor) == 1)
        self.assertTrue(
            xnode_reduce_max.outputs_tensor[0].shape == [1, 1, 224, 224,]
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeReduceSum(self):
        self.xmodel = XModel("dummy", "tensorflow", Layout.NHWC.name)
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 224, 224, 12]
        # reduce_sum
        xnode_reduce_sum = XModelNodeReduceSum("xnode_reduce_sum")
        xnode_reduce_sum.axis = [-1]
        xnode_reduce_sum.keep_dims = True

        xnode_reduce_sum.bottom = [xnode_input.op_name]
        xnode_input.top = [xnode_reduce_sum.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_reduce_sum])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 12, 224, 224])
        # reduce_sum
        self.assertTrue(xnode_reduce_sum.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_reduce_sum.outputs_tensor) == 1)
        self.assertTrue(
            xnode_reduce_sum.outputs_tensor[0].shape == [1, 1, 224, 224,]
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 224, 224, 12])
        # reduce_sum
        self.assertTrue(xnode_reduce_sum.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_reduce_sum.outputs_tensor) == 1)
        self.assertTrue(xnode_reduce_sum.outputs_tensor[0].shape == [1, 224, 224, 1])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 12, 224, 224])
        # reduce_sum
        self.assertTrue(xnode_reduce_sum.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_reduce_sum.outputs_tensor) == 1)
        self.assertTrue(
            xnode_reduce_sum.outputs_tensor[0].shape == [1, 1, 224, 224,]
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_XModelNodeReduceProd(self):
        self.xmodel = XModel("dummy", "tensorflow", Layout.NHWC.name)
        # input
        xnode_input = XModelNodeInput("xnode_input")
        xnode_input.shape = [1, 224, 224, 12]
        # reduce_prod
        xnode_reduce_prod = XModelNodeReduceProd("xnode_reduce_prod")
        xnode_reduce_prod.axis = [-1]
        xnode_reduce_prod.keep_dims = True

        xnode_reduce_prod.bottom = [xnode_input.op_name]
        xnode_input.top = [xnode_reduce_prod.op_name]
        self.xmodel.add_xnodes([xnode_input, xnode_reduce_prod])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 12, 224, 224])
        # reduce_prod
        self.assertTrue(xnode_reduce_prod.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_reduce_prod.outputs_tensor) == 1)
        self.assertTrue(
            xnode_reduce_prod.outputs_tensor[0].shape == [1, 1, 224, 224,]
        )

        # nchw -> nhwc
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        ok, error = self.xmodel.infer_shape(Layout.NHWC)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 224, 224, 12])
        # reduce_prod
        self.assertTrue(xnode_reduce_prod.layout == Layout.NHWC.name)
        self.assertTrue(len(xnode_reduce_prod.outputs_tensor) == 1)
        self.assertTrue(xnode_reduce_prod.outputs_tensor[0].shape == [1, 224, 224, 1])

        # nhwc -> nchw
        self.assertTrue(self.xmodel.layout == Layout.NHWC.name)
        ok, error = self.xmodel.infer_shape(Layout.NCHW)
        self.assertTrue(ok and error is None)
        self.assertTrue(self.xmodel.layout == Layout.NCHW.name)
        # input
        self.assertTrue(xnode_input.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_input.outputs_tensor) == 1)
        self.assertTrue(xnode_input.outputs_tensor[0].shape == [1, 12, 224, 224])
        # reduce_prod
        self.assertTrue(xnode_reduce_prod.layout == Layout.NCHW.name)
        self.assertTrue(len(xnode_reduce_prod.outputs_tensor) == 1)
        self.assertTrue(
            xnode_reduce_prod.outputs_tensor[0].shape == [1, 1, 224, 224,]
        )


if __name__ == "__main__":
    unittest.main()
