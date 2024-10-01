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


import argparse
from pathlib import Path
import sys

curr_path = Path(__file__).resolve()
PRJ_DIR = curr_path.parents[1]
sys.path.append(str(PRJ_DIR.resolve()))

from xnnc import cli, runner

# import logging
import unittest


class CLITestCase(unittest.TestCase):
    # True: stop all test cases; otherwise, start all.
    stop_all = True

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_batchsize_option(self):
        model_type = "tensorflow"
        layout = "NHWC"
        model_file = "demo.pb"
        out_fn = Path("demo.xmodel").resolve()
        batchsize = "10"

        parser = cli.parse_args(
            [
                "--type",
                model_type,
                "--layout",
                layout,
                "--model",
                model_file,
                "--batchsize",
                batchsize,
                "--out",
                str(out_fn),
            ]
        )

        self.assertEqual(layout, parser.layout)
        self.assertEqual([model_file], parser.model_files)
        self.assertEqual(model_type, parser.model_type)
        self.assertEqual(str(out_fn), parser.out_filename)
        self.assertEqual(int(batchsize), parser.batchsize)

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_inputs_shape_option(self):
        model_type = "tensorflow"
        layout = "NHWC"
        model_file = "demo.pb"
        out_fn = Path("demo.xmodel").resolve()
        input_shape = "1,224,224,3"

        parser = cli.parse_args(
            [
                "--type",
                model_type,
                "--layout",
                layout,
                "--model",
                model_file,
                "--inputs-shape",
                input_shape,
                "--out",
                str(out_fn),
            ]
        )

        self.assertEqual(layout, parser.layout)
        self.assertEqual([model_file], parser.model_files)
        self.assertEqual(model_type, parser.model_type)
        self.assertEqual(str(out_fn), parser.out_filename)
        self.assertEqual([input_shape], parser.inputs_shape)

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_batchsize_or_inputs_shape_options(self):
        model_type = "tensorflow"
        layout = "NHWC"
        model_file = "demo.pb"
        out_fn = Path("demo.xmodel").resolve()
        batchsize = "10"
        input_shape = "1,224,224,3"

        with self.assertRaises(SystemExit):
            cli.parse_args(
                [
                    "--type",
                    model_type,
                    "--layout",
                    layout,
                    "--model",
                    model_file,
                    "--batchsize",
                    batchsize,
                    "--inputs-shape",
                    input_shape,
                    "--out",
                    str(out_fn),
                ]
            )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_named_inputs_shape_option(self):
        model_type = "tensorflow"
        layout = "NHWC"
        model_file = "demo.pb"
        out_fn = Path("demo.xmodel").resolve()
        input_layer_1 = "input_layer_1 : 1,224,224,3"
        input_layer_2 = "input_layer_2 : 7,7"

        args = cli.parse_args(
            [
                "--type",
                model_type,
                "--layout",
                layout,
                "--model",
                model_file,
                "--out",
                str(out_fn),
                "--named-inputs-shape",
                input_layer_1,
                input_layer_2,
            ]
        )

        self.assertEqual(layout, args.layout)
        self.assertEqual([model_file], args.model_files)
        self.assertEqual(model_type, args.model_type)
        self.assertEqual(str(out_fn), args.out_filename)
        self.assertEqual(
            {"input_layer_1": "1,224,224,3", "input_layer_2": "7,7"},
            args.named_inputs_shape,
        )

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_normal_run_inputs_shape_option(self):
        model_type = "tensorflow"
        layout = "NHWC"
        model_file = "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/ssd_resnet50_v1_fpn_coco/quantize_eval_model.pb"
        out_fn = Path("demo.xmodel").resolve()
        input_shape = "1,640,640,3"

        args = cli.parse_args(
            [
                "--type",
                model_type,
                "--layout",
                layout,
                "--model",
                model_file,
                "--inputs-shape",
                input_shape,
                "--out",
                str(out_fn),
            ]
        )

        runner.normal_run(args)
        self.assertTrue(out_fn.exists())
        out_fn.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_normal_run_named_inputs_shape_option_tf(self):
        model_type = "tensorflow"
        layout = "NHWC"
        model_file = "/root/workspace/samshin/shaped_xmodel_zoo/tensorflow_xmodel/ssd_resnet50_v1_fpn_coco/quantize_eval_model.pb"
        out_fn = Path("demo.xmodel").resolve()
        named_input_shape = "image_tensor: 1,640,640,3"

        args = cli.parse_args(
            [
                "--type",
                model_type,
                "--layout",
                layout,
                "--model",
                model_file,
                "--out",
                str(out_fn),
                "--named-inputs-shape",
                named_input_shape,
            ]
        )

        runner.normal_run(args)
        self.assertTrue(out_fn.exists())
        out_fn.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_normal_run_named_inputs_shape_option_tf2(self):
        model_type = "tensorflow2"
        layout = "NHWC"
        model_file = "/root/workspace/samshin/shaped_xmodel_zoo/tf2_xmodel/Mobilenet_v3_small_1.0/quantized.h5"
        out_fn = Path("demo.xmodel").resolve()
        named_input_shape = "input_1: 1,224,224,3"

        args = cli.parse_args(
            [
                "--type",
                model_type,
                "--layout",
                layout,
                "--model",
                model_file,
                "--out",
                str(out_fn),
                "--named-inputs-shape",
                named_input_shape,
            ]
        )

        runner.normal_run(args)
        self.assertTrue(out_fn.exists())
        out_fn.unlink()

    @unittest.skipIf(stop_all, "skip over this routine")
    def test_normal_run_named_inputs_shape_option_caffe(self):
        model_type = "caffe"
        layout = "NCHW"
        model_file = "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/resnet50_baseline9213_ck/deploy.caffemodel"
        proto_file = "/root/workspace/samshin/shaped_xmodel_zoo/caffe_xmodel/resnet50_baseline9213_ck/deploy.prototxt"
        out_fn = Path("demo.xmodel").resolve()
        named_input_shape = "data: 1,3,224,224"

        args = cli.parse_args(
            [
                "--type",
                model_type,
                "--layout",
                layout,
                "--model",
                model_file,
                "--proto",
                proto_file,
                "--out",
                str(out_fn),
                "--named-inputs-shape",
                named_input_shape,
            ]
        )

        runner.normal_run(args)
        self.assertTrue(out_fn.exists())
        out_fn.unlink()


if __name__ == "__main__":
    unittest.main()