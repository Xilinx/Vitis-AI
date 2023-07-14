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
import sys
import subprocess

from xnnc.version import __version__, __git_version__

def version_string():
  version = __version__
  version += " git version " + __git_version__
  return version


def parse_args(args=None):
    if not args:
        args = sys.argv[1:]

    desc = """
    Convert TensorFlow, Caffe or PyTorch models into XIR by XNNC. To generate XIR graph from TensorFlow resnet50 model, for instance, run the command:

        xnnc-run --type tensorflow --layout NHWC --model /path/to/resnet50.pb --out /path/to/resnet50.xmodel

    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--model",
        dest="model_files",
        action="append",
        required=True,
        help="file name to TensorFlow 1.12+ frozen model file (*.pb),  TensorFlow 2.0+ frozen model file (*.h5), or Caffe caffemodel (*.caffemodel).",
    )
    parser.add_argument(
        "--proto", dest="proto", help="file name to Caffe prototxt files."
    )
    parser.add_argument(
        "--type",
        dest="model_type",
        choices=["tensorflow", "tensorflow2", "caffe"],
        required=True,
        help="type of raw model.",
    )
    parser.add_argument(
        "--layout",
        dest="layout",
        choices=["NHWC", "NCHW"],
        required=True,
        help="data format used in raw model: 'NHWC' or 'NCHW', default='NHWC'.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version_string()}"
    )
    parser.add_argument(
        "--out",
        dest="out_filename",
        help="file name of the generated xmodel.",
        required=True,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--inputs-shape",
        dest="inputs_shape",
        help="shape of one or more input feature maps. For example, to specify two input shapes, use --inputs-shape 1,416,416,3 --inputs-shape 1,256,256,3",
        action="append",
    )
    group.add_argument(
        "--batchsize",
        dest="batchsize",
        action=BatchsizeAction,
        type=int,
        default=1,
        help="target batchsize >= 1",
    )
    group.add_argument(
        "--named-inputs-shape",
        dest="named_inputs_shape",
        nargs="*",
        help="pairs of name and shape of two or more input layers. For example, to specify two input shapes, use --inputs input_layer1_name:1,416,416,3 input_layer2_name:1,256,256,3",
        action=ParseKwargs,
    )

    args = parser.parse_args(args)
    return args


class BatchsizeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values < 1:
            parser.error(f"Minimum batchsize for {option_string} is 1")

        setattr(namespace, self.dest, values)


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split(":")
            getattr(namespace, self.dest)[key.strip()] = value.strip()
