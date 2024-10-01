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
from typing import List

from xnnc.ir.enums import TargetType
from xnnc.xconverter import XConverter, __version__


def normal_run(args):
    assert args is not None, "'args' should not be None."
    # model type
    assert args.model_type.lower() in [
        "tensorflow",
        "tensorflow2",
        "caffe",
        "pytorch",
        "nndct",
    ], "'model_type' should be 'tensorflow', 'tensorflow2', 'caffe' or 'nndct."
    model_t = args.model_type.lower()

    # layout
    assert args.layout.lower() in [
        "nhwc",
        "nchw",
    ], "'layout' should be 'NHWC' or 'NCHW'."
    layout = args.layout.lower()
    if args.model_type.lower() == "caffe" and layout != "nchw":
        print("[ERROR] Only support Caffe model of the 'NCHW' layout.")
        sys.exit(1)

    # model files
    assert len(args.model_files) > 0, "'model_files' should have one or more paths."
    model_files: List[Path] = []
    for file_path in args.model_files:
        model_files.append(Path(file_path).resolve())
    if model_t == "caffe":
        model_files.append(Path(args.proto).resolve())
    # check model files
    for fname in model_files:
        if not fname.exists():
            print(f"[ERROR] Not found the file or directory: {str(fname.absolute())}")
            sys.exit(1)
    if args.model_type.lower() == "tensorflow" and model_files[0].suffix != ".pb":
        print(
            "[ERROR] Only support TensorFlow 1.12+ frozen model file with the '.pb' extension."
        )
        sys.exit(1)
    if args.model_type.lower() == "tensorflow2" and model_files[0].suffix != ".h5":
        print(
            "[ERROR] Only support TensorFlow 2.0+ frozen model file with the '.h5' extension."
        )
        sys.exit(1)
    if args.model_type.lower() == "nndct" and model_files[0].suffix != ".pb":
        print("[ERROR] Only support nndct model file with the '.pb' extension.")
        sys.exit(1)

    # source dir of raw model
    src_dir = model_files[0].parent

    # validate model files
    if not validate(model_t, model_files):
        print(f"[WARNING] Found {len(model_files)} {model_t} model files in {src_dir}")
        sys.exit(1)

    # shape info
    if model_t == "pytorch":
        assert (
            args.inputs_shape is not None
        ), "[ERROR] Shape info of inputs should be given for PyTorch model [Use --inputs-shape option]. For example, --inputs-shape 1,3,256,512"

    in_shapes = None
    if args.inputs_shape:
        assert (
            len(args.inputs_shape) > 0
        ), "Please specify one or more input shapes by '--inputs-shape' option."

        in_shapes = []
        for shape in args.inputs_shape:
            in_shapes.append([int(x.strip()) for x in shape.split(",")])
        print(f"in_shapes: {in_shapes}")

    elif args.named_inputs_shape:
        assert (
            len(args.named_inputs_shape) > 0
        ), "Please specify two or more input shapes by '--named-inputs-shape' option."

        in_shapes = dict()
        for name, shape in args.named_inputs_shape.items():
            in_shapes[name] = [int(x.strip()) for x in shape.split(",")]
        print(f"in_shapes: {in_shapes}")

    # set target type
    fname_frozen = Path(args.out_filename).resolve()
    if fname_frozen.suffix.lower() == ".pb":
        target = TargetType.OPENIR
    else:
        target = TargetType.XIR

    # * convert raw models into XIR graph
    XConverter.run(
         model_files,
         model_t,
         layout,
         in_shapes=in_shapes if in_shapes else None,
         batchsize=args.batchsize,
         dump_to=Path(args.out_filename),
         target=target,
    )


def validate(model_t: bool, model_files):
    t = model_t.lower()
    if t == "caffe":
        return len(model_files) == 2
    elif t in ["tensorflow", "tensorflow2"]:
        return len(model_files) == 1
    elif t in ["pytorch", "nndct"]:
        return len(model_files) == 1
    else:
        return False
