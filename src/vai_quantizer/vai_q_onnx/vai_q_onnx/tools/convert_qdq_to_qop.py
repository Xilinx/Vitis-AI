#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Convert QDQ to QOperator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import onnx
import os
import argparse

from vai_q_onnx.optimizations import convert_transforms_pipeline
from vai_q_onnx.utils import model_utils


def convert_qdq_to_qop(model):
    """Convert QDQ to QOperator."""
    convert_pipeline = convert_transforms_pipeline.ConvertQDQToQOPTransformsPipeline(
    )
    converted_model, _ = convert_pipeline.apply(model,
                                                candidate_nodes=None,
                                                node_metadata=None)
    return converted_model


def run_main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",
                        type=str,
                        default="",
                        help="input onnx model file path.")
    parser.add_argument("--output_model",
                        type=str,
                        default="",
                        help="output onnx model file path.")
    FLAGS, uparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.input_model):
        print("Input model file '{}' does not exist!".format(FLAGS.input_model))
        print(
            "Usage: python -m vai_q_onnx.tools.convert_qdq_to_qop --input_model INPUT_MODEL_PATH --output_model OUTPUT_MODEL_PATH."
        )
        exit()

    model = onnx.load_model(FLAGS.input_model)
    model = model_utils.copy_shared_nodes(model)
    converted_model = convert_qdq_to_qop(model)
    onnx.save(converted_model, FLAGS.output_model)
    print('Conversion Finished!')
    print('Converted model saved in: {}'.format(FLAGS.output_model))


if __name__ == '__main__':
    run_main()
