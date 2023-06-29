#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Graph transforms for the conversion of onnx models."""

import os
import argparse
import onnx

from google.protobuf import text_format


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
                        help="output onnxtxt model file path.")
    FLAGS, uparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.input_model):
        print("Input model file '{}' does not exist!".format(FLAGS.input_model))
        print(
            "Usage: python convert_onnx_to_onnxtxt --input_model INPUT_MODEL_PATH --output_model OUTPUT_MODEL_PATH."
        )
        exit()

    model = onnx.load_model(FLAGS.input_model)
    with open(FLAGS.output_model, 'w') as f:
        f.write(text_format.MessageToString(model))
    print('Conversion Finished!')
    print('Converted model saved in: {}'.format(FLAGS.output_model))


if __name__ == '__main__':
    run_main()
