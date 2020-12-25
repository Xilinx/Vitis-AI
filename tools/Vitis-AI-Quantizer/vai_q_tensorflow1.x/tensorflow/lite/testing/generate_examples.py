# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Generate a series of TensorFlow graphs that become tflite test cases.

Usage:

generate_examples <output directory>

bazel run //tensorflow/lite/testing:generate_examples

To more easily debug failures use (or override) the --save_graphdefs flag to
place text proto graphdefs into the generated zip files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import argparse
import os
import sys
from tensorflow.lite.testing import generate_examples_lib
from tensorflow.lite.testing import toco_convert

# TODO(aselle): Disable GPU for now
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


parser = argparse.ArgumentParser(description="Script to generate TFLite tests.")
parser.add_argument("output_path",
                    help="Directory where the outputs will be go.")
parser.add_argument(
    "--zip_to_output",
    type=str,
    help="Particular zip to output.",
    required=True)
parser.add_argument("--toco",
                    type=str,
                    help="Path to toco tool.",
                    required=True)
parser.add_argument(
    "--known_bugs_are_errors",
    action="store_true",
    help=("If a particular model is affected by a known bug,"
          " count it as a converter error."))
parser.add_argument(
    "--ignore_converter_errors",
    action="store_true",
    help="Raise an exception if any converter error is encountered.")
parser.add_argument(
    "--save_graphdefs",
    action="store_true",
    help="Include intermediate graphdefs in the output zip files.")
parser.add_argument(
    "--run_with_flex",
    action="store_true",
    help="Whether the TFLite Flex converter is being used.")
parser.add_argument(
    "--make_edgetpu_tests",
    action="store_true",
    help="Whether to generate test cases for edgetpu.")
parser.add_argument(
    "--make_forward_compat_test",
    action="store_true",
    help="Make tests by setting TF forward compatibility horizon to the future")


# Toco binary path provided by the generate rule.
bin_path = None


def main(unused_args):
  options = generate_examples_lib.Options()

  options.output_path = FLAGS.output_path
  options.zip_to_output = FLAGS.zip_to_output
  options.toco = FLAGS.toco
  options.known_bugs_are_errors = FLAGS.known_bugs_are_errors
  options.ignore_converter_errors = FLAGS.ignore_converter_errors
  options.save_graphdefs = FLAGS.save_graphdefs
  options.run_with_flex = FLAGS.run_with_flex
  options.make_edgetpu_tests = FLAGS.make_edgetpu_tests
  options.make_forward_compat_test = FLAGS.make_forward_compat_test
  options.tflite_convert_function = toco_convert.toco_convert

  generate_examples_lib.generate_examples(options)


if __name__ == "__main__":
  FLAGS, unparsed = parser.parse_known_args()

  if unparsed:
    print("Usage: %s <path out> <zip file to generate>")
    exit(1)
  else:
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
