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
"""Tool to generate open source api_docs for tensorflow_model_optimization.

To use:

  1. Install the tensorflow docs package, which is only compatible with Python

    python3 -m pip install git+https://github.com/tensorflow/docs

  2. Install TensorFlow Model Optimization. The API docs are generated from
  `tfmot` from the import of the tfmot package below, based on what is exposed
  under
   https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/api.

    See https://www.tensorflow.org/model_optimization/guide/install.

  3. Run build_docs.py.

    python3 build_docs.py --output_dir=/tmp/model_optimization_api

  4. View the generated markdown files on a viewer. One option is to fork
     https://github.com/tensorflow/model-optimization/, push a change that
     copies the files to tensorflow_model_optimization/g3doc, and then
     view the files on Github.

Note:
  If duplicate or spurious docs are generated (e.g. internal names), consider
  blacklisting them via the `private_map` argument below.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from tensorflow_docs.api_generator import generate_lib

import tensorflow_model_optimization as tfmot

flags.DEFINE_string("output_dir", "/tmp/model_optimization_api",
                    "Where to output the docs")

flags.DEFINE_string(
    "code_url_prefix",
    ("https://github.com/tensorflow/model-optimization/blob/master/"
     "tensorflow_model_optimization"),
    "The url prefix for links to code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "model_optimization/api_docs/python",
                    "Path prefix in the _toc.yaml")

FLAGS = flags.FLAGS


def main(unused_argv):
  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow Model Optimization",
      py_modules=[("tfmot", tfmot)],
      base_dir=os.path.dirname(tfmot.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      # TODO(tfmot): remove this once the next release after 0.3.0 happens.
      # This is needed in the interim because the API docs reflect
      # the latest release and the current release still wildcard imports
      # all of the classes below.
      private_map={
          "tfmot.sparsity.keras": [
              # List of internal classes which get exposed when imported.
              "InputLayer",
              "custom_object_scope",
              "pruning_sched",
              "pruning_wrapper",
              "absolute_import",
              "division",
              "print_function",
              "compat"
          ]
      },
  )

  doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)
