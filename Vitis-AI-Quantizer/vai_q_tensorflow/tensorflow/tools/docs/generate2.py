# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""A tool to generate api_docs for TensorFlow2.

```
python generate2.py --output_dir=/tmp/out
```

Requires a local installation of:
  https://github.com/tensorflow/docs/tree/master/tools
  tf-nightly-2.0-preview
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
import textwrap

from absl import app
from absl import flags
from distutils.version import LooseVersion

import tensorflow as tf

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import doc_generator_visitor
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import parser

import tensorboard
import tensorflow_estimator
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect

# Use tensorflow's `tf_inspect`, which is aware of `tf_decorator`.
parser.tf_inspect = tf_inspect

# `tf` has an `__all__` that doesn't list important things like `keras`.
# The doc generator recognizes `__all__` as the list of public symbols.
# So patch `tf.__all__` to list everything.
tf.__all__ = [item_name for item_name, value in tf_inspect.getmembers(tf)]


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "code_url_prefix",
    "/code/stable/tensorflow",
    "A url to prepend to code paths when creating links to defining code")

flags.DEFINE_string(
    "output_dir", "/tmp/out",
    "A directory, where the docs will be output to.")

flags.DEFINE_bool("search_hints", True,
                  "Include meta-data search hints at the top of each file.")

flags.DEFINE_string("site_path", "",
                    "The prefix ({site-path}/api_docs/python/...) used in the "
                    "`_toc.yaml` and `_redirects.yaml` files")


if tf.__version__.startswith('1'):
  PRIVATE_MAP = {
      'tf.contrib.autograph': ['utils', 'operators'],
      'tf.test': ['mock'],
      'tf.contrib.estimator': ['python'],
  }

  DO_NOT_DESCEND_MAP = {
      'tf': ['cli', 'lib', 'wrappers'],
      'tf.contrib': [
          'compiler',
          'grid_rnn',
          # Block contrib.keras to de-clutter the docs
          'keras',
          'labeled_tensor',
          'quantization',
          'session_bundle',
          'slim',
          'solvers',
          'specs',
          'tensor_forest',
          'tensorboard',
          'testing',
          'tfprof',
      ],
      'tf.contrib.bayesflow': [
          'special_math', 'stochastic_gradient_estimators',
          'stochastic_variables'
      ],
      'tf.contrib.ffmpeg': ['ffmpeg_ops'],
      'tf.contrib.graph_editor': [
          'edit', 'match', 'reroute', 'subgraph', 'transform', 'select', 'util'
      ],
      'tf.contrib.keras': ['api', 'python'],
      'tf.contrib.layers': ['feature_column', 'summaries'],
      'tf.contrib.learn': [
          'datasets',
          'head',
          'graph_actions',
          'io',
          'models',
          'monitors',
          'ops',
          'preprocessing',
          'utils',
      ],
      'tf.contrib.util': ['loader'],
  }
else:
  PRIVATE_MAP = {}
  DO_NOT_DESCEND_MAP = {}
  tf.__doc__ = """
    ## TensorFlow 2.0 Beta

    Caution:  This is a developer preview.  You will likely find some bugs,
    performance issues, and more, and we encourage you to tell us about them.
    We value your feedback!

    These docs were generated from the beta build of TensorFlow 2.0.

    You can install the exact version that was used to generate these docs
    with:

    ```
    pip install tensorflow==2.0.0-beta1
    ```
    """

_raw_ops_doc = textwrap.dedent("""\n
  Note: `tf.raw_ops` provides direct/low level access to all TensorFlow ops. See \
  [the RFC](https://github.com/tensorflow/community/blob/master/rfcs/20181225-tf-raw-ops.md)
  for details. Unless you are library writer, you likely do not need to use these
  ops directly.""")

if LooseVersion(tf.__version__) < LooseVersion('2'):
  tf.raw_ops.__doc__ = _raw_ops_doc
  tf.contrib.__doc__ = """
    Contrib module containing volatile or experimental code.

    Warning: The `tf.contrib` module will not be included in TensorFlow 2.0. Many
    of its submodules have been integrated into TensorFlow core, or spun-off into
    other projects like [`tensorflow_io`](https://github.com/tensorflow/io), or
    [`tensorflow_addons`](https://github.com/tensorflow/addons). For instructions
    on how to upgrade see the
    [Migration guide](https://www.tensorflow.org/beta/guide/migration_guide).
    """
else:
  tf.raw_ops.__doc__ += _raw_ops_doc


# The doc generator isn't aware of tf_export.
# So prefix the score tuples with -1 when this is the canonical name, +1
# otherwise. The generator chooses the name with the lowest score.
class TfExportAwareDocGeneratorVisitor(
    doc_generator_visitor.DocGeneratorVisitor):
  """A `tf_export` aware doc_visitor."""

  def _score_name(self, name):
    canonical = tf_export.get_canonical_name_for_symbol(self._index[name])

    canonical_score = 1
    if canonical is not None and name == "tf." + canonical:
      canonical_score = -1

    scores = super(TfExportAwareDocGeneratorVisitor, self)._score_name(name)
    return (canonical_score,) + scores


def _hide_layer_and_module_methods():
  """Hide methods and properties defined in the base classes of keras layers."""
  # __dict__ only sees attributes defined in *this* class, not on parent classes
  module_contents = list(tf.Module.__dict__.items())
  layer_contents = list(tf.keras.layers.Layer.__dict__.items())

  for name, obj in module_contents + layer_contents:
    if name == "__init__":
      continue

    if isinstance(obj, property):
      obj = obj.fget

    if isinstance(obj, (staticmethod, classmethod)):
      obj = obj.__func__

    try:
      doc_controls.do_not_doc_in_subclasses(obj)
    except AttributeError:
      pass


def build_docs(output_dir, code_url_prefix, search_hints=True):
  """Build api docs for tensorflow v2.

  Args:
    output_dir: A string path, where to put the files.
    code_url_prefix: prefix for "Defined in" links.
    search_hints: Bool. Include meta-data search hints at the top of each file.
  """
  _hide_layer_and_module_methods()

  try:
    doc_controls.do_not_generate_docs(tf.tools)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.compat.v1.pywrap_tensorflow)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.pywrap_tensorflow)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.flags)
  except AttributeError:
    pass

  base_dir = path.dirname(tf.__file__)

  base_dirs = (
      base_dir,
      # External packages base directories,
      path.dirname(tensorboard.__file__),
      path.dirname(tensorflow_estimator.__file__),
  )

  code_url_prefixes = (
      code_url_prefix,
      # External packages source repositories,
      "https://github.com/tensorflow/tensorboard/tree/master/tensorboard",
      "https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator",
  )

  if LooseVersion(tf.__version__) < LooseVersion('2'):
    root_title = 'TensorFlow'
  elif LooseVersion(tf.__version__) >= LooseVersion('2'):
    root_title = 'TensorFlow 2.0'

  doc_generator = generate_lib.DocGenerator(
      root_title=root_title,
      py_modules=[("tf", tf)],
      base_dir=base_dirs,
      search_hints=search_hints,
      code_url_prefix=code_url_prefixes,
      site_path=FLAGS.site_path,
      visitor_cls=TfExportAwareDocGeneratorVisitor,
      private_map=PRIVATE_MAP,
      do_not_descend_map=DO_NOT_DESCEND_MAP)

  doc_generator.build(output_dir)


def main(argv):
  del argv
  build_docs(output_dir=FLAGS.output_dir,
             code_url_prefix=FLAGS.code_url_prefix,
             search_hints=FLAGS.search_hints)


if __name__ == "__main__":
  app.run(main)
