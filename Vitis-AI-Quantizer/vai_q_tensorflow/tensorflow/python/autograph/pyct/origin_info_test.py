# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for origin_info module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import textwrap

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct.testing import basic_definitions
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


class OriginInfoTest(test.TestCase):

  def test_create_source_map(self):

    source = """
      def test_fn(x):
        return x + 1
    """
    source = textwrap.dedent(source)

    node = parser.parse_str(source)
    fake_origin = origin_info.OriginInfo(
        loc=origin_info.Location('fake_filename', 3, 7),
        function_name='fake_function_name',
        source_code_line='fake source line',
        comment=None)
    anno.setanno(node, anno.Basic.ORIGIN, fake_origin)

    source_map = origin_info.create_source_map(node, source, 'test_filename')

    loc = origin_info.LineLocation('test_filename', 2)
    self.assertIn(loc, source_map)
    self.assertIs(source_map[loc], fake_origin)

  def _create_source_map(self, test_fn):
    node, source = parser.parse_entity(test_fn, ())
    origin_info.resolve_entity(node, source, test_fn)
    # Creating a source map with the source code as output will create
    # an identity map.
    return origin_info.create_source_map(node, source, 'test_filename')

  def test_create_source_map_identity(self):
    test_fn = basic_definitions.simple_function
    source_map = self._create_source_map(test_fn)
    module_path = tf_inspect.getsourcefile(test_fn)

    # Origin line numbers below should match those in basic_definitions.py

    definition_loc = origin_info.LineLocation('test_filename', 1)
    self.assertIn(definition_loc, source_map)
    self.assertEqual(source_map[definition_loc].loc.lineno, 23)
    self.assertEqual(source_map[definition_loc].loc.filename, module_path)
    self.assertEqual(source_map[definition_loc].function_name,
                     'simple_function')

  def test_create_source_map_multiline_call(self):
    test_fn = basic_definitions.function_with_multiline_call
    source_map = self._create_source_map(test_fn)
    module_path = tf_inspect.getsourcefile(test_fn)

    # Origin line numbers below should match those in basic_definitions.py

    call_loc = origin_info.LineLocation('test_filename', 3)
    self.assertIn(call_loc, source_map)
    self.assertEqual(source_map[call_loc].loc.lineno, 55)
    self.assertEqual(source_map[call_loc].loc.filename, module_path)
    self.assertEqual(source_map[call_loc].function_name,
                     'function_with_multiline_call')
    self.assertEqual(source_map[call_loc].source_code_line, '  return range(')

    second_arg_loc = origin_info.LineLocation('test_filename', 5)
    self.assertIn(second_arg_loc, source_map)
    self.assertEqual(source_map[second_arg_loc].loc.lineno, 57)
    self.assertEqual(source_map[second_arg_loc].loc.filename, module_path)
    self.assertEqual(source_map[second_arg_loc].function_name,
                     'function_with_multiline_call')
    self.assertEqual(source_map[second_arg_loc].source_code_line,
                     '      x + 1,')

  def test_create_source_map_no_origin_info(self):

    test_fn = basic_definitions.simple_function
    node, _ = parser.parse_entity(test_fn,
                                  inspect_utils.getfutureimports(test_fn))
    # No origin information should result in an empty map.
    test_fn_lines, _ = tf_inspect.getsourcelines(test_fn)
    source_map = origin_info.create_source_map(node, '\n'.join(test_fn_lines),
                                               test_fn)

    self.assertEmpty(source_map)

  def test_resolve(self):

    source = """
      def test_fn(x):
        '''Docstring.'''
        return x  # comment
    """
    source = textwrap.dedent(source)
    node = parser.parse_str(source)
    origin_info.resolve(node, source, 'test_file', 10, 10)

    def_origin = anno.getanno(node, anno.Basic.ORIGIN)
    self.assertEqual(def_origin.loc.filename, 'test_file')
    self.assertEqual(def_origin.loc.lineno, 10)
    self.assertEqual(def_origin.loc.col_offset, 10)
    self.assertEqual(def_origin.source_code_line, 'def test_fn(x):')
    self.assertIsNone(def_origin.comment)

    docstring_origin = anno.getanno(node.body[0], anno.Basic.ORIGIN)
    self.assertEqual(def_origin.loc.filename, 'test_file')
    self.assertEqual(docstring_origin.loc.lineno, 11)
    self.assertEqual(docstring_origin.loc.col_offset, 12)
    self.assertEqual(docstring_origin.source_code_line, "  '''Docstring.'''")
    self.assertIsNone(docstring_origin.comment)

    ret_origin = anno.getanno(node.body[1], anno.Basic.ORIGIN)
    self.assertEqual(def_origin.loc.filename, 'test_file')
    self.assertEqual(ret_origin.loc.lineno, 12)
    self.assertEqual(ret_origin.loc.col_offset, 12)
    self.assertEqual(ret_origin.source_code_line, '  return x  # comment')
    self.assertEqual(ret_origin.comment, 'comment')

  def test_resolve_entity(self):
    test_fn = basic_definitions.simple_function
    node, source = parser.parse_entity(
        test_fn, inspect_utils.getfutureimports(test_fn))
    origin_info.resolve_entity(node, source, test_fn)

    # The line numbers below should match those in basic_definitions.py

    def_origin = anno.getanno(node, anno.Basic.ORIGIN)
    self.assertEqual(def_origin.loc.lineno, 23)
    self.assertEqual(def_origin.loc.col_offset, 0)
    self.assertEqual(def_origin.source_code_line, 'def simple_function(x):')
    self.assertIsNone(def_origin.comment)

    docstring_origin = anno.getanno(node.body[0], anno.Basic.ORIGIN)
    self.assertEqual(docstring_origin.loc.lineno, 24)
    self.assertEqual(docstring_origin.loc.col_offset, 2)
    self.assertEqual(docstring_origin.source_code_line, '  """Docstring."""')
    self.assertIsNone(docstring_origin.comment)

    ret_origin = anno.getanno(node.body[1], anno.Basic.ORIGIN)
    self.assertEqual(ret_origin.loc.lineno, 25)
    self.assertEqual(ret_origin.loc.col_offset, 2)
    self.assertEqual(ret_origin.source_code_line, '  return x  # comment')
    self.assertEqual(ret_origin.comment, 'comment')

  def test_resolve_entity_nested_function(self):

    test_fn = basic_definitions.nested_functions
    node, source = parser.parse_entity(
        test_fn, inspect_utils.getfutureimports(test_fn))
    origin_info.resolve_entity(node, source, test_fn)

    # The line numbers below should match those in basic_definitions.py

    inner_def_origin = anno.getanno(node.body[1], anno.Basic.ORIGIN)
    self.assertEqual(inner_def_origin.loc.lineno, 31)
    self.assertEqual(inner_def_origin.loc.col_offset, 2)
    self.assertEqual(inner_def_origin.source_code_line, '  def inner_fn(y):')
    self.assertIsNone(inner_def_origin.comment)

    inner_ret_origin = anno.getanno(node.body[1].body[0], anno.Basic.ORIGIN)
    self.assertEqual(inner_ret_origin.loc.lineno, 32)
    self.assertEqual(inner_ret_origin.loc.col_offset, 4)
    self.assertEqual(inner_ret_origin.source_code_line, '    return y')
    self.assertIsNone(inner_ret_origin.comment)

  def test_resolve_entity_indented_block(self):

    test_fn = basic_definitions.SimpleClass.simple_method
    node, source = parser.parse_entity(
        test_fn, inspect_utils.getfutureimports(test_fn))
    origin_info.resolve_entity(node, source, test_fn)

    # The line numbers below should match those in basic_definitions.py

    def_origin = anno.getanno(node, anno.Basic.ORIGIN)
    self.assertEqual(def_origin.loc.lineno, 46)
    self.assertEqual(def_origin.loc.col_offset, 2)
    self.assertEqual(def_origin.source_code_line, 'def simple_method(self):')
    self.assertIsNone(def_origin.comment)

    ret_origin = anno.getanno(node.body[0], anno.Basic.ORIGIN)
    self.assertEqual(ret_origin.loc.lineno, 47)
    self.assertEqual(ret_origin.loc.col_offset, 4)
    self.assertEqual(ret_origin.source_code_line, '  return self')
    self.assertIsNone(ret_origin.comment)


if __name__ == '__main__':
  test.main()
