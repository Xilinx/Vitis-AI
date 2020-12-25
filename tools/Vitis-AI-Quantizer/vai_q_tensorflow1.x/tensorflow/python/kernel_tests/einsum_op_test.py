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
"""Tests for tensorflow.ops.Einsum."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class EinsumOpTest(test.TestCase):

  def _check(self, s, *input_shapes, **kwargs):
    dtype = kwargs.pop('dtype', np.float32)
    r = np.random.RandomState(0)
    inputs = []
    for shape in input_shapes:
      arr = np.array(r.randn(*shape)).astype(dtype)
      if dtype == np.complex64 or dtype == np.complex128:
        arr += 1j * np.array(r.randn(*shape)).astype(dtype)
      inputs.append(arr)
    input_tensors = [constant_op.constant(x, shape=x.shape) for x in inputs]
    a = np.einsum(s, *inputs)
    b = self.evaluate(gen_linalg_ops.einsum(input_tensors, s))
    self.assertAllClose(a, b, atol=1e-4, rtol=1e-4)

  def testUnary(self):
    self._check('->', ())
    self._check('aa->', (3, 3))
    self._check('aa->a', (3, 3))
    self._check('aaa->', (3, 3, 3))
    self._check('aaa->a', (3, 3, 3))
    self._check('aab->a', (3, 3, 4))
    self._check('ab->', (3, 3))
    self._check('ab->ab', (3, 3))
    self._check('abc->b', (3, 4, 5))
    self._check('abc->ca', (3, 4, 5))
    self._check('abc->cab', (3, 4, 5))
    self._check('aabcc->a', (3, 3, 5, 4, 4))
    self._check('aabcc->ac', (3, 3, 5, 4, 4))
    self._check('aabcd->ad', (3, 3, 5, 4, 4))

  def testUnaryEllipsis(self):
    self._check('...->...', ())
    self._check('...->', ())
    self._check('->...', ())

    # Tests from dask
    self._check('a...a->a...', (2, 2))
    self._check('a...a->', (2, 2))
    self._check('a...a->...', (2, 5, 1, 2))
    self._check('a...a->a...', (2, 1, 2))
    self._check('a...a->a...', (2, 3, 4, 5, 2))

    self._check('...ijk->...ki', (3, 4, 5))
    self._check('...ijk->...ki', (1, 3, 4, 5))
    self._check('...ijk->...ki', (2, 2, 3, 4, 5))

    # Repeated indices.
    self._check('i...ii->...i', (3, 2, 3, 3))

  def testBinary(self):
    self._check(',->', (), ())
    self._check('a,a->', (3,), (3,))
    self._check('a,a->a', (3,), (3,))
    self._check('ba,b->', (3, 2), (3,))
    self._check('ab,b->a', (3, 4), (4,))
    self._check('ab,ab->', (3, 4), (3, 4))
    self._check('nij,jk->nik', (5, 2, 3), (3, 4))
    self._check('abc,bad->abcd', (1, 2, 3), (2, 1, 4))
    # Repeated indices.
    self._check('ijj,k->ik', (2, 3, 3), (4,))
    self._check('aba,a->b', (3, 4, 3), (3,))
    # From https://github.com/dask/dask/pull/3412#discussion_r182413444
    self._check('aab,bc->ac', (2, 2, 3), (3, 4))
    self._check('aab,bcc->ac', (2, 2, 3), (3, 4, 4))
    # Based on https://github.com/google/jax/issues/37#issuecomment-448572187
    self._check('sa,shb->shab', (2, 1), (2, 3, 4))

  def testBroadcasting(self):
    # Batch matmul without broadcasting.
    self._check('...ij,...jk->...ik', (5, 1, 2, 3), (5, 1, 3, 4))
    # Batch matmul with broadcasting.
    self._check('...ij,...jk->...ik', (1, 2, 3), (3, 5))
    self._check('...ij,...jk->...ik', (2, 3), (1, 3, 5))
    self._check('...ij,...jk->...ik', (5, 2, 3), (3, 5))
    self._check('...ij,...jk->...ik', (2, 3), (5, 3, 5))
    self._check('...ij,...jk->...ik', (3, 1, 2, 3), (1, 1, 7, 3, 5))
    self._check('i...j,j...k->...ik', (2, 1, 3, 1, 3), (3, 1, 7, 5))
    # Broadcasting with repeated indices.
    self._check('ij,jk...k->i...', (3, 2), (2, 4, 1, 4))
    self._check('ij,jk...k->...i', (3, 2), (2, 4, 5, 4))
    self._check('ijj,jk...k->i...', (3, 2, 2), (2, 4, 1, 4))
    self._check('i...jj,jk...k->i...', (3, 3, 1, 2, 2), (2, 4, 1, 5, 4))
    # Following 2 from # https://stackoverflow.com/a/19203475/1611416
    self._check('...abc,...abcd->...d', (1, 1, 2, 3, 4), (5, 2, 3, 4, 6))
    self._check('ab...,b->ab...', (2, 3, 1, 1, 5), (3,))

  def testDtypes(self):
    for dtype in [np.float64, np.float32, np.complex64, np.complex128]:
      self._check('ij,jk->ik', (2, 2), (2, 2), dtype=dtype)
      self._check('ji,jk->ik', (2, 2), (2, 2), dtype=dtype)
      self._check('ji,kj->ik', (2, 2), (2, 2), dtype=dtype)
      self._check('ij,jk->ki', (2, 2), (2, 2), dtype=dtype)
      self._check('ji,kj->ki', (2, 2), (2, 2), dtype=dtype)

  @test_util.run_in_graph_and_eager_modes
  def testInvalid(self):
    r = np.random.RandomState(0)
    cases = [
        # incorrect rank.
        ('ij,jk->ik', r.randn(1, 2, 3), r.randn(3, 4)),
        ('...ij,jk->ik', r.randn(3), r.randn(3, 4)),
        # inconsistent dimensions.
        ('ij,jk->ik', r.randn(2, 3), r.randn(4, 4)),
        # broadcasting is invalid
        ('...ij,...jk->...ik', r.randn(5, 2, 3), r.randn(7, 3, 4)),
        # output should have ellipsis when broadcasting shape is
        # non-empty.
        ('...ij,...jk->ik', r.randn(2, 2, 3), r.randn(3, 4)),
    ]
    for args in cases:
      with self.assertRaises((ValueError, errors.InvalidArgumentError)):
        _ = self.evaluate(gen_linalg_ops.einsum(args[1:], args[0]))

      placeholders = [
          array_ops.placeholder_with_default(x, shape=None) for x in args[1:]
      ]
      with self.assertRaises((ValueError, errors.InvalidArgumentError)):
        _ = self.evaluate(gen_linalg_ops.einsum(placeholders, args[0]))

  @test_util.run_in_graph_and_eager_modes
  def testPlaceholder(self):

    def check(equation, *input_and_placeholder_shapes):
      r = np.random.RandomState(0)
      inputs = []
      input_placeholders = []
      for actual_shape, placeholder_shape in input_and_placeholder_shapes:
        input_np = np.array(r.randn(*actual_shape))
        inputs.append(input_np)
        input_placeholders.append(
            array_ops.placeholder_with_default(input_np, placeholder_shape))

      a = np.einsum(equation, *inputs)
      b = self.evaluate(gen_linalg_ops.einsum(input_placeholders, equation))
      self.assertAllClose(a, b, atol=1e-4, rtol=1e-4)

    check('bijl,bjkm->bik', ((9, 2, 3, 5), (None, None, None, 5)),
          ((9, 3, 4, 7), (None, None, 4, None)))
    check('bijl,bjkm->bik', ((9, 2, 3, 5), None), ((9, 3, 4, 7), None))
    check('...ij,...->...i', ((4, 3, 1, 2), (None, 3, None, 2)),
          ((4, 3), (None, 3)))
    check('...ij,...jk->...ik', ((3, 1, 2, 3), None), ((1, 7, 3, 4), None))

  def testOutputRepeatedLabels(self):
    # This is the reverse operation of repeated input labels, to be used for
    # computing symbolic gradients of einsum.
    r = np.random.RandomState(0)
    a = r.randn(2, 2)
    s = 'a->aa'
    diag_a = np.diag(np.diag(a))
    b = self.evaluate(gen_linalg_ops.einsum([np.diag(a)], s))
    self.assertAllClose(diag_a, b, atol=1e-4, rtol=1e-4)


class EinsumBenchmark(test.Benchmark):
  cases = [
      # Unary cases.
      ['ijk->i', 100],
      ['ijk->kji', 100],
      # Regular matmul or batch matmul.
      ['ij,jk->ik', 1000],
      ['ji,kj->ik', 1000],
      ['ab,ab->', 100],
      ['ab,ba->', 100],
      ['abc,abc->', 100],
      ['abc,bac->', 100],
      ['abc,cba->', 100],
      ['bij,bjk->bik', 100],
      ['bji,bjk->bki', 100],
      ['ikl,kji->kl', 100],
      ['klj,lki->ij', 100],
      ['ijk,ilj->kli', 100],
      ['kij,mkb->ijmb', 100],
      ['abcd,ad->bc', 40],
      # Larger binary contractions.
      ['ijk,jklm->il', 40],
      ['efabc,eabcd->efd', 30],
      ['fabec,abcde->fde', 30],
      ['efabc,edabc->efd', 30],
      ['eadbf,dfebc->ecfad', 30],
      ['abcdef,bcdfg->abcdeg', 30],
  ]

  def benchmarkEinsum(self):
    for equation, dim in self.cases:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device('/cpu:0'):
        r = np.random.RandomState(0)
        input_subscripts = equation.split('->')[0].split(',')
        input_vars = []
        for subscript in input_subscripts:
          input_shape = (dim,) * len(subscript)
          input_vars.append(
              variables.Variable(np.array(r.randn(*input_shape), np.float32)))
        variables.global_variables_initializer().run()

        # Call einsum_v1.
        self.run_op_benchmark(
            sess,
            special_math_ops.einsum(equation, *input_vars),
            min_iters=50,
            name='einsum_v1_cpu_({})_{}'.format(equation, dim))

        # Call gen_linalg_ops.einsum.
        self.run_op_benchmark(
            sess,
            gen_linalg_ops.einsum(input_vars, equation),
            min_iters=50,
            name='einsum_v2_cpu_({})_{}'.format(equation, dim))


if __name__ == '__main__':
  test.main()
