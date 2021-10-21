#!/usr/bin/env python3
# Copyright 2019 Xilinx, Inc.
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

import numpy as np
import scipy.stats
import sys
import os
import argparse
import pdb
from cgSolver import CG_Solver


def dense_SPD(N, k=128):
    a = np.random.random(size=(N, N))
    ortho = scipy.linalg.orth(a)
    #ortho = scipy.stats.ortho_group.rvs(N)
    vec = np.linspace(0.1, k / 10, N)
    M = np.diag(vec)
    A = ortho.transpose() @ M @ ortho
    return A


def main(args):
    if args.datatype == 'float':
        dtype = np.float32
    elif args.datatype == 'double':
        dtype = np.float64
    else:
        sys.exit("Wrong data type received.")

    A = dense_SPD(args.dimension, args.condition_number).astype(dtype=dtype)
    x = np.random.random(args.dimension).astype(dtype=dtype)
    b = A @ x

    A.tofile(os.path.join(args.path, 'A.mat'))
    x.tofile(os.path.join(args.path, 'x.mat'))
    b.tofile(os.path.join(args.path, 'b.mat'))

    if args.verify:
        solver = CG_Solver(A, b, debug=args.debug,
                           maxIter=args.maxIter, tol=1e-8)
        if args.preconditioner is None:
            x_cg, n_iter = solver.solve()
        elif args.preconditioner == "Jacobi":
            x_cg, n_iter = solver.solve_Jacobi()

        print(">>>>> Solver finished at %d iterations." % n_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate SPD Matrix.')
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='result file')
    parser.add_argument(
        '--condition_number',
        type=int,
        default=128,
        help='Matrix condition number')
    parser.add_argument(
        '--maxIter',
        type=int,
        default=32,
        help='Maximum No. Iterations')
    parser.add_argument(
        '--dimension',
        type=int,
        default=32,
        help='Dense matrix dimension')
    parser.add_argument(
        '--verify',
        action="store_true"
    )
    parser.add_argument(
        '--debug',
        action="store_true"
    )
    parser.add_argument(
        '--datatype',
        type=str,
        default='double',
        help="data type"
    )
    parser.add_argument(
        '--preconditioner',
        type=str,
        help="preconditioner name"
    )
    args = parser.parse_args()
    main(args)
