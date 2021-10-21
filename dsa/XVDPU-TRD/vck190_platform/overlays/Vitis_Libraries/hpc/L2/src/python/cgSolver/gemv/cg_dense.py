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
import sys
import os
import argparse
import pdb
import pandas
from cgSolver import CG_Solver, compare


def main(args):
    if args.datatype == 'float':
        dtype = np.float32
    elif args.datatype == 'double':
        dtype = np.float64
    else:
        sys.exit("Wrong data type received.")
    if args.filepath:

        fileA = os.path.join(args.filepath, 'A.mat')
        A = np.fromfile(fileA, dtype=dtype)
        dim = int(np.sqrt(A.size))
        A = A.reshape([dim, dim])
        w, v = np.linalg.eig(A)
        assert(np.min(w) > 0)

        filex = os.path.join(args.filepath, 'x.mat')
        x = np.fromfile(filex, dtype=dtype)

        fileb = os.path.join(args.filepath, 'b.mat')
        b = np.fromfile(fileb, dtype=dtype)
    else:
        A = np.fromfile(args.filename, dtype=dtype)
        w, v = np.linalg.eig(A)
        assert(np.min(w) > 0)

        x = 2 * (np.random.random(A.shape[0]) - 0.5)
        b = A @ x

    solver = CG_Solver(
        A,
        b,
        verify=args.verify,
        maxIter=args.maxIter,
        tol=1e-8)

    x_cg, n_iter = solver.solve()
    print("No.iterations: ", n_iter)
    matches, ratio, rerr, aerr = compare(x_cg, x)
    if matches:
        print("Result from CG solver matches the solution.")
    else:
        print(
            "WARNING: %.2f %% matches the solution." % ratio)
        print("Max relative error is %f, max abosolute error is %f." %
              (rerr, aerr))
    print("=" * 80)

    x_cg, n_iter = solver.solve_Jacobi()
    print("No.iterations: ", n_iter)
    matches, ratio, rerr, aerr = compare(x_cg, x)
    if matches:
        print("Result from CG solver matches the solution.")
    else:
        print(
            "WARNING: %.2f %% matches the solution." % ratio)
        print("Max relative error is %f, max abosolute error is %f." %
              (rerr, aerr))
    print("=" * 80)


if __name__ == "__main__":
    from genMat import dense_SPD
    parser = argparse.ArgumentParser(
        description='Conjugate Gradient Solver for Sparse Matrix.')
    filePath = parser.add_mutually_exclusive_group(required=True)
    filePath.add_argument(
        '--filename',
        type=str,
        help='Dense matrix filename')
    filePath.add_argument(
        '--filepath',
        type=str,
        help='Dense matrix filepath')
    parser.add_argument(
        '--datatype',
        type=str,
        default='double',
        help="data type"
    )
    parser.add_argument(
        '--verify',
        action="store_true"
    )
    parser.add_argument(
        '--maxIter',
        type=int,
        default=1000,
        help='Maximum No. Iterations')
    args = parser.parse_args()
    main(args)
