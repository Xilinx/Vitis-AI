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
import scipy.io
import sys
import argparse
import pdb
import pandas
from cgSolver import CG_Solver, compare


def load_sparse_matrix(filename):
    matrix = scipy.io.mmread(filename)


def main(args):
    A = scipy.io.mmread(args.filename).toarray()
    w, v = np.linalg.eig(A)
    assert(np.min(w) > 0)

    x = 2 * (np.random.random(A.shape[0]) - 0.5)
    b = A @ x

    solver = CG_Solver(A, b, verify=True, maxIter=3000, tol=1e-8)

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

    x_cg, n_iter = solver.solve_SSOR()
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

    x_cg, n_iter = solver.solve_ICH()
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
    parser.add_argument(
        '--filename',
        type=str,
        required=True
        help='Sparse matrix file')
    args = parser.parse_args()
    main(args)
