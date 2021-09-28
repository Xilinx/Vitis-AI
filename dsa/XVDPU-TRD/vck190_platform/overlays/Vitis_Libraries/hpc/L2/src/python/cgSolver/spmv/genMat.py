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
import scipy.io as sio
import scipy.sparse as sp
from cgSolver import CG_Solver


def main(args):
    if args.datatype == 'float':
        dtype = np.float32
    elif args.datatype == 'double':
        dtype = np.float64
    else:
        sys.exit("Wrong data type received.")

    sparse_mat_name = os.path.splitext(os.path.basename(args.mtx))[0]

    sparse_mat = sio.mmread(args.mtx)
    dense_mat = sparse_mat.toarray().astype(dtype=dtype)

    dimension = sparse_mat.shape[0]

    if args.readX == 'False':
        x = np.random.random(dimension).astype(dtype=dtype)
        x.tofile(os.path.join(args.path, 'x.mat'))
    else:
        print("read X from file")
        x = np.fromfile(args.path + 'x.mat', dtype=dtype)

    b = dense_mat @ x
    b.tofile(os.path.join(args.path, 'b.mat'))

    dense_mat.tofile(os.path.join(args.path, 'A.mat'))

    csr_mat = sp.csr_matrix(sparse_mat)

    csr_data = csr_mat.data.astype(dtype=dtype)
    csr_indices = csr_mat.indices.astype(dtype=np.int32)
    csr_indptr = csr_mat.indptr.astype(dtype=np.int32)

    csr_diagonal = (1.0 / sparse_mat.diagonal()).astype(dtype=dtype)

    with open(args.path + "info.txt", 'w') as info_file:
        info_file.write(sparse_mat_name + '\n')
        info_file.write(str(sparse_mat.shape[0]) + '\n')
        info_file.write(str(sparse_mat.nnz) + '\n')
        info_file.close()

    csr_data.tofile(os.path.join(args.path, 'data.bin'))
    csr_indices.tofile(os.path.join(args.path, 'indices.bin'))
    csr_indptr.tofile(os.path.join(args.path, 'indptr.bin'))
    csr_diagonal.tofile(os.path.join(args.path, 'diagonal.bin'))

    if args.verify:
        solver = CG_Solver(dense_mat, b, debug=args.debug,
                           maxIter=args.maxIter, tol=1e-8)
        if args.preconditioner is None:
            x_cg, n_iter = solver.solve()
        elif args.preconditioner == "Jacobi":
            x_cg, n_iter = solver.solve_Jacobi()

        diff = np.isclose(x, x_cg, 1e-3, 1e-5)
        mismatch = ((diff == 0).nonzero())
        for i in mismatch[0]:
            print(x[i], x_cg[i])

        print(">>>>> Solver finished at %d iterations in Python." % n_iter)


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
    parser.add_argument('--mtx', type=str)
    parser.add_argument('--readX', type=str, default=False)
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
