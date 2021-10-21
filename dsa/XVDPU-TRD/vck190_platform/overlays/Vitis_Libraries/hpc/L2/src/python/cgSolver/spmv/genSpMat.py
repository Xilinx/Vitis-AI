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
import subprocess
import pdb
import argparse
import os
import sys
import scipy.io
import scipy.stats
import numpy as np
import math
from cgSolver import CG_Solver


def get_matrix(mtxFileName, parEntries):
    l_arr = scipy.io.mmread(mtxFileName).toarray().astype(np.float64)
    l_n = math.ceil(l_arr.shape[0] / parEntries) * parEntries
    A = np.zeros([l_n, l_n], dtype=np.float64)
    A[:l_arr.shape[0], :l_arr.shape[1]] = l_arr
    for i in range(l_arr.shape[0], l_n):
        A[i, i] = 1
    return A, l_arr.shape[0]


def gen_ref(vecPath, mtxName, A, origDim, isVerify, isDebug):
    x = np.random.random(A.shape[0]).astype(np.float64)
    for i in range(origDim, A.shape[0]):
        x[i] = 0
    b = A @ x

    A.tofile(os.path.join(vecPath, mtxName, 'A.mat'))
    A.diagonal().tofile(os.path.join(vecPath, mtxName, 'A_diag.mat'))
    x.tofile(os.path.join(vecPath, mtxName, 'x.mat'))
    b.astype(np.float64).tofile(os.path.join(vecPath, mtxName, 'b.mat'))

    if isVerify:
        solver = CG_Solver(A, b, verify=isVerify, debug=isDebug,
                           maxIter=args.maxIter, tol=1e-8)
        x_cg, n_iter = solver.solve_Jacobi()
        x_cg.astype(np.float64).tofile(
            os.path.join(vecPath, mtxName, 'x_cg.mat'))
        print(">>>>> Solver finished at %d iterations." % n_iter)


def process_matrices(mtxList, parEntries, vecPath, isVerify, isDebug):
    downloadList = open(mtxList, 'r')
    downloadNames = downloadList.readlines()
    subprocess.run(["mkdir", "-p", vecPath])
    for line in downloadNames:
        mtxHttp = line.strip()
        strList = mtxHttp.split('/')
        mtxFileName = strList[len(strList) - 1]
        strList = mtxFileName.split('.')
        mtxName = strList[0]
        mtxFullName = './' + mtxName + '/' + mtxName + '.mtx'
        if not os.path.exists(mtxFileName):
            subprocess.run(["wget", mtxHttp])
        subprocess.run(["tar", "-xzf", mtxFileName])
        subprocess.run(["mkdir", "-p", vecPath + '/' + mtxName])
        A, origDim = get_matrix(mtxFullName, parEntries)
        gen_ref(vecPath, mtxName, A, origDim, isVerify, isDebug)
        subprocess.run(["rm", "-f", mtxFileName])
        subprocess.run(["rm", "-rf", mtxName + '/'])


def main(args):
    if (args.usage):
        print('Usage example:')
        print(
            'python genSpMat.py --mtxList ./test_matrices.txt --maxIter 16 --path ./vec_dat [--verify] [--debug]')
    else:
        process_matrices(args.mtxList, args.parEntries,
                         args.path, args.verify, args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate rhs vectors and golden references for CG+Jacobi with given sparse matrices list.')
    parser.add_argument(
        '--usage',
        action='store_true',
        help='print usage example'
    )
    parser.add_argument(
        '--mtxList',
        type=str,
        help='a file containing URLs for downloading sparse matrices'
    )
    parser.add_argument(
        '--parEntries',
        type=int,
        default=4,
        help='number of entries that are processed in parallel, the vector and matrix dimensions have to be padded to multiply of this number'
    )
    parser.add_argument(
        '--maxIter',
        type=int,
        default=32,
        help='Maximum No. Iterations')
    parser.add_argument(
        '--path',
        type=str,
        default='./vec_dat',
        help='directory for storing reference data, default value ./vec_dat')
    parser.add_argument(
        '--verify',
        action="store_true"
    )
    parser.add_argument(
        '--debug',
        action="store_true"
    )
    args = parser.parse_args()
    main(args)
