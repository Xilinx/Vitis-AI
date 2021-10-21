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
import scipy.io as sio
import scipy.sparse as sp
from os import path
import subprocess
import argparse
import math
from matrix_params import *

def write_vec(p_vec, p_vecFileName):
    fo = open(p_vecFileName, "wb")
    p_vec.tofile(fo)
    fo.close()

def gen_vectors(spm, isPcg, mtxName, vecPath, parEntries):
    if isPcg:
        l_inVecFileName = vecPath+'/'+mtxName+'/x.mat'
        l_refVecFileName = vecPath+'/'+mtxName+'/b.mat'
        l_diagMatFileName = vecPath+'/'+mtxName+'/A_diag.mat'
    else:
        l_inVecFileName = vecPath+'/'+mtxName+'/inVec.dat'
        l_refVecFileName = vecPath+'/'+mtxName+'/refVec.dat'
        l_diagMatFileName = vecPath+'/'+mtxName+'/mat_diag.dat'
    
    l_nPad = ((spm.n+parEntries-1)//parEntries) * parEntries
    l_mPad = ((spm.m+parEntries-1)//parEntries) * parEntries
    l_inVec = np.zeros(l_nPad, dtype=np.float64)
    l_refVec = np.zeros(l_mPad, dtype=np.float64)
    l_inVec[0:spm.n] = np.random.rand(spm.n).astype(np.float64)
    l_cooMat = sp.coo_matrix((spm.data, (spm.row, spm.col)), shape=(spm.m, spm.n), dtype=np.float64)
    l_refVec[0:spm.m] = l_cooMat.dot(l_inVec[0:spm.n])
    if spm.m == spm.n:
        l_diagVec = np.full(l_mPad, 1, dtype=np.float64)
        for i in range(spm.nnz):
            if spm.row[i] == spm.col[i]:
                l_diagVec[spm.row[i]] = spm.data[i]
        write_vec(l_diagVec, l_diagMatFileName)
    write_vec(l_inVec, l_inVecFileName)
    write_vec(l_refVec, l_refVecFileName)

def process_matrices(isGenVec, isPcg, isClean, mtxList, parEntries, vecPath):
    downloadList = open(mtxList, 'r')
    downloadNames = downloadList.readlines()
    if not path.exists(vecPath):
        subprocess.run(["mkdir","-p",vecPath])
    if not path.exists("./mtx_files"):
        subprocess.run(["mkdir", "-p", "./mtx_files"])
    for line in downloadNames:
        mtxHttp = line.strip()
        strList = mtxHttp.split('/')
        mtxFileName = strList[len(strList)-1]
        strList = mtxFileName.split('.')
        mtxName = strList[0]
        mtxFullName = './mtx_files/' + mtxName+'/'+ mtxName + '.mtx'
        mtxVecPath = vecPath+'/'+mtxName
        if not path.exists(mtxVecPath):
            subprocess.run(["mkdir", "-p", mtxVecPath])
        if not path.exists(mtxFullName):
            subprocess.run(["wget", mtxHttp, "-P", "./mtx_files"])
            subprocess.run(["tar", "-xzf", './mtx_files/'+mtxFileName, "-C", "./mtx_files"])
        if isGenVec:
            spm = sparse_matrix()
            spm.read_matrix(mtxFullName, mtxName)
            gen_vectors(spm, isPcg, mtxName, vecPath, parEntries)
        if isClean:
            subprocess.run(["rm", "-rf", './mtx_files/'+mtxName+'/'])
    if isClean:
        subprocess.run(["rm", "-rf", "./mtx_files"])

    
def main(args):
    if (args.usage):
        print('Usage example:')
        print('python gen_vectors.py --gen_vec [--pcg] [--clean] --mtx_list ./test_matrices.txt --vec_path ./vec_dat')
    else:
        process_matrices(args.gen_vec, args.pcg, args.clean, args.mtx_list, args.par_entries, args.vec_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='read sparse matrix file, generate input vector and golden reference vector for SpMv')
    parser.add_argument('--usage',action='store_true',help='print usage example')
    parser.add_argument('--gen_vec',action='store_true',help='generate input and output vectors for a set of sparse matrices')
    parser.add_argument('--pcg',action='store_true',help='generate the vector files required by PCG solver')
    parser.add_argument('--clean',action='store_true',help='clean up downloaded .mtx file after the run')
    parser.add_argument('--mtx_list',type=str,help='a file containing URLs for downloading sprase matrices')
    parser.add_argument('--par_entries',type=int,default=4,help='number of NNZ entries retrieved from one HBM channel')
    parser.add_argument('--vec_path',type=str,default='./vec_dat',help='directory for storing vectors, default value ./vec_dat')
    args = parser.parse_args()
    main(args)
  
