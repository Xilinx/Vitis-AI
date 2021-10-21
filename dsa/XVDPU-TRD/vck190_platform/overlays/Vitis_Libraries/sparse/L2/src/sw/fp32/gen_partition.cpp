/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file gen_bin.cpp
 * @brief main function for generating memory image for parallelCscMV kernel
 *
 * This file is part of Vitis SPARSE Library.
 */
#include "L2_definitions.hpp"

using namespace std;
using namespace xf::sparse;
int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "ERROR: passed " << argc << " arguments, expected at least 2 arguments." << endl;
        cout << "  Usage: gen_partition.exe mtxFile" << endl;
        return EXIT_FAILURE;
    }

    string l_mtxFileName = argv[1];
    MtxFileType l_mtxFile;
    l_mtxFile.loadFile(l_mtxFileName);
    vector<NnzUnitType> l_a;
    if (l_mtxFile.good()) {
        l_a = l_mtxFile.getNnzUnits();
    }

    unsigned int l_nnzs = l_mtxFile.nnzs();
    unsigned int l_rows = l_mtxFile.rows();
    unsigned int l_cols = l_mtxFile.cols();

    vector<SPARSE_dataType> l_x;
    ProgramType l_program;
    GenVecType l_genColVec;
    GenCscPartitionType l_genCscPartition;

    l_genColVec.genEntVecFromRnd(l_cols, 10, 3 / 2, l_x);

    // sor l_a along cols
    sort(l_a.begin(), l_a.end());
    // duplicate the input Nnz Vector
    vector<NnzUnitType> l_aRef;
    for (unsigned int i = 0; i < l_nnzs; ++i) {
        if (l_a[i].getVal() != 0) {
            l_aRef.push_back(l_a[i]);
        }
    }
    // print out original matrix info
    cout << "INFO: input sparse matrix has:" << endl
         << "     " << l_rows << " rows" << endl
         << "     " << l_cols << " cols" << endl
         << "     " << l_nnzs << " nnzs" << endl
         << "     " << (l_a.front()).getRow() << " rowMinIdx" << endl
         << "     " << (l_a.front()).getCol() << " colMinIdx" << endl
         << "     " << (l_a.back()).getRow() << " rowMaxIdx" << endl
         << "     " << (l_a.back()).getCol() << " colMaxIdx" << endl;
    // generate partitions
    unsigned int l_rowsPerKernel[SPARSE_hbmChannels];
    unsigned int l_rowIdxBase[SPARSE_hbmChannels];
    unsigned int l_colsPerKernel;
    unsigned int l_colIdxBase;
    for (unsigned int i = 0; i < SPARSE_hbmChannels; ++i) {
        l_rowsPerKernel[i] = 0;
        l_rowIdxBase[i] = 0;
    }
    l_colsPerKernel = 0;
    l_colIdxBase = 0;
    bool l_genRes = true;
    unsigned int l_par = 0;
    vector<NnzUnitType> l_aOut;
    vector<SPARSE_dataType> l_xOut;
    while (l_genRes && l_nnzs != 0) {
        CscPartitionType l_cscPartition;
        l_genRes = l_genCscPartition.genPartition(l_rows, l_cols, l_rowsPerKernel, l_rowIdxBase, l_colsPerKernel,
                                                  l_colIdxBase, l_a, l_x, l_program, l_cscPartition);
        if (!l_genRes) {
            cout << "ERROR: failed to generate parition" << endl;
            return EXIT_FAILURE;
        }
        l_nnzs = l_a.size();
        // output row, col ranges in the partition
        cout << "INFO: partition " << l_par << endl;
        ColVecType l_colVec;
        l_colVec = l_cscPartition.getVec();
        cout << "    ColVec has: " << l_colVec.getEntries() << " entries." << endl;
        vector<SPARSE_dataType> l_xVec;
        l_colVec.loadVal(l_xVec);
        while (l_xVec.size() != l_colsPerKernel) {
            // there is padding in the colVector
            if (l_xVec.back() != 0) {
                cout << "ERROR: padded non-zeros in the partitioned vector" << endl;
                return EXIT_FAILURE;
            }
            l_xVec.pop_back();
        }
        if (l_colIdxBase == l_xOut.size()) {
            for (unsigned int i = 0; i < l_xVec.size(); ++i) {
                l_xOut.push_back(l_xVec[i]);
            }
        }
        for (unsigned int i = 0; i < SPARSE_hbmChannels; ++i) {
            CscMatType l_cscMat;
            l_cscMat = l_cscPartition.getMat(i);
            cout << "    CscMat[" << i << "] has:" << endl
                 << "        " << l_cscMat.getRows() << " rows" << endl
                 << "        " << l_cscMat.getCols() << " cols" << endl
                 << "        " << l_cscMat.getNnzs() << " nnzs" << endl
                 << "        " << l_cscMat.getRowMinIdx() << " rowMinIdx" << endl
                 << "        " << l_cscMat.getRowMaxIdx() << " rowMaxIdx" << endl
                 << "        " << l_cscMat.getColMinIdx() << " colMinIdx" << endl
                 << "        " << l_cscMat.getColMaxIdx() << " colMaxIdx" << endl;
            vector<NnzUnitType> l_nnzUnits;
            l_cscMat.loadNnzUnits(l_nnzUnits);
            for (unsigned int j = 0; j < l_cscMat.getNnzs(); ++j) {
                if (l_nnzUnits[j].getVal() != 0) {
                    l_aOut.push_back(l_nnzUnits[j]);
                }
            }
        }
        l_par++;
    }

    sort(l_aOut.begin(), l_aOut.end());
    if (l_aOut.size() != l_aRef.size()) {
        cout << "ERROR: sum of partitioned matrix has different size" << endl;
        cout << "       sum of partition matrix has " << l_aOut.size() << " NNZs" << endl;
        cout << "       input matrix has " << l_aRef.size() << " NNZs" << endl;
        return EXIT_FAILURE;
    }
    unsigned int l_errs = 0;
    for (unsigned int i = 0; i < l_aRef.size(); ++i) {
        bool l_sameRow = (l_aRef[i].getRow() == l_aOut[i].getRow());
        bool l_sameCol = (l_aRef[i].getCol() == l_aOut[i].getCol());
        bool l_sameVal = (l_aRef[i].getVal() == l_aOut[i].getVal());
        if (!l_sameRow || !l_sameCol || !l_sameVal) {
            l_errs++;
            cout << "ERROR: at nnz entry " << i << endl;
        }
    }

    if (l_xOut.size() != l_x.size()) {
        cout << "ERROR: sum of partitioned vector has different" << endl;
        cout << "       sum of partition vector has " << l_xOut.size() << " entries" << endl;
        cout << "       input vector has " << l_x.size() << " entries" << endl;
        return EXIT_FAILURE;
    }
    for (unsigned int i = 0; i < l_x.size(); ++i) {
        if (l_xOut[i] != l_x[i]) {
            l_errs++;
            cout << "ERROR: at vector entry " << i << endl;
        }
    }

    if (l_errs == 0) {
        cout << "TEST PASS" << endl;
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
