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
#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "xf_sparse.hpp"
#include "L1_utils.hpp"
#include "uut_top.hpp"

#define NnzBlocks 16
#define ColBlocks 4

using namespace xf::sparse;
using namespace std;

int main(int argc, char** argv) {
    hls::stream<ap_uint<SPARSE_indexBits * SPARSE_parEntries> > l_colPtrStr;
    hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > l_colValStr;
    hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > l_nnzColValStr;
#pragma DATA_PACK variable = l_colPtrStr
#pragma DATA_PACK variable = l_colSelStr

    WideType<SPARSE_indexType, SPARSE_parEntries> l_colPtr;
    WideType<SPARSE_dataType, SPARSE_parEntries> l_colVal;
    ap_uint<SPARSE_indexBits * SPARSE_parEntries> l_colPtrBits;
    ap_uint<SPARSE_dataBits * SPARSE_parEntries> l_colValBits;

    vector<SPARSE_dataType> l_nnzColValGolden;

    bool l_isRndInput = (argc == 1);
    bool l_isFileInput = (argc == 2 || argc == 3);

    unsigned int l_nnzInputs = 0;
    cout << "Input:" << endl;

    unsigned int l_colPtrBlocks = ColBlocks;
    unsigned int l_nnzBlocks = NnzBlocks;

    if (l_isRndInput) {
        // srand(time(nullptr));
        for (unsigned int i = 0; i < ColBlocks; ++i) {
            for (unsigned int j = 0; j < SPARSE_parEntries; ++j) {
                l_colVal[j] = (SPARSE_dataType)(i * SPARSE_parEntries + j);
                SPARSE_indexType l_nnzsInCol = ((NnzBlocks * SPARSE_parEntries - l_nnzInputs) == 0)
                                                   ? 0
                                                   : rand() % (NnzBlocks * SPARSE_parEntries - l_nnzInputs);
                cout << "l_nnzsInCol: " << l_nnzsInCol << endl;
                l_colPtr[j] = ((i == ColBlocks - 1) && (j == SPARSE_parEntries - 1) &&
                               (l_nnzInputs < NnzBlocks * SPARSE_parEntries))
                                  ? NnzBlocks * SPARSE_parEntries
                                  : l_nnzInputs;
                if ((i == ColBlocks - 1) && (j == SPARSE_parEntries - 1) &&
                    (l_nnzInputs < NnzBlocks * SPARSE_parEntries)) {
                    l_colPtr[j] = NnzBlocks * SPARSE_parEntries;
                    l_nnzsInCol = NnzBlocks * SPARSE_parEntries - l_nnzInputs;
                } else {
                    l_colPtr[j] = l_nnzInputs + l_nnzsInCol;
                }
                l_nnzInputs += l_nnzsInCol;
                for (unsigned int k = 0; k < l_nnzsInCol; ++k) {
                    l_nnzColValGolden.push_back(l_colVal[j]);
                }
            }
            l_colPtrBits = l_colPtr;
            l_colValBits = l_colVal;
            l_colPtrStr.write(l_colPtrBits);
            l_colValStr.write(l_colValBits);
            cout << "colPtr " << i << ": " << l_colPtr << ", colVal " << i << ": " << l_colVal << endl;
        }
    } else if (l_isFileInput) {
        // Load input streams from file
        openInputFile(argv[1], [&](std::ifstream& l_if) {
            l_if.read(reinterpret_cast<char*>(&l_colPtrBlocks), sizeof(unsigned int));
            l_if.read(reinterpret_cast<char*>(&l_nnzBlocks), sizeof(unsigned int));
            loadStream(l_if, l_colPtrStr);
            loadStream(l_if, l_colValStr);
        });

        if (argc == 3) {
            hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > l_nnzColValGoldenStr;

            // Load output stream from file
            openInputFile(argv[2], [&](std::ifstream& l_if) { loadStream(l_if, l_nnzColValGoldenStr); });

            while (!l_nnzColValGoldenStr.empty()) {
                l_colVal = l_nnzColValGoldenStr.read();
                for (unsigned int i = 0; i < SPARSE_parEntries; i++) {
                    l_nnzColValGolden.push_back(l_colVal[i]);
                }
            }
        } else {
            // Generate refernce data
            auto l_colPtrVec = copyStreamToVector(l_colPtrStr);
            auto l_colValVec = copyStreamToVector(l_colValStr);
            SPARSE_indexType pos = 0;
            for (unsigned int i = 0; i < l_colPtrBlocks; ++i) {
                l_colPtr = l_colPtrVec[i];
                l_colVal = l_colValVec[i];
                for (unsigned int j = 0; j < SPARSE_parEntries; ++j) {
                    while (pos != l_colPtr[j] && pos < l_nnzBlocks * SPARSE_parEntries) {
                        l_nnzColValGolden.push_back(l_colVal[j]);
                        pos++;
                    }
                }
            }
            cout << l_nnzColValGolden.size() << endl;
        }
    }

    assert(l_nnzColValGolden.size() == l_nnzBlocks * SPARSE_parEntries);
    cout << "Golden reference:";
    for (unsigned int i = 0; i < l_nnzColValGolden.size(); ++i) {
        if ((i % SPARSE_parEntries) == 0) {
            cout << endl;
        }
        cout << setw(SPARSE_printWidth) << l_nnzColValGolden[i] << " ";
    }
    cout << endl;

    cout << "ColBlocks: " << l_colPtrBlocks << endl;
    cout << "NnzBlocks: " << l_nnzBlocks << endl;

    uut_top(l_colPtrBlocks, l_nnzBlocks, l_colPtrStr, l_colValStr, l_nnzColValStr);

    unsigned int l_errors = 0;
    for (unsigned int b = 0; b < l_nnzBlocks; ++b) {
        ap_uint<SPARSE_dataBits * SPARSE_parEntries> l_nnzColValBits;
        l_nnzColValBits = l_nnzColValStr.read();
        WideType<SPARSE_dataType, SPARSE_parEntries> l_nnzColVal(l_nnzColValBits);
        cout << "Output " << b << " : " << l_nnzColVal << endl;
        for (unsigned int i = 0; i < SPARSE_parEntries; ++i) {
            if (l_nnzColVal[i] != l_nnzColValGolden[b * SPARSE_parEntries + i]) {
                cout << "ERROR: "
                     << "block=" << b << " id=" << i;
                cout << " refVal=" << l_nnzColValGolden[b * SPARSE_parEntries + i];
                cout << " outVal=" << l_nnzColVal[i] << endl;
                l_errors++;
            }
        }
    }

    if (l_errors != 0) {
        return -1;
    } else {
        return 0;
    }
}
