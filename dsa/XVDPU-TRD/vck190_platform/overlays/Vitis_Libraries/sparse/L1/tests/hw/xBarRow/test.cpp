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
#include <vector>
#include <cstdlib>
#include <ctime>
#include <unordered_map>
#include "xf_sparse.hpp"

#define NnzBlocks 4
#define MaxRows 2048
using namespace xf::sparse;
using namespace xf::blas;
using namespace std;

void uut_top(unsigned int p_nnzBlocks,
             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& p_nnzValStr,
             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& p_nnzColValStr,
             hls::stream<ap_uint<SPARSE_indexBits * SPARSE_parEntries> >& p_rowIndexStr,
             hls::stream<ap_uint<SPARSE_dataBits + SPARSE_indexBits> > p_rowEntryStr[SPARSE_parEntries],
             hls::stream<ap_uint<1> > p_isEndStr[SPARSE_parEntries]) {
    xBarRow<SPARSE_logParEntries, SPARSE_dataType, SPARSE_indexType, SPARSE_dataBits, SPARSE_indexBits>(
        p_nnzBlocks, p_nnzValStr, p_nnzColValStr, p_rowIndexStr, p_rowEntryStr, p_isEndStr);
}

int main() {
    hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > l_nnzValStr;
    hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > l_nnzColValStr;
    hls::stream<ap_uint<SPARSE_indexBits * SPARSE_parEntries> > l_rowIndexStr;
    hls::stream<ap_uint<SPARSE_dataBits + SPARSE_indexBits> > l_rowEntryStr[SPARSE_parEntries];
    hls::stream<ap_uint<1> > l_isEndStr[SPARSE_parEntries];

    WideType<SPARSE_dataType, SPARSE_parEntries> l_nnzVal;
    WideType<SPARSE_dataType, SPARSE_parEntries> l_nnzColVal;
    WideType<SPARSE_indexType, SPARSE_parEntries> l_rowIndex;

    ap_uint<SPARSE_dataBits * SPARSE_parEntries> l_nnzValBits;
    ap_uint<SPARSE_dataBits * SPARSE_parEntries> l_nnzColValBits;
    ap_uint<SPARSE_dataBits * SPARSE_parEntries> l_rowIndexBits;

    unordered_map<SPARSE_dataType, SPARSE_indexType> l_inputArr;
    // srand(time(nullptr));
    cout << "Inputs:" << endl;
    for (unsigned int i = 0; i < NnzBlocks; ++i) {
        for (unsigned int b = 0; b < SPARSE_parEntries; ++b) {
            l_nnzVal[b] = i * SPARSE_parEntries + b;
            l_nnzColVal[b] = l_nnzVal[b] + 1;
            l_rowIndex[b] = rand() % MaxRows;
            l_inputArr[l_nnzVal[b] * l_nnzColVal[b]] = l_rowIndex[b];
        }
        l_nnzValBits = l_nnzVal;
        l_nnzColValBits = l_nnzColVal;
        l_rowIndexBits = l_rowIndex;
        l_nnzValStr.write(l_nnzValBits);
        l_nnzColValStr.write(l_nnzColValBits);
        l_rowIndexStr.write(l_rowIndexBits);
        cout << "Nnz Block " << i << endl;
        cout << "    Nnz vals:        " << l_nnzVal << endl;
        cout << "    Nnz colVals:     " << l_nnzColVal << endl;
        cout << "    Nnz Row Indices: " << l_rowIndex << endl;
    }

    uut_top(NnzBlocks, l_nnzValStr, l_nnzColValStr, l_rowIndexStr, l_rowEntryStr, l_isEndStr);

    ap_uint<SPARSE_parEntries> l_isEnd = 0;
    ap_uint<SPARSE_parEntries> l_activity = 0;
    l_activity.b_not();
    ap_uint<1> l_exit = 0;

    unsigned int l_nnzs = 0;
    unsigned int l_errors = 0;
    while (!l_exit) {
        if (l_isEnd.and_reduce() && !l_activity.or_reduce()) {
            l_exit = true;
        }
        l_activity = 0;
        for (unsigned int b = 0; b < SPARSE_parEntries; ++b) {
            ap_uint<1> l_unused;
            if (l_isEndStr[b].read_nb(l_unused)) {
                l_isEnd[b] = 1;
            }
            RowEntry<SPARSE_dataType, SPARSE_indexType, SPARSE_dataBits, SPARSE_indexBits> l_rowEntry;
            ap_uint<SPARSE_dataBits + SPARSE_indexBits> l_rowEntryBits;
            if (l_rowEntryStr[b].read_nb(l_rowEntryBits)) {
                l_rowEntry.toVal(l_rowEntryBits);
                l_activity[b] = 1;
                cout << "Bank " << b << ": " << l_rowEntry;
                if (l_inputArr.find(l_rowEntry.getVal()) == l_inputArr.end()) {
                    cout << " ERROR: val " << l_rowEntry.getVal() << " doesn't exist!";
                    l_errors++;
                } else if (l_inputArr[l_rowEntry.getVal()] != l_rowEntry.getRow()) {
                    cout << " ERROR: row " << l_rowEntry.getRow() << "doesn't exist!";
                    l_errors++;
                } else if ((l_rowEntry.getRow() % SPARSE_parEntries) != b) {
                    cout << " ERROR";
                    l_errors++;
                }
                cout << endl;
                l_nnzs++;
            }
        }
    }

    cout << "totoal nnzs: " << l_nnzs << endl;
    cout << "totoal errors: " << l_errors << endl;
    if (l_errors == 0) {
        return 0;
    } else {
        return -1;
    }
}
