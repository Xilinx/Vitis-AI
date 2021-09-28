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
#define NnzGroupEntries 18
#define MaxNumSameRows 10
using namespace xf::sparse;
using namespace std;

template <typename T>
bool isClose(float p_tolRel, float p_tolAbs, T p_vRef, T p_v, bool& p_exactMatch) {
    float l_diffAbs = abs(p_v - p_vRef);
    p_exactMatch = (p_vRef == p_v);
    bool l_status = (l_diffAbs <= (p_tolAbs + p_tolRel * abs(p_vRef)));
    return (l_status);
}
template <typename T>
bool compare(T x, T ref) {
    return x == ref;
}

template <>
bool compare<double>(double x, double ref) {
    bool l_exactMatch;
    return isClose<float>(1e-3, 3e-6, x, ref, l_exactMatch);
}
template <>
bool compare<float>(float x, float ref) {
    bool l_exactMatch;
    return isClose<float>(1e-3, 3e-6, x, ref, l_exactMatch);
}

void uut_top(const unsigned int p_rowBlocks,
             hls::stream<ap_uint<SPARSE_dataBits + SPARSE_indexBits> >& p_rowEntryStr,
             hls::stream<ap_uint<1> >& p_isEndStr,
             hls::stream<ap_uint<SPARSE_dataBits> >& p_rowValStr) {
    rowMemAcc<SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_dataType, SPARSE_indexType, SPARSE_dataBits,
              SPARSE_indexBits>(p_rowBlocks, p_rowEntryStr, p_isEndStr, p_rowValStr);
}

int main() {
    const unsigned int t_RowOffsetBits = SPARSE_indexBits;

    unsigned int l_rowBlocks = 0;
    hls::stream<ap_uint<SPARSE_dataBits + t_RowOffsetBits> > l_rowEntryStr;
    hls::stream<ap_uint<1> > l_isEndStr;
    unordered_map<SPARSE_indexType, SPARSE_dataType> l_goldenArr;
    unordered_map<SPARSE_indexType, SPARSE_dataType>::iterator l_goldenArrIt;

    // srand(time(nullptr));
    cout << "Inputs:"
         << "  "
         << "row offset"
         << "val" << endl;
    unsigned int l_nnzRows = 0;
    unsigned int l_nnzGroupEntries = 0;
    unsigned int l_numSameRows = 0;
    unsigned int l_maxNumSameRows = (rand() % MaxNumSameRows) + 1;
    SPARSE_indexType l_row;
    l_row = rand() % SPARSE_maxRowBlocks;
    while (l_nnzGroupEntries < NnzGroupEntries) {
        RowEntry<SPARSE_dataType, SPARSE_indexType, SPARSE_dataBits, t_RowOffsetBits> l_rowEntry;
        l_rowEntry.getVal() = l_nnzGroupEntries;
        if (l_numSameRows < l_maxNumSameRows) {
            if (l_row > l_rowBlocks) {
                l_rowBlocks = l_row;
            }
            l_rowEntry.getRow() = l_row * SPARSE_parEntries;
            l_rowEntryStr.write(l_rowEntry.toBits());
            cout << "Nnz " << l_nnzGroupEntries << ": " << l_rowEntry << endl;
            if (l_goldenArr.find(l_row) == l_goldenArr.end()) {
                l_goldenArr[l_row] = l_rowEntry.getVal();
                l_nnzRows++;
            } else {
                l_goldenArr[l_row] += l_rowEntry.getVal();
            }
            l_numSameRows++;
            l_nnzGroupEntries++;
        } else {
            l_numSameRows = 0;
            SPARSE_indexType l_rowTmp = rand() % SPARSE_maxRowBlocks;
            while (l_rowTmp == l_row) {
                l_rowTmp = rand() % SPARSE_maxRowBlocks;
            }
            l_row = l_rowTmp;
            l_maxNumSameRows = (rand() % MaxNumSameRows) + 1;
        }
    }
    l_isEndStr.write(1);

    l_goldenArrIt = l_goldenArr.begin();
    unsigned int l_goldenIndex = 0;
    while (l_goldenArrIt != l_goldenArr.end()) {
        cout << "Golden res " << l_goldenIndex << ": " << setw(SPARSE_printWidth) << l_goldenArrIt->first
             << setw(SPARSE_printWidth) << l_goldenArrIt->second << endl;
        l_goldenArrIt++;
        l_goldenIndex++;
    }

    hls::stream<ap_uint<SPARSE_dataBits> > l_rowValStr;

    uut_top(l_rowBlocks + 1, l_rowEntryStr, l_isEndStr, l_rowValStr);

    unsigned int l_rowValOutIdx = 0;
    unsigned int l_errors = 0;
    while (l_rowValOutIdx <= l_rowBlocks) {
        ap_uint<SPARSE_dataBits> l_rowValBits;
        l_rowValStr.read(l_rowValBits);
        BitConv<SPARSE_dataType> l_conVal;
        SPARSE_dataType l_rowVal = l_conVal.toType(l_rowValBits);

        cout << "Output row " << l_rowValOutIdx << ": " << l_rowVal;
        if ((l_goldenArr.find((SPARSE_indexType)l_rowValOutIdx) == l_goldenArr.end()) &&
            (!compare<SPARSE_dataType>(l_rowVal, (SPARSE_dataType)0))) {
            cout << "ERROR: Row offset " << l_rowValOutIdx << "doesn't exist!" << endl;
            l_errors++;
        } else if (!compare<SPARSE_dataType>(l_goldenArr[l_rowValOutIdx], l_rowVal)) {
            cout << "ERROR: Row offset " << l_rowValOutIdx << " has accumulation error.";
            cout << "refVal = " << l_goldenArr[l_rowValOutIdx] << " realVal = " << l_rowVal;
            cout << endl;
            l_errors++;
        } else {
            l_goldenArr[l_rowValOutIdx] -= l_rowVal;
        }
        cout << endl;
        l_rowValOutIdx++;
    }

    l_goldenArrIt = l_goldenArr.begin();
    while (l_goldenArrIt != l_goldenArr.end()) {
        if (!compare<SPARSE_dataType>(l_goldenArrIt->second, (SPARSE_dataType)0)) {
            cout << "ERROR: Row offset " << l_goldenArrIt->first << " has accumulation error.";
            cout << " Left over data in Golden reference array is: " << l_goldenArrIt->second << endl;
            l_errors++;
        }
        l_goldenArrIt++;
    }

    cout << "total output rowRows: " << l_rowBlocks + 1 << endl;
    cout << "total errors: " << l_errors << endl;
    if (l_errors == 0) {
        return 0;
    } else {
        return -1;
    }
}
