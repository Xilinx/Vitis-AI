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

#define MaxRowBlocks 18
#define MinRowBlocks 5

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
             hls::stream<ap_uint<SPARSE_dataBits> > p_rowValStr[SPARSE_parEntries],
             hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& p_rowAggStr) {
    rowAgg<SPARSE_parEntries, SPARSE_parGroups, SPARSE_dataType, SPARSE_indexType, SPARSE_dataBits>(
        p_rowBlocks, p_rowValStr, p_rowAggStr);
}

int main() {
    const unsigned int t_RowOffsetBits = SPARSE_indexBits - SPARSE_logParEntries - SPARSE_logParGroups;

    unsigned int l_rowBlocks = 0;
    hls::stream<ap_uint<SPARSE_dataBits> > l_rowValStr[SPARSE_parEntries];
    hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > l_rowAggStr;

    BitConv<SPARSE_dataType> l_conv;
    // srand(time(nullptr));
    while (l_rowBlocks < MinRowBlocks) {
        l_rowBlocks = rand() % MaxRowBlocks;
    }
    cout << "INFO: number of input blocks is: " << l_rowBlocks << endl;
    SPARSE_dataType l_inVal = (SPARSE_dataType)0;
    for (unsigned int i = 0; i < l_rowBlocks; ++i) {
        cout << "INFO: BLOCK " << i << endl;
        for (unsigned int b = 0; b < SPARSE_parEntries; ++b) {
            cout << "      BANK " << b << endl;
            cout << "           ";
            cout << "val = " << l_inVal << " ";
            ap_uint<SPARSE_dataBits> l_valBits = l_conv.toBits(l_inVal);
            l_rowValStr[b].write(l_valBits);
            l_inVal++;
            cout << endl;
        }
    }
    uut_top(l_rowBlocks, l_rowValStr, l_rowAggStr);

    unsigned int l_errs = 0;
    cout << "INFO: output blocks" << endl;
    for (unsigned int i = 0; i < l_rowBlocks; ++i) {
        cout << "      block " << i << endl;
        ap_uint<SPARSE_dataBits* SPARSE_parEntries> l_val = l_rowAggStr.read();
        for (unsigned int b = 0; b < SPARSE_parEntries; ++b) {
            ap_uint<SPARSE_dataBits> l_entryBits = l_val.range((b + 1) * SPARSE_dataBits - 1, b * SPARSE_dataBits);
            SPARSE_dataType l_entryVal = l_conv.toType(l_entryBits);
            SPARSE_dataType l_entryRef = i * SPARSE_parEntries + b;
            if (!compare(l_entryVal, l_entryRef)) {
                cout << "ERROR: at block " << i << " bank " << b;
                cout << "       output = " << l_entryVal << " refVal = " << l_entryRef << endl;
                l_errs++;
            }
            cout << "val = " << l_entryVal << " ";
        }
        cout << endl;
    }

    cout << "Total errors: " << l_errs << endl;
    if (l_errs != 0) {
        return -1;
    }
    return 0;
}
