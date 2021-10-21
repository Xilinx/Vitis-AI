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
#include "uut_top.hpp"
#include "L1_utils.hpp"

using namespace xf::sparse;
using namespace xf::blas;
using namespace std;

int main() {
    const unsigned int t_MaxCols = SPARSE_parEntries * SPARSE_maxColParBlocks;
    const unsigned int t_IntsPerColParam = 2 + SPARSE_hbmChannels * 2;
    const unsigned int t_ParamsPerPar = SPARSE_dataBits * SPARSE_parEntries / 32;
    const unsigned int t_ParBlocks4Param = (t_IntsPerColParam + t_ParamsPerPar - 1) / t_ParamsPerPar;

    hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > l_datStr;
    hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> > l_datOutStr[SPARSE_hbmChannels];

    ap_uint<32> l_chBlocks[SPARSE_hbmChannels];
    ap_uint<32> l_nnzBlocks[SPARSE_hbmChannels];
    ap_uint<32> l_vecBlocks = 0;

    // srand(time(nullptr));
    for (unsigned int i = 0; i < SPARSE_hbmChannels; ++i) {
        do {
            l_chBlocks[i] = (ap_uint<32>)(rand() % SPARSE_maxColParBlocks);
            l_nnzBlocks[i] = l_chBlocks[i];
        } while (l_chBlocks[i] == 0);
    }

    cout << "Inputs:" << endl;
    for (unsigned int i = 0; i < SPARSE_hbmChannels; ++i) {
        l_vecBlocks += l_chBlocks[i];
        cout << "Channel " << i << endl;
        cout << "    chBlocks:  " << l_chBlocks[i] << endl;
        cout << "    nnzBlocks:  " << l_nnzBlocks[i] << endl;
    }
    cout << "Total blocks: " << l_vecBlocks << endl;

    ap_uint<32> l_params[t_ParBlocks4Param * t_ParamsPerPar];
    l_params[1] = l_vecBlocks;
    for (unsigned int i = 0; i < SPARSE_hbmChannels; ++i) {
        l_params[2 + i] = l_chBlocks[i];
        l_params[2 + SPARSE_hbmChannels + i] = l_nnzBlocks[i];
    }
    // send inputs to the stream
    for (unsigned int i = 0; i < t_ParBlocks4Param; ++i) {
        WideType<ap_uint<32>, t_ParamsPerPar> l_paramVal;
        for (unsigned int j = 0; j < t_ParamsPerPar; ++j) {
            l_paramVal[j] = l_params[i * t_ParamsPerPar + j];
        }
        l_datStr.write(l_paramVal);
    }
    for (unsigned int ch = 0; ch < SPARSE_hbmChannels; ++ch) {
        unsigned int l_blocks = l_chBlocks[ch];
        for (unsigned int i = 0; i < l_blocks; ++i) {
            ap_uint<SPARSE_dataBits * SPARSE_parEntries> l_valVecBits;
            WideType<ap_uint<SPARSE_dataBits>, SPARSE_parEntries> l_valVec;
            for (unsigned int j = 0; j < SPARSE_parEntries; ++j) {
                l_valVec[j] = (ap_uint<SPARSE_dataBits>)(ch + i * SPARSE_parEntries + j);
            }
            l_valVecBits = l_valVec;
            l_datStr.write(l_valVecBits);
        }
    }

    uut_top(l_datStr, l_datOutStr);

    unsigned int l_errs = 0;
    for (unsigned int t_chId = 0; t_chId < SPARSE_hbmChannels; ++t_chId) {
        cout << "INFO: channel id = " << t_chId << endl;
        WideType<ap_uint<32>, t_ParamsPerPar> l_paramVal = l_datOutStr[t_chId].read();
        ap_uint<32> l_chBlocksOut = l_paramVal[0];
        ap_uint<32> l_nnzBlocksOut = l_paramVal[1];
        if (l_chBlocksOut != l_chBlocks[t_chId]) {
            cout << "ERROR: output chBlocks " << l_chBlocksOut;
            cout << " != original blocks" << l_chBlocks[t_chId] << endl;
            l_errs++;
        }
        if (l_nnzBlocksOut != l_nnzBlocks[t_chId]) {
            cout << "ERROR: output blocks " << l_nnzBlocksOut;
            cout << " != original blocks" << l_chBlocks[t_chId] << endl;
            l_errs++;
        }
        for (unsigned int i = 0; i < l_chBlocks[t_chId]; ++i) {
            ap_uint<SPARSE_dataBits * SPARSE_parEntries> l_vecOutBits;
            l_vecOutBits = l_datOutStr[t_chId].read();
            WideType<ap_uint<SPARSE_dataBits>, SPARSE_parEntries> l_vecOut(l_vecOutBits);
            for (unsigned int j = 0; j < SPARSE_parEntries; ++j) {
                unsigned int l_idx = t_chId + i * SPARSE_parEntries + j;
                ap_uint<SPARSE_dataBits> l_valRef = l_idx;
                if (l_valRef != l_vecOut[j]) {
                    cout << "ERROR: l_valOut at index " << l_idx << endl;
                    cout << "       output is: " << l_vecOut[j] << " original value is: " << l_valRef << endl;
                    l_errs++;
                }
            }
        }
    }
    if (l_errs == 0) {
        cout << "TEST PASS!" << endl;
        return 0;
    } else {
        cout << "total errors: " << l_errs << endl;
        return -1;
    }
}
