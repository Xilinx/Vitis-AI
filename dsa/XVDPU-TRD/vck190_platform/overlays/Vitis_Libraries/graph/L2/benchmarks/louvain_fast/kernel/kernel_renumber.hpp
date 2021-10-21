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
 * @file kernel_renumber.cpp
 *
 * @brief This file contains top function of test case.
 */

#ifndef _KERNEL_RENUMBER_HPP_
#define _KERNEL_RENUMBER_HPP_

#include "louvain_modularity.hpp"

namespace xf {
namespace graph {

template <int DWIDTH, int VERTEX>
void renumberClusters(int64_t numVertex,
                      int64_t& numClusters,
                      ap_uint<DWIDTH> cidPrev[VERTEX],
                      ap_uint<DWIDTH> cidSize[VERTEX]) {
#pragma HLS INLINE off

    numClusters = 0;
    const int NvBt = 32;
    const int NV = DWIDTH / NvBt;
    const int numNV = (numVertex + (NV - 1)) / NV;

    ap_uint<NvBt> csize, cidOld, cidNew;
    ap_uint<DWIDTH> tmpSize;
    ap_uint<DWIDTH> mapSize;
    ap_uint<DWIDTH> tmpCid, tmpNew;

    f_cast_<int, ap_uint<32> > ss, cc;

CSIZE_SCAN:
    for (int i = 0; i < numNV; i++) {
#pragma HLS PIPELINE
        tmpSize = cidSize[i];
        for (int j = 0; j < NV; j++) {
#pragma HLS UNROLL
            if ((i * NV + j) < numVertex) {
                csize = tmpSize((j + 1) * NvBt - 1, j * NvBt);
                if (csize > 0) {
                    mapSize((j + 1) * NvBt - 1, j * NvBt) = numClusters++; // cid > 0 , based cid=1
                } else {
                    mapSize((j + 1) * NvBt - 1, j * NvBt) = 0; // cid is a invalid
                                                               // printf("cidsize[%d]=%ld \n", i*NV+j, 0);
                }
            }
        }
        cidSize[i] = mapSize;
    }

    AxiMap<int, DWIDTH> axi_cidSize(cidSize);

UPDATE_RENUMBER:
    for (int i = 0; i < numNV; i++) {
        tmpCid = cidPrev[i];
        for (int j = 0; j < NV; j++) {
#pragma HLS PIPELINE
#pragma HLS DEPENDENCE variable = cidPrev inter false
            if ((i * NV + j) < numVertex) {
                cidOld = tmpCid((j + 1) * NvBt - 1, j * NvBt);
                printf("cidOld[%d]=%ld cidNew[%d]=%ld \n", i * NV + j, cc.f, i * NV + j, ss.f);

                ap_uint<NvBt> tmp = axi_cidSize.rdi(cidOld);
                tmpNew((j + 1) * NvBt - 1, j * NvBt) = tmp;
            }
        }
        cidPrev[i] = tmpNew;
    }
}

template <int DWIDTH, int VERTEX>
void renumberClusters_ghost(int64_t numVertex,
                            int64_t& numClusters,
                            int64_t NV_l,
                            ap_uint<DWIDTH> cidPrev[VERTEX],
                            ap_uint<DWIDTH> cidSize[VERTEX]) {
#pragma HLS INLINE off

    long cnt = 0;
    long numGhost = NV_l;
    const int NvBt = 32;
    const int NV = DWIDTH / NvBt;
    const int numNV = (numVertex + (NV - 1)) / NV;

    ap_uint<NvBt> csize, cidOld, cidNew;
    ap_uint<DWIDTH> tmpSize, tmpCid;
    ap_uint<DWIDTH> mapSize, tmpNew;

    f_cast_<int, ap_uint<32> > ss, cc;

CSIZE_SCAN:
    for (int i = 0; i < numNV; i++) {
#pragma HLS PIPELINE
        tmpSize = cidSize[i];
        tmpCid = cidPrev[i];
        for (int j = 0; j < NV; j++) {
#pragma HLS UNROLL
            if ((i * NV + j) < numVertex) {
                csize = tmpSize((j + 1) * NvBt - 1, j * NvBt);
                cidOld = tmpCid((j + 1) * NvBt - 1, j * NvBt);
                cc.i = cidOld;
                ss.i = csize;
                if ((i * NV + j) < NV_l) {
                    if (csize > 0)
                        mapSize((j + 1) * NvBt - 1, j * NvBt) = cnt++; // cid > 0 , based cid=1
                    else
                        mapSize((j + 1) * NvBt - 1, j * NvBt) = 0; // cid is a invalid

                } else {
                    if (csize > 0)
                        mapSize((j + 1) * NvBt - 1, j * NvBt) = numGhost++;
                    else
                        mapSize((j + 1) * NvBt - 1, j * NvBt) = 0; // cid is a invalid
                }
            }
        }
        cidSize[i] = mapSize;
    }

    AxiMap<int, DWIDTH> axi_cidSize(cidSize);
    int64_t offset = NV_l - cnt;

UPDATE_RENUMBER:
    for (int i = 0; i < numNV; i++) {
        tmpCid = cidPrev[i];
        for (int j = 0; j < NV; j++) {
#pragma HLS PIPELINE
#pragma HLS DEPENDENCE variable = cidPrev inter false
            if ((i * NV + j) < numVertex) {
                cidOld = tmpCid((j + 1) * NvBt - 1, j * NvBt);
                ap_uint<NvBt> tmp = axi_cidSize.rdi(cidOld);

                if (cidOld < NV_l) {
                    tmpNew((j + 1) * NvBt - 1, j * NvBt) = tmp;
                    cc.i = tmp;
                } else {
                    tmpNew((j + 1) * NvBt - 1, j * NvBt) = tmp - offset;
                    ss.i = tmp;
                    cc.i = tmp - offset;
                }
            }
        }
        cidPrev[i] = tmpNew;
    }
    numClusters = cnt + numGhost - NV_l;
}

} // graph
} // xf
#endif // _KERNEL_RENUMBER_HPP_
