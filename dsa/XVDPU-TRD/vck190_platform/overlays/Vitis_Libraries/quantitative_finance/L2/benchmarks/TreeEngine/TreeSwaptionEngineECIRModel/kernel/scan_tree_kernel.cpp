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

#include "tree_engine_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

static void treeKernel(hls::stream<int>& typeStrm,
                       hls::stream<DT>& fixedRateStrm,
                       hls::stream<int>& timestepStrm,
                       hls::stream<int>& initSizeStrm,
                       hls::stream<DT>& aStrm,
                       hls::stream<DT>& sigmaStrm,
                       hls::stream<DT>& flatRateStrm,
                       hls::stream<DT>& x0Strm,
                       hls::stream<DT>& nominalStrm,
                       hls::stream<DT>& spreadStrm,
                       hls::stream<DT>& initTimeStrm,
                       hls::stream<int>& exerciseCntStrm,
                       hls::stream<int>& floatingCntStrm,
                       hls::stream<int>& fixedCntStrm,
                       hls::stream<DT>& NPVStrm) {
    int type = typeStrm.read();
    DT fixedRate = fixedRateStrm.read();
    int timestep = timestepStrm.read();
    int initSize = initSizeStrm.read();
    DT a = aStrm.read();
    DT sigma = sigmaStrm.read();
    DT flatRate = flatRateStrm.read();
    DT x0 = x0Strm.read();
    DT nominal = nominalStrm.read();
    DT spread = spreadStrm.read();
    DT NPV[1];

    DT initTime[InitTimeLen];
    int exerciseCnt[ExerciseLen];
    int floatingCnt[FloatingLen];
    int fixedCnt[FixedLen];

    for (int i = 0; i < initSize; i++) initTime[i] = initTimeStrm.read();
    for (int i = 0; i < ExerciseLen; i++) exerciseCnt[i] = exerciseCntStrm.read();
    for (int i = 0; i < FloatingLen; i++) floatingCnt[i] = floatingCntStrm.read();
    for (int i = 0; i < FixedLen; i++) fixedCnt[i] = fixedCntStrm.read();

    Model model;
    model.initialization(flatRate, spread);
    DT process[4] = {a, sigma, 0.36440284225769976, 0.20683023322988522};

    treeSwaptionEngine<DT, Model, Process, DIM, LEN, LEN2>(model, process, type, fixedRate, timestep, initTime,
                                                           initSize, exerciseCnt, floatingCnt, fixedCnt, flatRate,
                                                           nominal, x0, spread, NPV);

#ifndef __SYNTHESIS__
    std::cout << "type=" << type << ",NPV=" << NPV[0] << std::endl;
#endif
    NPVStrm.write(NPV[0]);
}

static void scanDataDist(int len,
                         ScanInputParam0 inputParam0[1],
                         ScanInputParam1 inputParam1[1],
                         hls::stream<int> typeStrm[K],
                         hls::stream<DT> fixedRateStrm[K],
                         hls::stream<int> timestepStrm[K],
                         hls::stream<int> initSizeStrm[K],
                         hls::stream<DT> aStrm[K],
                         hls::stream<DT> sigmaStrm[K],
                         hls::stream<DT> flatRateStrm[K],
                         hls::stream<DT> x0Strm[K],
                         hls::stream<DT> nominalStrm[K],
                         hls::stream<DT> spreadStrm[K],
                         hls::stream<DT> initTimeStrm[K],
                         hls::stream<int> exerciseCntStrm[K],
                         hls::stream<int> floatingCntStrm[K],
                         hls::stream<int> fixedCntStrm[K]) {
    for (int i = 0; i < len; i++) {
        typeStrm[i % K].write(inputParam1[0].type);
        fixedRateStrm[i % K].write(inputParam1[0].fixedRate);
        timestepStrm[i % K].write(inputParam1[0].timestep);
        initSizeStrm[i % K].write(inputParam1[0].initSize);
        aStrm[i % K].write(inputParam1[0].a);
        sigmaStrm[i % K].write(inputParam1[0].sigma);
        flatRateStrm[i % K].write(inputParam1[0].flatRate);
        x0Strm[i % K].write(inputParam0[0].x0);
        nominalStrm[i % K].write(inputParam0[0].nominal);
        spreadStrm[i % K].write(inputParam0[0].spread);
        for (int j = 0; j < inputParam1[0].initSize; j++) {
            initTimeStrm[i % K].write(inputParam0[0].initTime[j]);
        }
        for (int j = 0; j < ExerciseLen; j++) {
            exerciseCntStrm[i % K].write(inputParam1[0].exerciseCnt[j]);
        }
        for (int j = 0; j < FloatingLen; j++) {
            floatingCntStrm[i % K].write(inputParam1[0].floatingCnt[j]);
        }
        for (int j = 0; j < FixedLen; j++) {
            fixedCntStrm[i % K].write(inputParam1[0].fixedCnt[j]);
        }
    }
}

static void treeKernelWrapper(hls::stream<int> typeStrm[K],
                              hls::stream<DT> fixedRateStrm[K],
                              hls::stream<int> timestepStrm[K],
                              hls::stream<int> initSizeStrm[K],
                              hls::stream<DT> aStrm[K],
                              hls::stream<DT> sigmaStrm[K],
                              hls::stream<DT> flatRateStrm[K],
                              hls::stream<DT> x0Strm[K],
                              hls::stream<DT> nominalStrm[K],
                              hls::stream<DT> spreadStrm[K],
                              hls::stream<DT> initTimeStrm[K],
                              hls::stream<int> exerciseCntStrm[K],
                              hls::stream<int> floatingCntStrm[K],
                              hls::stream<int> fixedCntStrm[K],
                              hls::stream<DT> NPVStrm[K]) {
#pragma HLS dataflow
    for (int k = 0; k < K; k++) {
#pragma HLS unroll
        treeKernel(typeStrm[k], fixedRateStrm[k], timestepStrm[k], initSizeStrm[k], aStrm[k], sigmaStrm[k],
                   flatRateStrm[k], x0Strm[k], nominalStrm[k], spreadStrm[k], initTimeStrm[k], exerciseCntStrm[k],
                   floatingCntStrm[k], fixedCntStrm[k], NPVStrm[k]);
    }
}

static void treeKernelLoop(int len,
                           hls::stream<int> typeStrm[K],
                           hls::stream<DT> fixedRateStrm[K],
                           hls::stream<int> timestepStrm[K],
                           hls::stream<int> initSizeStrm[K],
                           hls::stream<DT> aStrm[K],
                           hls::stream<DT> sigmaStrm[K],
                           hls::stream<DT> flatRateStrm[K],
                           hls::stream<DT> x0Strm[K],
                           hls::stream<DT> nominalStrm[K],
                           hls::stream<DT> spreadStrm[K],
                           hls::stream<DT> initTimeStrm[K],
                           hls::stream<int> exerciseCntStrm[K],
                           hls::stream<int> floatingCntStrm[K],
                           hls::stream<int> fixedCntStrm[K],
                           hls::stream<DT> NPVStrm[K]) {
    for (int i = 0; i < len / K; i++) {
        treeKernelWrapper(typeStrm, fixedRateStrm, timestepStrm, initSizeStrm, aStrm, sigmaStrm, flatRateStrm, x0Strm,
                          nominalStrm, spreadStrm, initTimeStrm, exerciseCntStrm, floatingCntStrm, fixedCntStrm,
                          NPVStrm);
    }
}

static void scanWrapper(int len,
                        ScanInputParam0 inputParam0[1],
                        ScanInputParam1 inputParam1[1],
                        hls::stream<DT> NPVStrm[K]) {
#pragma HLS dataflow
    hls::stream<int> typeStrm[K];
    hls::stream<DT> fixedRateStrm[K];
    hls::stream<int> timestepStrm[K];
    hls::stream<int> initSizeStrm[K];
    hls::stream<DT> aStrm[K];
    hls::stream<DT> sigmaStrm[K];
    hls::stream<DT> flatRateStrm[K];
    hls::stream<DT> x0Strm[K];
    hls::stream<DT> nominalStrm[K];
    hls::stream<DT> spreadStrm[K];
    hls::stream<DT> initTimeStrm[K];
    hls::stream<int> exerciseCntStrm[K];
    hls::stream<int> floatingCntStrm[K];
    hls::stream<int> fixedCntStrm[K];
#pragma HLS stream variable = typeStrm depth = 10
#pragma HLS stream variable = fixedRateStrm depth = 10
#pragma HLS stream variable = timestepStrm depth = 10
#pragma HLS stream variable = initSizeStrm depth = 10
#pragma HLS stream variable = aStrm depth = 10
#pragma HLS stream variable = sigmaStrm depth = 10
#pragma HLS stream variable = flatRateStrm depth = 10
#pragma HLS stream variable = x0Strm depth = 10
#pragma HLS stream variable = nominalStrm depth = 10
#pragma HLS stream variable = spreadStrm depth = 10
#pragma HLS stream variable = initTimeStrm depth = 120
#pragma HLS stream variable = exerciseCntStrm depth = 50
#pragma HLS stream variable = floatingCntStrm depth = 100
#pragma HLS stream variable = fixedCntStrm depth = 50

#pragma HLS RESOURCE variable = typeStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = fixedRateStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = timestepStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = initSizeStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = aStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = sigmaStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = flatRateStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = x0Strm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = nominalStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = spreadStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = initTimeStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = exerciseCntStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = floatingCntStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = fixedCntStrm core = FIFO_LUTRAM
    scanDataDist(len, inputParam0, inputParam1, typeStrm, fixedRateStrm, timestepStrm, initSizeStrm, aStrm, sigmaStrm,
                 flatRateStrm, x0Strm, nominalStrm, spreadStrm, initTimeStrm, exerciseCntStrm, floatingCntStrm,
                 fixedCntStrm);
    treeKernelLoop(len, typeStrm, fixedRateStrm, timestepStrm, initSizeStrm, aStrm, sigmaStrm, flatRateStrm, x0Strm,
                   nominalStrm, spreadStrm, initTimeStrm, exerciseCntStrm, floatingCntStrm, fixedCntStrm, NPVStrm);
}

extern "C" void scanTreeKernel(int len, ScanInputParam0 inputParam0[1], ScanInputParam1 inputParam1[1], DT NPV[N]) {
#pragma HLS INTERFACE m_axi port = inputParam0 latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 64 max_read_burst_length = 64 bundle = gmem0 offset = slave
#pragma HLS INTERFACE m_axi port = inputParam1 latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 64 max_read_burst_length = 64 bundle = gmem1 offset = slave
#pragma HLS INTERFACE m_axi port = NPV latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 64 max_read_burst_length = 64 bundle = gmem2 offset = slave

#pragma HLS INTERFACE s_axilite port = len bundle = control
#pragma HLS INTERFACE s_axilite port = inputParam0 bundle = control
#pragma HLS INTERFACE s_axilite port = inputParam1 bundle = control
#pragma HLS INTERFACE s_axilite port = NPV bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS data_pack variable = inputParam0
#pragma HLS data_pack variable = inputParam1

    hls::stream<DT> NPVStrm[K];
    hls::stream<bool> NPVFlagStrm;
#pragma HLS stream variable = NPVStrm depth = 100
#pragma HLS stream variable = NPVFlagStrm depth = 400
#pragma HLS RESOURCE variable = NPVStrm core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = NPVFlagStrm core = FIFO_LUTRAM

    scanWrapper(len, inputParam0, inputParam1, NPVStrm);

    for (int i = 0; i < len; i++) {
        DT data = NPVStrm[i % K].read();
        NPV[i] = data;
    }
}
