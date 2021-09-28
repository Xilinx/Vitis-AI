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

#include "xf_data_analytics/common/stream_local_processing.hpp"
#include "xf_data_analytics/common/table_sample.hpp"
#include "xf_data_analytics/classification/logisticRegression.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

const int K = 4;
const int Kmax = 20;
const int D = 8;
const int Dmax = 20;

double funca(double op1, double op2) {
    return op1 * op2;
}

void funcb(double& reg, double op) {
    reg += op;
}

double funcC(double op) {
    return op;
}

void dut(double inputW[K][D][Kmax * Dmax],
         double inputI[K][Kmax],
         const ap_uint<32> cols,
         const ap_uint<32> classNum,
         hls::stream<double> opStrm[D],
         hls::stream<bool>& eOpStrm,
         hls::stream<ap_uint<32> >& retStrm,
         hls::stream<bool>& eRetStrm) {
    using namespace xf::data_analytics::common::internal;
    xf::data_analytics::classification::logisticRegressionPredict<double, D, Dmax, K, Kmax, BRAM, BRAM> processor;
    // sl2<double, D, Dmax, K, Kmax, &funcA, &funcB, &funcC, 6, BRAM, URAM> processor;
    processor.setWeight(inputW, cols, classNum);
    processor.setIntercept(inputI, classNum);
    processor.predict(opStrm, eOpStrm, cols, classNum, retStrm, eRetStrm);
}

#ifndef __SYNTHESIS__
int main() {
    xf::data_analytics::common::internal::MT19937 rng;
    rng.seedInitialization(42);
    const int rows = 200; // 2000
    const int cols = 99;  // 13
    const int ws = 17;    // 5
    const int cols_batch = (cols + D - 1) / D;

    double RawWeight[ws][cols];
    for (int i = 0; i < ws; i++) {
        for (int j = 0; j < cols; j++) {
            RawWeight[i][j] = rng.next();
            RawWeight[i][j] -= 0.55;
        }
    }

    double inputW[K][D][Kmax * Dmax];
    for (int i = 0; i < ws; i++) {
        for (int j = 0; j < cols; j++) {
            inputW[i % K][j % D][(i / K) * cols_batch + j / D] = RawWeight[i][j];
        }
    }

    double RawIntercept[ws];
    for (int i = 0; i < ws; i++) {
        RawIntercept[i] = rng.next();
        RawIntercept[i] -= 0.5;
    }

    double inputI[K][Kmax];
    for (int i = 0; i < ws; i++) {
        inputI[i % K][i / K] = RawIntercept[i];
    }

    double goldenM[rows * ws];
    int goldenK[rows];

    hls::stream<double> opStrm[D];
    hls::stream<bool> eOpStrm;
    hls::stream<ap_uint<32> > retStrm;
    hls::stream<bool> eRetStrm;

    for (int i = 0; i < rows; i++) {
        double tmp[cols];
        for (int j = 0; j < cols; j += D) {
            for (int k = 0; k < D; k++) {
                double tmpr = rng.next();
                // tmpr -= 0.5;
                opStrm[k].write(tmpr);
                int p = j + k;
                if (p < cols) {
                    tmp[p] = tmpr;
                }
            }
            eOpStrm.write(false);
        }
        for (int j = 0; j < ws; j++) {
            double res = RawIntercept[j];
            for (int k = 0; k < cols; k++) {
                res += tmp[k] * RawWeight[j][k];
            }
            goldenM[i * ws + j] = res;
        }
    }
    eOpStrm.write(true);

    for (int i = 0; i < rows; i++) {
        double margin = 0;
        int index = 0;
        for (int j = 0; j < ws; j++) {
            if (goldenM[i * ws + j] > margin) {
                margin = goldenM[i * ws + j];
                index = j;
            }
        }
        if (margin > 0) {
            goldenK[i] = index + 1;
        } else {
            goldenK[i] = 0;
        }
    }

    ap_uint<32> classNum = ws + 1;
    dut(inputW, inputI, cols, classNum, opStrm, eOpStrm, retStrm, eRetStrm);

    bool tested = true;
    for (int i = 0; i < rows; i++) {
        ap_uint<32> index = retStrm.read();
        eRetStrm.read();
        if (index != goldenK[i]) {
            tested = false;
            std::cout << "error! golden: " << goldenK[i] << "  act: " << index << std::endl;
        }
    }
    eRetStrm.read();

    if (tested) {
        return 0;
    } else {
        return 1;
    }
}
#endif
