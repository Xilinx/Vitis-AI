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

#include <math.h>
#include <iostream>
#include <hls_stream.h>
#include <iomanip>
#include <ap_int.h>
#include <stdint.h>
#define DT double
#define DTLEN 64
#define N 66
#define M 1280
#define TI 2
#define TO 2

void dut(int rows,
         int cols,
         hls::stream<ap_uint<DTLEN * TI> >& inMatStrm,
         hls::stream<ap_uint<DTLEN * TI> >& outCovStrm);

void covCalc(int rows, int cols, DT inMat[N][M], DT outCov[N][N]) {
    static DT tmpMat[N][M];
    for (int i = 0; i < rows; i++) {
        DT ave = 0.0;
        for (int j = 0; j < cols; j++) {
            ave += inMat[i][j] / cols;
        }
        for (int j = 0; j < cols; j++) {
            tmpMat[i][j] = inMat[i][j] - ave;
        }
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j <= i; j++) {
            DT tmp = 0.0;
            for (int k = 0; k < cols; k++) {
                tmp += tmpMat[i][k] * tmpMat[j][k];
            }
            outCov[i][j] = tmp / (cols - 1);
            outCov[j][i] = tmp / (cols - 1);
        }
    }
}

union DTConvert {
    uint64_t dt0;
    DT dt1;
};

int main() {
    int nerr = 0;
    DT err = 1e-12;
    const int rows = 6;
    const int cols = 126;

    static DT inMat[N][M];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            inMat[i][j] = (rand() % 100) * 0.1;
        }
    }
    hls::stream<ap_uint<DTLEN * TI> > inMatStrm;
    ap_uint<DTLEN * TI> inTmp;
    DTConvert dtc;
    hls::stream<ap_uint<DTLEN * TO> > outCovStrm;

    for (int i = 0; i * TI < rows * cols; i++) {
        DTConvert dtc;
        ap_uint<DTLEN * TI> inTmp;
        for (int t = 0; t < TI; t++) {
            if (TI * i + t < rows * cols) {
                dtc.dt1 = inMat[(i * TI + t) % rows][(i * TI + t) / rows];
                inTmp.range(DTLEN - 1 + t * DTLEN, DTLEN * t) = dtc.dt0;
            }
        }
        inMatStrm.write(inTmp);
    }
    static DT outMat[N][N];
    static DT goldenMat[N][N];
    covCalc(rows, cols, inMat, goldenMat);
    dut(rows, cols, inMatStrm, outCovStrm);
    for (int i = 0; i * TO < rows * rows; i++) {
        ap_uint<DTLEN* TO> tmp = outCovStrm.read();
        DTConvert dtc;
        for (int t = 0; t < TO; t++) {
            if (i * TO + t < rows * rows) {
                dtc.dt0 = tmp.range(DTLEN * t + DTLEN - 1, DTLEN * t);
                outMat[(i * TO + t) / rows][(i * TO + t) % rows] = dtc.dt1;
            }
        }
    }

    for (int i = 0; i < rows; i++) {
        std::cout << std::endl;
        for (int j = 0; j < rows; j++) {
            DT tmp = outMat[i][j];
            std::cout << tmp << ", ";
            if (std::abs(tmp - goldenMat[i][j]) > err) {
                std::cout << std::setprecision(14) << "outMat[" << i << "][" << j << "]=" << tmp << ", goldenMat[" << i
                          << "][" << j << "]=" << goldenMat[i][j] << ",diff=" << std::abs(tmp - goldenMat[i][j])
                          << std::endl;
                nerr++;
            }
        }
    }
    std::cout << nerr << std::endl;
    return nerr;
}
