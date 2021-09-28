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
 * @file covariance.hpp
 * @brief This file include the function covCoreStrm and the function covCoreMatrix
 *
 */

#ifndef __XF_FINTECH_COVARIANCE_HPP_
#define __XF_FINTECH_COVARIANCE_HPP_

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include <stdint.h>

#ifndef __SYNTHESIS__
#include "iostream"
using namespace std;
#endif

namespace xf {

namespace fintech {

namespace internal {

union DTConvert64 {
    uint64_t dt0;
    double dt1;
};

union DTConvert32 {
    uint32_t dt0;
    float dt1;
};

template <typename DT>
DT addTree4(DT values4[4]) {
#pragma HLS inline
    DT values2_0 = values4[0] + values4[2];
    DT values2_1 = values4[1] + values4[3];
    DT value = values2_0 + values2_1;
    return value;
}

template <typename DT>
DT addTree8(DT values8[8]) {
#pragma HLS inline
    DT values4[4];
#pragma HLS array_partition variable = values4 dim = 0
    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
        values4[i] = values8[i] + values8[i + 4];
    }
    DT values2_0 = values4[0] + values4[2];
    DT values2_1 = values4[1] + values4[3];
    DT value = values2_0 + values2_1;
    return value;
}

template <typename DT>
DT addTree16(DT values16[16]) {
#pragma HLS inline
    DT values8[8];
#pragma HLS array_partition variable = values8 dim = 0
    for (int k = 0; k < 8; k++) {
#pragma HLS unroll
        values8[k] = values16[k] + values16[k + 8];
    }
    DT values4[4];
#pragma HLS array_partition variable = values4 dim = 0
    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
        values4[i] = values8[i] + values8[i + 4];
    }
    DT values2_0 = values4[0] + values4[2];
    DT values2_1 = values4[1] + values4[3];
    DT value = values2_0 + values2_1;
    return value;
}

template <typename DT>
DT addTree64(DT values64[64]) {
#pragma HLS inline
    DT values32[32];
#pragma HLS array_partition variable = values32 dim = 0
    for (int k = 0; k < 32; k++) {
        values32[k] = values64[k] + values64[k + 32];
    }

    DT values16[16];
#pragma HLS array_partition variable = values16 dim = 0
    for (int k = 0; k < 16; k++) {
        values16[k] = values32[k] + values32[k + 16];
    }

    DT values8[8];
#pragma HLS array_partition variable = values8 dim = 0
    for (int k = 0; k < 8; k++) {
#pragma HLS unroll
        values8[k] = values16[k] + values16[k + 8];
    }

    DT values4[4];
#pragma HLS array_partition variable = values4 dim = 0
    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
        values4[i] = values8[i] + values8[i + 4];
    }
    DT values2_0 = values4[0] + values4[2];
    DT values2_1 = values4[1] + values4[3];
    DT value = values2_0 + values2_1;
    return value;
}

template <typename DT, int DTLEN, int N, int M, int T>
void strm2RAM(int rows, int cols, hls::stream<ap_uint<DTLEN * T> >& inMatStrm, DT matRAM[N][M]) {
    for (int i = 0; i * T < rows * cols; i++) {
#pragma HLS pipeline
#pragma HLS dependence variable = matRAM false inter
        ap_uint<DTLEN* T> tmp = inMatStrm.read();
        int k = i * T;
        if (DTLEN == 64) {
            DTConvert64 dtc;
            for (int t = 0; t < T; t++) {
                if (k + t < rows * cols) {
                    dtc.dt0 = tmp.range(DTLEN * t + DTLEN - 1, DTLEN * t);
                    matRAM[(i * T + t) % rows][(i * T + t) / rows] = dtc.dt1;
                }
            }
        } else {
            DTConvert32 dtc;
            for (int t = 0; t < T; t++) {
                if (k + t < rows * cols) {
                    dtc.dt0 = tmp.range(DTLEN * t + DTLEN - 1, DTLEN * t);
                    matRAM[(i * T + t) % rows][(i * T + t) / rows] = dtc.dt1;
                }
            }
        }
    }
}

template <typename DT, int DTLEN, int N, int T>
void RAM2Strm(int rows, DT CovMatrix[N][N], hls::stream<ap_uint<DTLEN * T> >& outCovStrm) {
    for (int i = 0; i * T < rows * rows; i++) {
#pragma HLS pipeline
        ap_uint<DTLEN * T> tmp;
        if (DTLEN == 64) {
            DTConvert64 dtc;
            for (int t = 0; t < T; t++) {
                if (i * T + t < rows * rows) {
                    dtc.dt1 = CovMatrix[(i * T + t) / rows][(i * T + t) % rows];
                    tmp.range(DTLEN * t + DTLEN - 1, DTLEN * t) = dtc.dt0;
                }
            }
        } else {
            DTConvert32 dtc;
            for (int t = 0; t < T; t++) {
                if (i * T + t < rows * rows) {
                    dtc.dt1 = CovMatrix[(i * T + t) / rows][(i * T + t) % rows];
                    tmp.range(DTLEN * t + DTLEN - 1, DTLEN * t) = dtc.dt0;
                }
            }
        }
        outCovStrm.write(tmp);
    }
}

template <typename DT, int N, int M, int U, int V, int W>
void covCorePart1(int rows, int cols, int xcols, DT inMatrix[N][M], hls::stream<DT> values2Strm[V]) {
    DT values2[V][W];
    DT values3[V][U];
#pragma HLS array_partition variable = values2 dim = 1
#pragma HLS resource variable = values2 core = RAM_2P_LUTRAM
#pragma HLS array_partition variable = values3 dim = 0
    int k_cnt = (cols - 1) / U + 1;
    if (cols < U * W) k_cnt = W;
loop_c0:
    for (int i = 0; i < rows; i++) {
#pragma HLS loop_tripcount max = N min = N
    loop_c1:
        for (int j = 0; V * (j - 1) <= i; j++) {
#pragma HLS loop_tripcount max = N min = N
#pragma HLS loop_flatten off
        loop_c2:
            for (int k = 0; k < k_cnt; k++) {
#pragma HLS loop_tripcount max = M min = M
#pragma HLS pipeline
                for (int u = 0; u < U; u++) {
                    if (k * U + u < cols && j <= i) {
                        DT matValue = inMatrix[i][k * U + u];

                        for (int v = 0; v < V; v++) {
                            if (V * j + v <= i) {
                                values3[v][u] = matValue * inMatrix[V * j + v][k * U + u];
                            } else {
                                values3[v][u] = 0.0;
                            }
                        }
                    } else {
                        for (int v = 0; v < V; v++) {
                            values3[v][u] = 0.0;
                        }
                    }
                }
                DT tmp[V];
                for (int v = 0; v < V; v++) {
                    if (U == 1) tmp[v] = values3[v][0];
                    if (U == 2) tmp[v] = values3[v][0] + values3[v][1];
                    if (U == 4) tmp[v] = addTree4(values3[v]);
                    if (U == 8) tmp[v] = addTree8(values3[v]);
                    if (U == 16) tmp[v] = addTree16(values3[v]);
                    if (k < W) {
                        if (j > 0) values2Strm[v].write(values2[v][k]);
                        values2[v][k % W] = tmp[v];
                    } else {
                        values2[v][k % W] += tmp[v];
                    }
                }
            }
        }
    }
}

template <typename DT, int N, int V, int W>
void covCorePart2(int rows, int cols, hls::stream<DT> values2Strm[V], DT outCovMatrix[N][N]) {
    DT d1_cols = 1.0 / (cols - 1);
    for (int i = 0; i < rows; i++) {
#pragma HLS loop_tripcount max = N min = N
        for (int j = 0; V * j <= i; j++) {
#pragma HLS loop_tripcount max = N min = N
#pragma HLS pipeline
            for (int v = 0; v < V; v++) {
                DT cov = 0;
                for (int k = 0; k < W; k++) {
                    cov += values2Strm[v].read();
                }
                // cov = addTree16(values2[v]);
                if (V * j + v <= i) {
                    outCovMatrix[i][V * j + v] = cov * d1_cols;
                    outCovMatrix[V * j + v][i] = cov * d1_cols;
                }
            }
        }
    }
}

template <typename DT, int N, int M, int U, int V, int W>
void covCoreWrapper(int rows, int cols, DT inMatrix[N][M], DT outCovMatrix[N][N]) {
    hls::stream<DT> values2Strm[V];
#pragma HLS stream variable = values2Strm depth = 32
#pragma HLS dataflow
    int xcols = (cols - 1) / U + 1;
    covCorePart1<DT, N, M, U, V, W>(rows, cols, xcols, inMatrix, values2Strm);
    covCorePart2<DT, N, V, W>(rows, cols, values2Strm, outCovMatrix);
}

template <typename DT, int N, int M, int U, int V, int W>
void aveImpl(int rows, int cols, DT inMatrix[N][M]) {
    DT d_cols = 1.0 / cols;
    int xcols = (cols - 1) / U + 1;
    DT ave[V];
    DT values[V][W];
    DT values1[V][U];
#pragma HLS array_partition variable = ave dim = 0
#pragma HLS array_partition variable = values dim = 1
#pragma HLS resource variable = values core = RAM_2P_LUTRAM
    for (int w = 0; w < W; w++) {
#pragma HLS pipeline
        for (int v = 0; v < V; v++) {
            values[v][w] = 0.0;
        }
    }
loop_ave:
    for (int i = 0; V * i < rows; i++) {
#pragma HLS loop_tripcount max = N min = N
    loop_a1:
        for (int j = 0; j < xcols; j++) {
#pragma HLS loop_tripcount max = M min = M
#pragma HLS pipeline
            for (int u = 0; u < U; u++) {
                if (j * U + u < cols) {
                    for (int v = 0; v < V; v++) {
                        if (V * i + v < rows)
                            values1[v][u] = inMatrix[i * V + v][j * U + u] * d_cols;
                        else
                            values1[v][u] = 0.0;
                    }
                } else {
                    for (int v = 0; v < V; v++) {
                        values1[v][u] = 0.0;
                    }
                }
            }
            DT tmp[V];
            for (int v = 0; v < V; v++) {
                if (U == 1) tmp[v] = values1[v][0];
                if (U == 2) tmp[v] = values1[v][0] + values1[v][1];
                if (U == 4) tmp[v] = addTree4(values1[v]);
                if (U == 8) tmp[v] = addTree8(values1[v]);
                if (U == 16) tmp[v] = addTree16(values1[v]);
                if (j < W)
                    values[v][j % W] = tmp[v];
                else
                    values[v][j % W] += tmp[v];
            }
        }
        for (int v = 0; v < V; v++) {
#pragma HLS unroll
            ave[v] = addTree16(values[v]);
        }
    loop_a2:
        for (int j = 0; j < xcols; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount max = M min = M
            for (int u = 0; u < U; u++) {
                if (j * U + u < cols) {
                    for (int v = 0; v < V; v++) {
                        if (V * i + v < rows) inMatrix[i * V + v][j * U + u] -= ave[v];
                    }
                } else {
                    // break;
                }
            }
        }
    }
}
} // internal

/**
 * @brief covCoreMatrix calculate the covariance of the input matrix.
 *
 * @tparam DT data type supported include float and double
 * @tparam N maximum supported row
 * @tparam M maximum supported column
 * @tparam U unroll the 1-d inMatrix to improve throughput, support 4, 8, 16
 * @tparam V unroll the 2-d inMatrix to improve throughput, support 1, 2, 4, 8
 *
 * @param rows actual row number
 * @param cols actual column number
 * @param inMatrix input cols x rows matrix
 * @param outCovMatrix output rows x rows covariance matrix
 *
 */
template <typename DT, int N, int M, int U, int V>
void covCoreMatrix(int rows, int cols, DT inMatrix[N][M], DT outCovMatrix[N][N]) {
    const int W = 16;
    internal::aveImpl<DT, N, M, U, V, W>(rows, cols, inMatrix);
    internal::covCoreWrapper<DT, N, M, U, V, W>(rows, cols, inMatrix, outCovMatrix);
}

/**
 * @brief covCoreStrm calculate the covariance of the input matrix, the input matrix input in the order of the columns
 * by stream, the output covariance matrix output in the order of the rows by stream.
 *
 * @tparam DT data type supported include float and double
 * @tparam DTLEN length of DT
 * @tparam N maximum supported row
 * @tparam M maximum supported column
 * @tparam TI the bit-width of input stream is TI * DTLEN
 * @tparam TO the bit-width of output stream is TO * DTLEN
 *
 * @param rows actual row number
 * @param cols actual column number
 * @param inMatStrm according to stream way to input cols x rows matrix in the order of the columns
 * @param outCovStrm according to stream way to output rows x rows covariance matrix in the order of the rows
 *
 */
template <typename DT, int DTLEN, int N, int M, int TI, int TO>
void covCoreStrm(int rows,
                 int cols,
                 hls::stream<ap_uint<DTLEN * TI> >& inMatStrm,
                 hls::stream<ap_uint<DTLEN * TO> >& outCovStrm) {
    const int U = 8; // M, 4, 8, 16
    const int V = 4; // N, 1, 2, 4, 8
#ifndef __SYNTHESIS__
    static DT matRAM[N][M];
    static DT covMatRAM[N][N];
#else
    DT matRAM[N][M];
    DT covMatRAM[N][N];
#endif
#pragma HLS array_partition variable = matRAM dim = 1 cyclic factor = V
#pragma HLS array_partition variable = matRAM dim = 2 cyclic factor = U
#pragma HLS array_partition variable = covMatRAM cyclic factor = 2 dim = 1
#pragma HLS array_partition variable = covMatRAM cyclic factor = TO dim = 2

    internal::strm2RAM<DT, DTLEN, N, M, TI>(rows, cols, inMatStrm, matRAM);
    covCoreMatrix<DT, N, M, U, V>(rows, cols, matRAM, covMatRAM);
    internal::RAM2Strm<DT, DTLEN, N, TO>(rows, covMatRAM, outCovStrm);
}

} // fintech
} // xf
#endif //__XF_FINTECH_COVARIANCE_HPP_
