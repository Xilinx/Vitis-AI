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
 * @file corrand.hpp
 * @brief This file contains the implementation of correlated random number
 * generator.
 */
#ifndef XF_FINTECH_CORRAND_H
#define XF_FINTECH_CORRAND_H
#include "ap_int.h"
#include "hls_stream.h"
#include "xf_fintech/utils.hpp"
namespace xf {
namespace fintech {
namespace internal {
template <typename DT, int ASSETS>
void martixMul(unsigned int loopNm,
               hls::stream<DT>& inStrm,
               hls::stream<DT> outStrm[8],
               DT corrMatrix[ASSETS][ASSETS]) {
    DT buff[ASSETS][8];
#pragma HLS array_partition variable = buff dim = 1
    // why we cannot add this pragma
    //#pragma HLS array_partition variable=buff dim=0
    ap_uint<3> cnt = 0;
    for (int l = 0; l < loopNm; ++l) {
#pragma HLS loop_tripcount min = 1024 max = 1024
        for (int i = 0; i < ASSETS; ++i) {
#pragma HLS pipeline II = 1
            DT dw = inStrm.read();
            DT outB[ASSETS];
#pragma HLS array_partition variable = outB
            for (int j = 0; j < ASSETS; ++j) {
#pragma HLS unroll
                DT nDw;
#pragma HLS resource variable = nDw core = DMul_meddsp
                nDw = corrMatrix[j][i] * dw;
                DT pre = buff[j][cnt];
                DT oldD;
                if (i == 0) {
                    oldD = 0;
                } else {
                    oldD = pre;
                }
                DT newD;
#pragma HLS resource variable = newD core = DAddSub_nodsp
                newD = oldD + nDw;
                buff[j][cnt] = newD;
                outB[j] = newD;
            }
            cnt++;
            for (int k = 0; k < 8; k++) {
#pragma HLS unroll
                DT out;
                if (k + i < ASSETS) {
                    out = outB[k + i];
                } else {
                    out = 0;
                }
                outStrm[k].write(out);
            }
        }
    }
}

template <typename DT, int ASSETS>
void mergeS(unsigned int loopNm, hls::stream<DT> inStrm[8], hls::stream<DT>& outStrm) {
    DT buff[8];
#pragma HLS array_partition variable = buff dim = 0
    for (int i = 0; i < loopNm; ++i) {
#pragma HLS loop_tripcount min = 1024 max = 1024
        for (int p = 0; p < ASSETS / 8; ++p) {
#pragma HLS pipeline II = 1
            DT in[8][8];
#pragma HLS array_partition variable = in dim = 0
            DT out[8];
#pragma HLS array_partition variable = out dim = 0
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 8; k++) {
#pragma HLS unroll
                    in[k][j] = inStrm[k].read();
                }
            }
            DT pre[8];
#pragma HLS array_partition variable = pre dim = 0
            for (int k = 0; k < 8; k++) {
#pragma HLS unroll
                if (p == 0) {
                    pre[k] = 0;
                } else {
                    pre[k] = buff[0];
                }
            }
            out[0] = pre[0] + in[0][0];
            out[1] = pre[1] + in[0][1] + in[1][0];
            out[2] = pre[2] + in[0][2] + in[1][1] + in[2][0];
            out[3] = pre[3] + in[0][3] + in[1][2] + in[2][1] + in[3][0];
            out[4] = pre[4] + in[0][4] + in[1][3] + in[2][2] + in[3][1] + in[4][0];
            out[5] = pre[5] + in[0][5] + in[1][4] + in[2][3] + in[3][2] + in[4][1] + in[5][0];
            out[6] = pre[6] + in[0][6] + in[1][5] + in[2][4] + in[3][3] + in[4][2] + in[5][1] + in[6][0];
            out[7] = in[0][7] + in[1][6] + in[2][5] + in[3][4] + in[4][3] + in[5][2] + in[6][1] + in[7][0];

            buff[0] = in[1][7] + in[2][6] + in[3][5] + in[4][4] + in[5][3] + in[6][2] + in[7][1];
            buff[1] = in[2][7] + in[3][6] + in[4][5] + in[5][4] + in[6][3] + in[7][2];
            buff[2] = in[3][7] + in[4][6] + in[5][5] + in[6][4] + in[7][3];
            buff[3] = in[4][7] + in[5][6] + in[6][5] + in[7][4];
            buff[4] = in[5][7] + in[6][6] + in[7][5];
            buff[5] = in[6][7] + in[7][6];
            buff[6] = in[7][7];
            for (int k = 0; k < 8; k++) {
#pragma HLS unroll
                outStrm.write(out[k]);
            }
        }
    }
}

template <typename DT, int ASSETS>
void corrand(unsigned int loopNm, hls::stream<DT>& inStrm, hls::stream<DT>& outStrm, DT corrMatrix[ASSETS][ASSETS]) {
#pragma HLS dataflow
    hls::stream<DT> buffStrm[8];
#pragma HLS array_partition variable = buffStrm dim = 0
    martixMul(loopNm, inStrm, buffStrm, corrMatrix);
    mergeS<DT, ASSETS>(loopNm * ASSETS, buffStrm, outStrm);
}

/*
CORRAND_2 generates random number for MultiAssetEuropeanHestonEngine.
Option based on N underlying assets, 2 random numbers for each, 2*N in total.
2*N random numbers's correlation Matrix is
CORRAND_2 takes 2 independent random numbers in 1 cycle.
CORRAND_2 produce 2 correlated random number in 1 cycle, in the order of assets.
*/

template <typename DT, int PathNm, int ASSETS, int CFGNM>
class CORRAND_2 {
   public:
    bool firstrun;
    ap_uint<1> rounds;

    DT buff_0[ASSETS * 2][CFGNM][PathNm];
    DT buff_1[ASSETS * 2][CFGNM][PathNm];

    // configuration from  start
    DT corrMatrix[ASSETS * 2 + 1][CFGNM][ASSETS];

    CORRAND_2() {
#pragma HLS inline
#pragma HLS array_partition variable = buff_0 dim = 1
#pragma HLS array_partition variable = buff_1 dim = 1
//#pragma HLS resource variable=buff_0 core=XPM_MEMORY uram
//#pragma HLS resource variable=buff_1 core=XPM_MEMORY uram
#pragma HLS array_partition variable = &corrMatrix dim = 1
        //#pragma HLS resource variable=corrMatrix core=RAM_1P_LUTRAM
    }

    void init() {
        firstrun = true;
        rounds = 0;
    }

    void setup(DT inputMatrix[ASSETS * 2 + 1][ASSETS]) {
        for (int i = 0; i < ASSETS * 2 + 1; i++) {
            for (int j = 0; j < ASSETS; j++) {
#pragma HLS pipeline II = 1
                corrMatrix[i][j] = inputMatrix[i][j];
            }
        }
    }

    inline void _next_body(
        int t_itr, int a_itr, int p_itr, int c, ap_uint<8>& r_a, ap_uint<16>& r_p, DT z0, DT z1, DT& r_0, DT& r_1) {
#pragma HLS inline
        int L0 = ASSETS * 2 - a_itr;
        int L1 = a_itr;
        int L2 = L0 - 1;
        int O_A0 = 2 * a_itr;
        int O_A1 = O_A0 + 1;
        DT tmp_buff[ASSETS * 2];
#pragma HLS array_partition variable = tmp_buff dim = 0
        DT tmp_mul[ASSETS * 2 + 1];
#pragma HLS array_partition variable = tmp_mul dim = 0
        DT tmp_add[ASSETS * 2];
#pragma HLS array_partition variable = tmp_add dim = 0

        DT tmp_buff_dup[ASSETS * 2];
#pragma HLS array_partition variable = tmp_buff_dup dim = 0
        for (int k = 0; k < ASSETS * 2; k++) {
#pragma HLS unroll
            if (rounds == 0) {
                tmp_buff_dup[k] = buff_1[k][c][p_itr];
                tmp_buff[k] = buff_0[k][c][p_itr];
                // tmp_buff_dup[k] = buff_1[k][c][r_p];
            } else {
                tmp_buff_dup[k] = buff_0[k][c][p_itr];
                tmp_buff[k] = buff_1[k][c][p_itr];
                // tmp_buff_dup[k] = buff_0[k][c][r_p];
            }
        }

        if (!(firstrun && t_itr == 0)) {
            r_0 = tmp_buff_dup[O_A0];
            r_1 = tmp_buff_dup[O_A1];

            // r_0 = tmp_buff_dup[2*r_a];
            // r_1 = tmp_buff_dup[2*r_a+1];
            // outStrm[0].write(r_0);
            // outStrm[1].write(r_1);
        }
        // if(c == CFGNM - 1) {
        //    if(r_a == ASSETS - 1) {
        //        r_a = 0;
        //        if(r_p == PathNm - 1) r_p = 0;
        //        else r_p++;
        //    } else {
        //        r_a++;
        //    }
        //}

        for (int k = 0; k < ASSETS * 2 + 1; k++) {
#pragma HLS unroll
            DT rn_d;

            if (k < L0) {
                rn_d = z0;
            } else {
                rn_d = z1;
            }
            tmp_mul[k] = FPTwoMul(corrMatrix[k][c][a_itr], rn_d);
        }

        for (int k = 0; k < ASSETS * 2; k++) {
#pragma HLS unroll
            if (k < L1) {
                tmp_add[k] = 0;
            } else {
                tmp_add[k] = tmp_mul[k - L1];
            }
        }

        for (int k = 0; k < ASSETS * 2; k++) {
#pragma HLS unroll
            DT tmp_add_tmp = tmp_add[k];
            DT add_op;
            if (k < L2) {
                add_op = 0;
            } else {
                add_op = tmp_mul[k + 1];
            }
            tmp_add[k] = FPTwoAdd(tmp_add_tmp, add_op);
        }

        for (int k = 0; k < ASSETS * 2; k++) {
#pragma HLS unroll
            DT tmp_buff_tmp = 0; //= FPTwoAdd(tmp_buff[k], tmp_add[k]);
            if (a_itr == 0) {
                tmp_buff_tmp = tmp_add[k]; //= FPTwoAdd(tmp_buff[k], tmp_add[k]);
            } else {
                tmp_buff_tmp = FPTwoAdd(tmp_buff[k], tmp_add[k]);
            }
            if (rounds == 0) {
                buff_0[k][c][p_itr] = tmp_buff_tmp;
            } else {
                buff_1[k][c][p_itr] = tmp_buff_tmp;
            }
        }
        if (c == CFGNM - 1 && a_itr == ASSETS - 1 && p_itr == PathNm - 1) rounds++;
    }

    void corrPathCube(hls::stream<ap_uint<16> >& timestepStrm,
                      hls::stream<DT>& inStrm0,
                      hls::stream<DT>& inStrm1,
                      hls::stream<DT>& outStrm0,
                      hls::stream<DT>& outStrm1) {
#pragma HLS dependence variable = buff_0 inter false
#pragma HLS dependence variable = buff_1 inter false

        ap_uint<16> timesteps;
        timesteps = timestepStrm.read();

        if (firstrun) {
            timesteps += 1;
        }
        ap_uint<8> r_a = 0;
        ap_uint<16> r_p = 0;
    TIME_LOOP:
        for (int t_itr = 0; t_itr < timesteps; t_itr++) {
#pragma HLS loop_tripcount min = 10 max = 10
        ASSETS_LOOP:
            for (int a_itr = 0; a_itr < ASSETS; a_itr++) {
#pragma HLS loop_tripcount min = 5 max = 5
            PATH_LOOP:
                for (int p_itr = 0; p_itr < PathNm; p_itr++) {
#pragma HLS loop_tripcount min = 1024 max = 1024
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 2 max = 2
                    DT z0 = inStrm0.read();
                    DT z1 = inStrm1.read();

                    DT r_0, r_1;
                    _next_body(t_itr, a_itr, p_itr, 0, r_a, r_p, z0, z1, r_0, r_1);
                    if (!(firstrun && t_itr == 0)) {
                        outStrm0.write(r_0);
                        outStrm1.write(r_1);
                    }
                }
            }
        }
        firstrun = false;
    }

    void corrPathCube(hls::stream<ap_uint<16> >& timestepStrm,
                      hls::stream<DT>& inStrm0,
                      hls::stream<DT>& inStrm1,
                      hls::stream<DT>& outStrm0,
                      hls::stream<DT>& outStrm1,
                      hls::stream<DT>& outStrm2,
                      hls::stream<DT>& outStrm3) {
//#pragma HLS array_partition variable = buff_0 dim = 1
//#pragma HLS array_partition variable = buff_1 dim = 1
//#pragma HLS resource variable = buff_0 core = XPM_MEMORY uram
//#pragma HLS resource variable = buff_1 core = XPM_MEMORY uram
//#pragma HLS array_partition variable = &corrMatrix dim = 1
//#pragma HLS resource variable = corrMatrix core = RAM_1P_LUTRAM
#pragma HLS dependence variable = buff_0 inter false
#pragma HLS dependence variable = buff_1 inter false

        ap_uint<16> timesteps;
        timesteps = timestepStrm.read();

        if (firstrun) {
            timesteps += 1;
        }
        ap_uint<8> r_a = 0;
        ap_uint<16> r_p = 0;
    TIME_LOOP:
        for (int t_itr = 0; t_itr < timesteps; t_itr++) {
#pragma HLS loop_tripcount min = 10 max = 10
        ASSETS_LOOP:
            for (int a_itr = 0; a_itr < ASSETS; a_itr++) {
#pragma HLS loop_tripcount min = 5 max = 5
            PATH_LOOP:
                for (int p_itr = 0; p_itr < PathNm; p_itr++) {
#pragma HLS loop_tripcount min = 1024 max = 1024
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 2 max = 2
                    DT z0 = inStrm0.read();
                    DT z1 = inStrm1.read();

                    DT r_0, r_1;
                    _next_body(t_itr, a_itr, p_itr, 0, r_a, r_p, z0, z1, r_0, r_1);
                    if (!(firstrun && t_itr == 0)) {
                        outStrm0.write(r_0);
                        outStrm1.write(r_1);
                        outStrm2.write(-r_0);
                        outStrm3.write(-r_1);
                    }
                }
            }
        }
        firstrun = false;
    }
};

//*******************************************************************
// output order
// asset-0: path-0, p1, p2, p3, ......, p1023
// asset-1: path-0, p1, p2, p3, ......, p1023
// asset-2: path-0, p1, p2, p3, ......, p1023
//......
// asset-N: path-0, p1, p2, p3, ......, p1023
template <typename DT, int SampleNm, int ASSETS>
class CORRAND {
   public:
    CORRAND() {
#pragma HLS inline
    }
    void corrPathCube(unsigned int loopNm,
                      hls::stream<DT>& inStrm,
                      hls::stream<DT>& outStrm,
                      DT corrMatrix[ASSETS][ASSETS]) {
#ifndef __SYNTHESIS__
        std::cout << "----Correlated Matrix--" << std::endl;
        for (int i = 0; i < ASSETS; ++i) {
            for (int j = 0; j < ASSETS; ++j) {
                std::cout << corrMatrix[i][j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "-----------------------" << std::endl;
#endif
        DT buff[ASSETS][SampleNm];
#pragma HLS array_partition variable = buff dim = 1
        for (int n = 0; n < ASSETS; ++n) {
            for (int i = 0; i < loopNm; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1024 max = 1024
                DT dw = inStrm.read();
                DT pre[ASSETS];
#pragma HLS array_partition variable = pre dim = 0
                for (int k = 0; k < ASSETS; ++k) {
#pragma HLS unroll
                    if (i == 0)
                        pre[k] = 0;
                    else
                        pre[k] = buff[k][i];
                }
                DT newD[ASSETS];
#pragma HLS array_partition variable = newD dim = 0
                for (int m = 0; m < ASSETS; ++m) {
#pragma HLS unroll
                    DT mulDw;
                    mulDw = FPTwoMul(corrMatrix[m][n], dw);
                    DT addTmp;
                    addTmp = FPTwoAdd(pre[m], mulDw);
                    newD[m] = addTmp;
                }
                for (int k = 0; k < ASSETS; ++k) {
#pragma HLS unroll
                    buff[k][i] = newD[k];
                }
                outStrm.write(newD[n]);
            }
        }
    }
    void corrPathCube(ap_uint<8> assets,
                      ap_uint<16> timesteps,
                      ap_uint<16> paths,
                      hls::stream<DT>& inStrm,
                      hls::stream<DT>& outStrm,
                      DT corrMatrix[ASSETS][ASSETS]) {
#ifndef __SYNTHESIS__
        std::cout << "----Correlated Matrix--" << std::endl;
        for (int i = 0; i < ASSETS; ++i) {
            for (int j = 0; j < ASSETS; ++j) {
                std::cout << corrMatrix[i][j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "-----------------------" << std::endl;
#endif
        int kk = 0;
        DT buff[ASSETS][SampleNm];
        for (int m = 0; m < ASSETS; ++m) {
            for (int n = 0; n < timesteps; ++n) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1024 max = 1024
                buff[m][n] = 0;
            }
        }
#pragma HLS array_partition variable = buff dim = 1
        for (int p = 0; p < paths; ++p) {
            for (int n = 0; n < ASSETS; ++n) {
                for (int i = 0; i < timesteps; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1024 max = 1024
                    if (n < assets) {
                        DT dw = inStrm.read();
                        DT pre[ASSETS];
#pragma HLS array_partition variable = pre dim = 0
                        for (int k = 0; k < ASSETS; ++k) {
#pragma HLS unroll
                            if (n == 0)
                                pre[k] = 0;
                            else
                                pre[k] = buff[k][i];
                        }
                        DT newD[ASSETS];
#pragma HLS array_partition variable = newD dim = 0
                        for (int m = 0; m < ASSETS; ++m) {
#pragma HLS unroll
                            DT mulDw;
                            mulDw = FPTwoMul(corrMatrix[m][n], dw);
                            DT addTmp;
                            addTmp = FPTwoAdd(pre[m], mulDw);
                            newD[m] = addTmp;
                        }
                        for (int k = 0; k < ASSETS; ++k) {
#pragma HLS unroll
                            buff[k][i] = newD[k];
                        }
#ifndef __SYNTHESIS__
                        if (kk < 10) {
                            std::cout << "Number " << kk << " value=" << newD[n] << std::endl;
                        }
                        kk++;
#endif
                        outStrm.write(newD[n]);
                    }
                }
            }
        }
    }
};
} // namespace details
} // namespace fintech
} // namespace xf
#endif //#ifndef XF_FINTECH_CORRAND_H
