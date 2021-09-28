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
 * @file stream_local_processing.hpp
 * @brief Calculation patterns for stream input.
 *
 * This file is part of Vitis Data Analytics Library.
 */

#ifndef _XF_DATA_ANALYTICS_L1_STREAM_LOCAL_PROCESSING_HPP_
#define _XF_DATA_ANALYTICS_L1_STREAM_LOCAL_PROCESSING_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_data_analytics/common/math_helper.hpp"
#include "xf_data_analytics/common/enums.hpp"

namespace xf {
namespace data_analytics {
namespace common {
namespace internal {

template <typename MType>
class AdditionLatency;

template <>
class AdditionLatency<float> {
   public:
    static const int value = 6;
};

template <>
class AdditionLatency<double> {
   public:
    static const int value = 4;
};

template <typename MType>
class MultiplyLatency;

template <>
class MultiplyLatency<float> {
   public:
    static const int value = 3;
};

template <>
class MultiplyLatency<double> {
   public:
    static const int value = 5;
};

template <typename MType>
class ExpLatency;

template <>
class ExpLatency<float> {
   public:
    static const int value = 10;
};

template <>
class ExpLatency<double> {
   public:
    static const int value = 19;
};

/**
 * @brief Stream-Stream Processing
 *
 * @tparam MType Processing datatype.
 * @tparam D Stream number for vectors.
 * @tparam A Function pointer A.
 * @tparam B Function pointer B.
 * @tparam C Function pointer C.
 * @tparam Latency Latency of function B.
 */
template <typename MType,
          int D,
          MType (*A)(MType op1, MType op2),
          void (*B)(MType& reg, MType op),
          MType (*C)(MType op),
          int Latency>
class ss {
   private:
    // Local buffer latency is longer than Latency of function B.
    // In current design is longer by 5.
    static const int LatencyB = Latency + 5;
    static const int LatencyR = Latency + 5;
    static const int localStrmDepth = (LatencyB * LatencyB * 2);

    void process0(const ap_uint<32> cols,
                  hls::stream<ap_uint<32> >& batchStrm1,
                  hls::stream<ap_uint<32> >& batchStrm2) {
        ap_uint<32> batch = (cols + D - 1) / D;
        batchStrm1.write(batch);
        batchStrm2.write(batch);
    }

    void processA(hls::stream<MType> op1Strm[D],
                  hls::stream<MType> op2Strm[D],
                  const ap_uint<32> rows,
                  const ap_uint<32> cols,
                  hls::stream<ap_uint<32> >& batchStrm1,
                  hls::stream<MType>& partMergeStrm) {
        const ap_uint<32> batch = batchStrm1.read();
        const ap_uint<64> total = rows * batch;

        ap_uint<32> ptrB = 0;
        for (ap_uint<64> i = 0; i < total; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 2600 max = 2600 avg = 2600
            MType tmp[D];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int k = 0; k < D; k++) {
#pragma HLS unroll
                MType op1 = op1Strm[k].read();
                MType op2 = op2Strm[k].read();
                tmp[k] = A(op1, op2);
            }
            MType partMerge = 0;
            for (int k = 0; k < D; k++) {
                if (((ptrB * D) + k) < cols) {
                    B(partMerge, tmp[k]);
                }
            }
            partMergeStrm.write(partMerge);

            ptrB++;
            if (ptrB == batch) {
                ptrB = 0;
            }
        }
    }

    void processA(hls::stream<MType> op1Strm[D],
                  hls::stream<MType> op2Strm[D],
                  hls::stream<bool>& eOpStrm,
                  const ap_uint<32> cols,
                  hls::stream<ap_uint<32> >& batchStrm1,
                  hls::stream<MType>& partMergeStrm,
                  hls::stream<bool>& ePartMergeStrm) {
        const ap_uint<32> batch = batchStrm1.read();

        ap_uint<32> ptrB = 0;
        while (!eOpStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 2600 max = 2600 avg = 2600
            MType tmp[D];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int k = 0; k < D; k++) {
#pragma HLS unroll
                MType op1 = op1Strm[k].read();
                MType op2 = op2Strm[k].read();
                tmp[k] = A(op1, op2);
            }
            MType partMerge = 0;
            for (int k = 0; k < D; k++) {
                if (((ptrB * D) + k) < cols) {
                    B(partMerge, tmp[k]);
                }
            }
            partMergeStrm.write(partMerge);
            ePartMergeStrm.write(false);

            ptrB++;
            if (ptrB == batch) {
                ptrB = 0;
            }
        }
        ePartMergeStrm.write(true);
    }

    void processB(hls::stream<MType>& partMergeStrm,
                  hls::stream<MType> fullMergeStrm[LatencyB],
                  const ap_uint<32> rows,
                  hls::stream<ap_uint<32> >& batchStrm2) {
        const ap_uint<32> batch = batchStrm2.read();
        const ap_uint<64> total = rows * batch;

        MType buff[2][LatencyR][LatencyB];
#pragma HLS array_partition variable = buff dim = 1 complete
#pragma HLS array_partition variable = buff dim = 3 complete

        ap_uint<1> ptrU = 0; // ping-pong in buff
        int ptrR = 0;        // row in buff
        int ptrB = 0;        // batch in buff
        int ptrTR = 0;       // actual rows
        int ptrTB = 0;       // actual batch

        for (ap_uint<64> i = 0; i < total; i++) {
#pragma HLS loop_tripcount min = 2600 max = 2600 avg = 2600
#pragma HLS dependence variable = buff inter false
#pragma HLS pipeline II = 1
            // load data
            MType partMerge = partMergeStrm.read();
            MType reg;
            if (ptrU == 0) {
                reg = buff[0][ptrR][ptrB];
            } else {
                reg = buff[1][ptrR][ptrB];
            }
            if (ptrTB < LatencyB) {
                reg = 0;
            }
            // compute
            B(reg, partMerge);
            // update buff
            if (ptrU == 0) {
                buff[0][ptrR][ptrB] = reg;
            } else {
                buff[1][ptrR][ptrB] = reg;
            }
            // write out
            if ((ptrTB == (batch - 1)) && (ptrTR >= LatencyR)) {
                if (ptrU == 0) {
                    for (int k = 0; k < LatencyB; k++) {
#pragma HLS unroll
                        MType tmp = buff[1][ptrR][k];
                        if (k >= batch) {
                            tmp = 0;
                        }
                        fullMergeStrm[k].write(tmp);
                    }
                } else {
                    for (int k = 0; k < LatencyB; k++) {
#pragma HLS unroll
                        MType tmp = buff[0][ptrR][k];
                        if (k >= batch) {
                            tmp = 0;
                        }
                        fullMergeStrm[k].write(tmp);
                    }
                }
            }
            // update ptr
            ptrB++;
            if ((ptrB == LatencyB) || (ptrTB == (batch - 1))) {
                ptrB = 0;
            }
            if (ptrTB == (batch - 1)) {
                ptrR++;
                if (ptrR == LatencyR) {
                    ptrR = 0;
                    ptrU ^= ap_uint<1>(1);
                }
            }
            ptrTB++;
            if (ptrTB == batch) {
                ptrTB = 0;
                ptrTR++;
            }
        }

        int tailNum;
        if (rows >= LatencyR) {
            tailNum = LatencyR;
        } else {
            tailNum = rows;
            ptrU = 1;
            ptrR = 0;
        }

        for (int i = 0; i < tailNum; i++) {
#pragma HLS loop_tripcount min = LatencyB max = LatencyB avg = LatencyB
#pragma HLS pipeline II = 1
            for (int j = 0; j < LatencyB; j++) {
#pragma HLS unroll
                MType tmp = buff[1 - ptrU][ptrR][j];
                if (j >= batch) {
                    tmp = 0;
                }
                fullMergeStrm[j].write(tmp);
            }
            ptrR++;
            if (ptrR == LatencyR) {
                ptrR = 0;
                ptrU ^= ap_uint<1>(1);
            }
        }
    }

    void processB(hls::stream<MType>& partMergeStrm,
                  hls::stream<bool>& ePartMergeStrm,
                  hls::stream<MType> fullMergeStrm[LatencyB],
                  hls::stream<bool>& eFullMergeStrm,
                  hls::stream<ap_uint<32> >& batchStrm2) {
        const ap_uint<32> batch = batchStrm2.read();

        MType buff[2][LatencyR][LatencyB];
#pragma HLS array_partition variable = buff dim = 1 complete
#pragma HLS array_partition variable = buff dim = 3 complete

        ap_uint<1> ptrU = 0; // ping-pong in buff
        int ptrR = 0;        // row in buff
        int ptrB = 0;        // batch in buff
        int ptrTR = 0;       // actual rows
        int ptrTB = 0;       // actual batch

        while (!ePartMergeStrm.read()) {
#pragma HLS loop_tripcount min = 2600 max = 2600 avg = 2600
#pragma HLS dependence variable = buff inter false
#pragma HLS pipeline II = 1
            // load data
            MType partMerge = partMergeStrm.read();
            MType reg;
            if (ptrU == 0) {
                reg = buff[0][ptrR][ptrB];
            } else {
                reg = buff[1][ptrR][ptrB];
            }
            if (ptrTB < LatencyB) {
                reg = 0;
            }
            // compute
            B(reg, partMerge);
            // update buff
            if (ptrU == 0) {
                buff[0][ptrR][ptrB] = reg;
            } else {
                buff[1][ptrR][ptrB] = reg;
            }
            // write out
            if ((ptrTB == (batch - 1)) && (ptrTR >= LatencyR)) {
                if (ptrU == 0) {
                    for (int k = 0; k < LatencyB; k++) {
#pragma HLS unroll
                        MType tmp = buff[1][ptrR][k];
                        if (k >= batch) {
                            tmp = 0;
                        }
                        fullMergeStrm[k].write(tmp);
                    }
                    eFullMergeStrm.write(false);
                } else {
                    for (int k = 0; k < LatencyB; k++) {
#pragma HLS unroll
                        MType tmp = buff[0][ptrR][k];
                        if (k >= batch) {
                            tmp = 0;
                        }
                        fullMergeStrm[k].write(tmp);
                    }
                    eFullMergeStrm.write(false);
                }
            }
            // update ptr
            ptrB++;
            if ((ptrB == LatencyB) || (ptrTB == (batch - 1))) {
                ptrB = 0;
            }
            ptrTB++;
            if (ptrTB == batch) {
                ptrTB = 0;
                ptrR++;
                if (ptrR == LatencyR) {
                    ptrR = 0;
                    ptrU ^= ap_uint<1>(1);
                }
                ptrTR++;
            }
        }

        int tailNum;
        if (ptrTR >= LatencyR) {
            tailNum = LatencyR;
        } else {
            tailNum = ptrTR;
            ptrU = 1;
            ptrR = 0;
        }

        for (int i = 0; i < tailNum; i++) {
#pragma HLS loop_tripcount min = LatencyB max = LatencyB avg = LatencyB
#pragma HLS pipeline II = 1
            for (int j = 0; j < LatencyB; j++) {
#pragma HLS unroll
                MType tmp = buff[1 - ptrU][ptrR][j];
                if (j >= batch) {
                    tmp = 0;
                }
                fullMergeStrm[j].write(tmp);
            }
            eFullMergeStrm.write(false);
            ptrR++;
            if (ptrR == LatencyR) {
                ptrR = 0;
                ptrU ^= ap_uint<1>(1);
            }
        }
        eFullMergeStrm.write(true);
    }

    void processC(hls::stream<MType> fullMergeStrm[LatencyB],
                  hls::stream<MType>& retStrm,
                  const ap_uint<32> rows,
                  const ap_uint<32> cols) {
    LOOP_C_1:
        for (ap_uint<32> i = 0; i < rows; i++) {
#pragma HLS loop_tripcount min = 200 max = 200 avg = 200
#pragma HLS pipeline II = 1
            MType ret = 0;
        LOOP_C_2:
            for (int j = 0; j < LatencyB; j++) {
                MType tmp = fullMergeStrm[j].read();
                if (j < cols) {
                    B(ret, tmp);
                }
            }
            retStrm.write(C(ret));
        }
    }

    void processC(hls::stream<MType> fullMergeStrm[LatencyB],
                  hls::stream<bool>& eFullMergeStrm,
                  hls::stream<MType>& retStrm,
                  hls::stream<bool>& eRetStrm,
                  const ap_uint<32> cols) {
    LOOP_C_1:
        while (!eFullMergeStrm.read()) {
#pragma HLS loop_tripcount min = 200 max = 200 avg = 200
#pragma HLS pipeline II = 1
            MType ret = 0;
        LOOP_C_2:
            for (int j = 0; j < LatencyB; j++) {
                MType tmp = fullMergeStrm[j].read();
                if (j < cols) {
                    B(ret, tmp);
                }
            }
            retStrm.write(C(ret));
            eRetStrm.write(false);
        }
        eRetStrm.write(true);
    }

   public:
    /**
     * @brief Stream-Stream vector processing.
     * Assume vector op1 comes from op1Strm, op2 from op2Strm.
     * Result ret will go to reStrm.
     * tmp_1[k] = A(op1[k], op2[k]);
     * tmp_2 = B(...B(B(0, tmp[0]), tmp[1]) ..., tmp[k]);
     * ret = C(tmp2);
     * One thing must be notice that function B should not be chosen arbitrarily.
     * Current only support B(reg, o) :=  reg+=op
     *
     * @param op1Strm Streams that get op1.
     * @param op2Strm Streams that get op2.
     * @param retStrm Streams that send ret.
     * @param rows Number of op1/op2.
     * @param cols Dimension of op1/op2.
     */
    void process(hls::stream<MType> op1Strm[D],
                 hls::stream<MType> op2Strm[D],
                 hls::stream<MType>& retStrm,
                 const ap_uint<32> rows,
                 const ap_uint<32> cols) {
#pragma HLS dataflow
        hls::stream<ap_uint<32> > batchStrm1;
        hls::stream<ap_uint<32> > batchStrm2;
        hls::stream<MType> partMergeStrm;
        hls::stream<MType> fullMergeStrm[LatencyB];
#pragma HLS stream variable = batchStrm1 depth = 1
#pragma HLS stream variable = batchStrm2 depth = 1
#pragma HLS stream variable = partMergeStrm depth = localStrmDepth
#pragma HLS stream variable = fullMergeStrm depth = localStrmDepth
#pragma HLS array_partition variable = fullMergeStrm dim = 1 complete
        process0(cols, batchStrm1, batchStrm2);
        processA(op1Strm, op2Strm, rows, cols, batchStrm1, partMergeStrm);
        processB(partMergeStrm, fullMergeStrm, rows, batchStrm2);
        processC(fullMergeStrm, retStrm, rows, cols);
    }

    /**
     * @brief Stream-Stream vector processing.
     * Assume vector op1 comes from op1Strm, op2 from op2Strm.
     * Result ret will go to reStrm.
     * tmp_1[k] = A(op1[k], op2[k]);
     * tmp_2 = B(...B(B(0, tmp[0]), tmp[1]) ..., tmp[k]);
     * ret = C(tmp2);
     * One thing must be notice that function B should not be chosen arbitrarily.
     * Current only support B(reg, o) :=  reg+=op
     *
     * @param op1Strm Streams that get op1.
     * @param op2Strm Streams that get op2.
     * @param retStrm Streams that send ret.
     * @param rows Number of op1/op2.
     * @param cols Dimension of op1/op2.
     */
    void process(hls::stream<MType> op1Strm[D],
                 hls::stream<MType> op2Strm[D],
                 hls::stream<bool>& eOpStrm,
                 hls::stream<MType>& retStrm,
                 hls::stream<bool>& eRetStrm,
                 const ap_uint<32> cols) {
#pragma HLS dataflow
        hls::stream<ap_uint<32> > batchStrm1;
        hls::stream<ap_uint<32> > batchStrm2;
        hls::stream<MType> partMergeStrm;
        hls::stream<bool> ePartMergeStrm;
        hls::stream<MType> fullMergeStrm[LatencyB];
        hls::stream<bool> eFullMergeStrm;
#pragma HLS stream variable = batchStrm1 depth = 1
#pragma HLS stream variable = batchStrm2 depth = 1
#pragma HLS stream variable = partMergeStrm depth = localStrmDepth
#pragma HLS stream variable = ePartMergeStrm depth = localStrmDepth
#pragma HLS stream variable = fullMergeStrm depth = localStrmDepth
#pragma HLS stream variable = eFullMergeStrm depth = localStrmDepth
#pragma HLS array_partition variable = fullMergeStrm dim = 1 complete
        process0(cols, batchStrm1, batchStrm2);
        processA(op1Strm, op2Strm, eOpStrm, cols, batchStrm1, partMergeStrm, ePartMergeStrm);
        processB(partMergeStrm, ePartMergeStrm, fullMergeStrm, eFullMergeStrm, batchStrm2);
        processC(fullMergeStrm, eFullMergeStrm, retStrm, eRetStrm, cols);
    }
};

/**
 * D, dimension unroll, load D cordinate of single vector at one clock cycle.
 * K, vector unroll, load 2K vectors at the same time.
 * DMax, max dimension supported
 * KAMax, max vector supported in matrix A
 * KBMax, max vector supported in matrix B
 */
template <typename MType,
          int D,
          int K,
          int KAMax,
          int KBMax,
          MType (*funcA)(MType op1, MType op2),
          void (*funcB)(MType& reg, MType op),
          MType (*funcC)(MType op),
          RAMType RAMBuff>
class ll {
   public:
    MType MA[K][D][KAMax];
    MType MB[K][D][KBMax];

    ll() {
#pragma HLS inline
#pragma HLS array_partition variable = MA dim = 1
#pragma HLS array_partition variable = MA dim = 2
#pragma HLS array_partition variable = MB dim = 1
#pragma HLS array_partition variable = MB dim = 2
        if (RAMBuff == URAM) {
#pragma HLS bind_storage variable = MA type = ram_2p impl = uram
#pragma HLS bind_storage variable = MB type = ram_2p impl = uram
        } else if (RAMBuff == BRAM) {
#pragma HLS bind_storage variable = MA type = ram_2p impl = bram
#pragma HLS bind_storage variable = MB type = ram_2p impl = bram
        } else if (RAMBuff == LUTRAM) {
#pragma HLS bind_storage variable = MA type = ram_2p impl = lutram
#pragma HLS bind_storage variable = MB type = ram_2p impl = lutram
        }
    }

    void process(hls::stream<ap_uint<32> > ptrAStrm[K * 2],
                 hls::stream<ap_uint<32> > ptrBStrm[K * 2],
                 hls::stream<bool>& ePtrStrm,
                 const ap_uint<32> effectiveD,
                 hls::stream<MType> retStrm[K * 2],
                 hls::stream<bool>& eRetStrm) {
        while (!ePtrStrm.read()) {
#pragma HLS pipeline II = 1
            for (int i = 0; i < K * 2; i++) {
#pragma HLS unroll
                ap_uint<32> ptrA = ptrAStrm[i].read();
                ap_uint<32> ptrB = ptrBStrm[i].read();

                MType vecTmp[D];
                for (int j = 0; j < D; j++) {
#pragma HLS unroll
                    MType eleA = MA[i / 2][j][ptrA];
                    MType eleB = MB[i / 2][j][ptrB];
                    vecTmp[j] = funcA(eleA, eleB);
                }

                MType reg = 0;
                for (int j = 0; j < D; j++) {
                    if (j < effectiveD) {
                        funcB(reg, vecTmp[j]);
                    }
                }

                MType ret = funcC(reg);
                retStrm[i].write(ret);
            }
            eRetStrm.write(false);
        }
        eRetStrm.write(true);
    }

    void setMA(MType input[D][KAMax], const ap_uint<32> dep) {
        for (ap_uint<32> i = 0; i < dep; i++) {
#pragma HLS pipeline II = 1
            for (int j = 0; j < D; j++) {
#pragma HLS unroll
                MType tmp = input[j][i];
                for (int k = 0; k < K; k++) {
#pragma HLS unroll
                    MA[k][j][i] = tmp;
                }
            }
        }
    }

    void setMB(MType input[D][KBMax], const ap_uint<32> dep) {
        for (ap_uint<32> i = 0; i < dep; i++) {
#pragma HLS pipeline II = 1
            for (int j = 0; j < D; j++) {
#pragma HLS unroll
                MType tmp = input[j][i];
                for (int k = 0; k < K; k++) {
#pragma HLS unroll
                    MB[k][j][i] = tmp;
                }
            }
        }
    }
};

template <typename MType,
          int D,
          int DMax,
          int K,
          int KMax,
          MType (*A)(MType op1, MType op2),
          void (*B)(MType& reg, MType op),
          MType (*C)(MType op),
          int Latency,
          RAMType RAMWeight>
class sl {
   public:
    MType weight[K][KMax][D][DMax];
    sl() {
#pragma HLS inline
#pragma HLS array_partition variable = weight dim = 1
#pragma HLS array_partition variable = weight dim = 3
        if (RAMWeight == URAM) {
#pragma HLS bind_storage variable = weight type = ram_2p impl = uram
        } else if (RAMWeight == BRAM) {
#pragma HLS bind_storage variable = weight type = ram_2p impl = bram
        } else if (RAMWeight == LUTRAM) {
#pragma HLS bind_storage variable = weight type = ram_2p impl = lutram
        }
    }

   private:
    // Local buffer latency is longer than Latency of function B.
    // In current design is longer by 5.
    static const int LatencyB = Latency + 5;
    static const int LatencyR = Latency + 5;
    static const int localStrmDepth = (LatencyB * LatencyB * 2);

    void process0(const ap_uint<32> cols,
                  const ap_uint<32> ws,
                  hls::stream<ap_uint<64> >& batchStrmR,
                  hls::stream<ap_uint<64> >& batchStrmA,
                  hls::stream<ap_uint<64> >& batchStrmB,
                  hls::stream<ap_uint<64> >& batchStrmC) {
        const ap_uint<32> col_batch = (cols + D - 1) / D;
        const ap_uint<32> ws_batch = (ws + K - 1) / K;

        batchStrmR.write(ap_uint<64>(ws_batch));

        batchStrmA.write(ap_uint<64>(col_batch));
        batchStrmA.write(ap_uint<64>(ws_batch));

        batchStrmB.write(ap_uint<64>(col_batch));
        batchStrmB.write(ap_uint<64>(ws_batch));

        batchStrmC.write(ap_uint<64>(col_batch));
        batchStrmC.write(ap_uint<64>(ws_batch));
    }

    void processR(hls::stream<MType> opStrm[D],
                  hls::stream<bool>& eOpStrm,
                  hls::stream<MType> rOpStrm[D],
                  hls::stream<bool>& eROpStrm,
                  hls::stream<ap_uint<64> >& batchStrmR) {
        const ap_uint<32> ws_batch = batchStrmR.read();
        bool e = eOpStrm.read();
        ap_uint<32> counterWB = 0;
        MType opIn[D];
        while (!e) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 13000 max = 13000 avg = 13000
            if (counterWB == 0) {
                for (int i = 0; i < D; i++) {
#pragma HLS unroll
                    opIn[i] = opStrm[i].read();
                }
            }
            for (int i = 0; i < D; i++) {
#pragma HLS unroll
                rOpStrm[i].write(opIn[i]);
            }
            eROpStrm.write(false);

            counterWB++;
            if (counterWB == ws_batch) {
                counterWB = 0;
                e = eOpStrm.read();
            }
        }
        eROpStrm.write(true);
    }

    void processA(hls::stream<MType> rOpStrm[D],
                  hls::stream<bool>& eROpStrm,
                  const ap_uint<32> cols,
                  const ap_uint<32> ws,
                  hls::stream<ap_uint<64> >& batchStrmA,
                  hls::stream<MType> partMergeStrm[K],
                  hls::stream<bool>& ePartMergeStrm) {
        const ap_uint<32> col_batch = batchStrmA.read();
        const ap_uint<32> ws_batch = batchStrmA.read();

        ap_uint<32> ptrK = 0;
        ap_uint<32> ptrD = 0;
        while (!eROpStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 13000 max = 13000 avg = 13000

            MType tmp[K][D];
#pragma HLS array_partition variable = tmp dim = 1 complete
#pragma HLS array_partition variable = tmp dim = 2 complete

            for (int d = 0; d < D; d++) {
#pragma HLS unroll
                MType op = rOpStrm[d].read();
                for (int k = 0; k < K; k++) {
#pragma HLS unroll
                    tmp[k][d] = A(op, weight[k][ptrK][d][ptrD]);
                }
            }

            for (int k = 0; k < K; k++) {
#pragma HLS unroll
                MType partMerge = 0;
                for (int d = 0; d < D; d++) {
                    if ((ptrD * D + d) < cols) {
                        B(partMerge, tmp[k][d]);
                    }
                }
                partMergeStrm[k].write(partMerge);
            }

            ePartMergeStrm.write(false);
            // pointer update
            ptrK++;
            if (ptrK == ws_batch) {
                ptrK = 0;
                ptrD++;
                if (ptrD == col_batch) {
                    ptrD = 0;
                }
            }
        }
        ePartMergeStrm.write(true);
    }

    void processB(hls::stream<MType> partMergeStrm[K],
                  hls::stream<bool>& ePartMergeStrm,
                  const ap_uint<32> cols,
                  const ap_uint<32> ws,
                  hls::stream<ap_uint<64> >& batchStrmB,
                  hls::stream<MType> fullMergeStrm[K][LatencyB],
                  hls::stream<bool>& eFullMergeStrm) {
        const ap_uint<32> col_batch = batchStrmB.read();
        const ap_uint<32> ws_batch = batchStrmB.read();

        MType buff[K][KMax][2][LatencyR][LatencyB];
#pragma HLS array_partition variable = buff dim = 1 complete
#pragma HLS array_partition variable = buff dim = 3 complete
#pragma HLS array_partition variable = buff dim = 5 complete

        ap_uint<1> ptrU = 0; // ping-pong in buff
        int ptrR = 0;        // row in buff
        int ptrB = 0;        // Col batch in buff
        int ptrK = 0;        // K batch in buff
        int ptrTR = 0;       // actual rows
        int ptrTB = 0;       // actual batch

        while (!ePartMergeStrm.read()) {
#pragma HLS dependence variable = buff inter false
#pragma HLS loop_tripcount min = 13000 max = 13000 avg = 13000
#pragma HLS pipeline II = 1

            // load data
            MType partMerge[K];
#pragma HLS array_partition variable = partMerge dim = 1 complete
            MType reg[K];
#pragma HLS array_partition variable = reg dim = 1 complete

            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                partMerge[i] = partMergeStrm[i].read();
            }
            if (ptrU == 0) {
                for (int i = 0; i < K; i++) {
#pragma HLS unroll
                    reg[i] = buff[i][ptrK][0][ptrR][ptrB];
                    if (ptrTB < LatencyB) {
                        reg[i] = 0;
                    }
                }
            } else {
                for (int i = 0; i < K; i++) {
#pragma HLS unroll
                    reg[i] = buff[i][ptrK][1][ptrR][ptrB];
                    if (ptrTB < LatencyB) {
                        reg[i] = 0;
                    }
                }
            }
            // compute
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                B(reg[i], partMerge[i]);
            }
            // update buff
            if (ptrU == 0) {
                for (int i = 0; i < K; i++) {
#pragma HLS unroll
                    buff[i][ptrK][0][ptrR][ptrB] = reg[i];
                }
            } else {
                for (int i = 0; i < K; i++) {
#pragma HLS unroll
                    buff[i][ptrK][1][ptrR][ptrB] = reg[i];
                }
            }
            // write out
            if ((ptrTB == (col_batch - 1)) && (ptrTR >= LatencyR)) {
                if (ptrU == 0) {
                    for (int i = 0; i < K; i++) {
#pragma HLS unroll
                        for (int j = 0; j < LatencyB; j++) {
#pragma HLS unroll
                            MType tmp = buff[i][ptrK][1][ptrR][j];
                            if (j >= col_batch) {
                                tmp = 0;
                            }
                            fullMergeStrm[i][j].write(tmp);
                        }
                    }
                    eFullMergeStrm.write(false);
                } else {
                    for (int i = 0; i < K; i++) {
#pragma HLS unroll
                        for (int j = 0; j < LatencyB; j++) {
#pragma HLS unroll
                            MType tmp = buff[i][ptrK][0][ptrR][j];
                            if (j >= col_batch) {
                                tmp = 0;
                            }
                            fullMergeStrm[i][j].write(tmp);
                        }
                    }
                    eFullMergeStrm.write(false);
                }
            }
            // update ptr
            ptrK++;
            if (ptrK == ws_batch) {
                ptrK = 0;

                ptrB++;
                if ((ptrB == LatencyB) || (ptrTB == (col_batch - 1))) {
                    ptrB = 0;
                }
                ptrTB++;
                if (ptrTB == col_batch) {
                    ptrTB = 0;
                    ptrR++;
                    if (ptrR == LatencyR) {
                        ptrR = 0;
                        ptrU ^= ap_uint<1>(1);
                    }
                    ptrTR++;
                }
            }
        }

        int tailNum;
        if (ptrTR >= LatencyR) {
            tailNum = LatencyR;
        } else {
            tailNum = ptrTR;
            ptrU = 1;
            ptrR = 0;
        }

        for (int i = 0; i < tailNum; i++) {
#pragma HLS loop_tripcount min = 11 max = 11 avg = 11
            for (int j = 0; j < ws_batch; j++) {
#pragma HLS loop_tripcount min = 5 max = 5 avg = 5
#pragma HLS pipeline II = 1
                for (int k = 0; k < K; k++) {
#pragma HLS unroll
                    for (int l = 0; l < LatencyB; l++) {
#pragma HLS unroll
                        MType tmp = buff[k][j][1 - ptrU][ptrR][l];
                        if (l >= col_batch) {
                            tmp = 0;
                        }
                        fullMergeStrm[k][l].write(tmp);
                    }
                }
                eFullMergeStrm.write(false);
            }
            ptrR++;
            if (ptrR == LatencyR) {
                ptrR = 0;
                ptrU ^= ap_uint<1>(1);
            }
        }
        eFullMergeStrm.write(true);
    }

    void processC(hls::stream<MType> fullMergeStrm[K][LatencyB],
                  hls::stream<bool>& eFullMergeStrm,
                  const ap_uint<32> cols,
                  const ap_uint<32> ws,
                  hls::stream<ap_uint<64> >& batchStrmC,
                  hls::stream<MType> retStrm[K],
                  hls::stream<bool>& eRetStrm) {
        const ap_uint<32> col_batch = batchStrmC.read();
        const ap_uint<32> ws_batch = batchStrmC.read();

        while (!eFullMergeStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000 avg = 1000
            // update ptr
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                MType ret = 0;
                for (int j = 0; j < LatencyB; j++) {
                    MType tmp = fullMergeStrm[i][j].read();
                    if (j < cols) {
                        B(ret, tmp);
                    }
                }
                retStrm[i].write(C(ret));
            }
            eRetStrm.write(false);
        }
        eRetStrm.write(true);
    }

   public:
    void process(hls::stream<MType> opStrm[D],
                 hls::stream<bool>& eOpStrm,
                 hls::stream<MType> retStrm[K],
                 hls::stream<bool>& eRetStrm,
                 const ap_uint<32> cols,
                 const ap_uint<32> ws) {
#pragma HLS dataflow
        hls::stream<ap_uint<64> > batchStrmR;
        hls::stream<ap_uint<64> > batchStrmA;
        hls::stream<ap_uint<64> > batchStrmB;
        hls::stream<ap_uint<64> > batchStrmC;
        hls::stream<MType> rOpStrm[D];
        hls::stream<MType> partMergeStrm[K];
        hls::stream<MType> fullMergeStrm[K][LatencyB];
        hls::stream<bool> eROpStrm;
        hls::stream<bool> ePartMergeStrm;
        hls::stream<bool> eFullMergeStrm;
#pragma HLS stream variable = batchStrmR depth = 1
#pragma HLS stream variable = batchStrmA depth = 2
#pragma HLS stream variable = batchStrmB depth = 2
#pragma HLS stream variable = batchStrmC depth = 2
#pragma HLS stream variable = partMergeStrm depth = localStrmDepth
#pragma HLS stream variable = fullMergeStrm depth = localStrmDepth
#pragma HLS stream variable = ePartMergeStrm depth = localStrmDepth
#pragma HLS stream variable = eFullMergeStrm depth = localStrmDepth
#pragma HLS array_partition variable = partMergeStrm dim = 1 complete
#pragma HLS array_partition variable = fullMergeStrm dim = 1 complete
#pragma HLS array_partition variable = fullMergeStrm dim = 2 complete
        process0(cols, ws, batchStrmR, batchStrmA, batchStrmB, batchStrmC);
        processR(opStrm, eOpStrm, rOpStrm, eROpStrm, batchStrmR);
        processA(rOpStrm, eROpStrm, cols, ws, batchStrmA, partMergeStrm, ePartMergeStrm);
        processB(partMergeStrm, ePartMergeStrm, cols, ws, batchStrmB, fullMergeStrm, eFullMergeStrm);
        processC(fullMergeStrm, eFullMergeStrm, cols, ws, batchStrmC, retStrm, eRetStrm);
    }

    void setWeight(MType inputW[K][KMax][D][DMax], const ap_uint<32> cols, const ap_uint<32> ws) {
        bool e = false;

        ap_uint<32> ptrK = 0;
        ap_uint<32> ptrTK = 0;
        ap_uint<32> ptrD = 0;
        ap_uint<32> ptrTD = 0;
        while (!e) {
#pragma HLS pipeline II = 1
            for (int i = 0; i < D; i++) {
#pragma HLS unroll
                for (int j = 0; j < K; j++) {
                    weight[j][ptrK][i][ptrD] = inputW[j][ptrK][i][ptrD];
                }
            }
            // update ptr
            ptrK++;
            ptrTK += K;
            if (ptrTK >= ws) {
                ptrK = 0;
                ptrTK = 0;
                ptrD++;
                ptrTD += D;
                if (ptrTD >= cols) {
                    e = true;
                }
            }
        }
    }
};

template <typename MType,
          int D,
          int DMax,
          int K,
          int KMax,
          MType (*A)(MType op1, MType op2),
          void (*B)(MType& reg, MType op),
          MType (*C)(MType op),
          int Latency,
          RAMType RAMWeight,
          RAMType RAMIntercept>
class sl2 {
   public:
    static const int LatencyB = Latency + 5;
    static const int LatencyR = Latency + 5;
    static const int localStrmDepth = (LatencyB * LatencyB * 2);
    static const int Proc0Latency = 30;
    static const int ProcALatency = (Latency + 4) * D;
    static const int ProcBLatency = DMax * LatencyR * 2;
    static const int ProcCLatency = LatencyB * (Latency + 6);
    static const int LatencyT = KMax * (Proc0Latency + ProcALatency + ProcBLatency + ProcCLatency);

    MType weight[K][D][KMax * DMax];
    MType intercept[K][KMax];
    sl2() {
#pragma HLS inline
#pragma HLS array_partition variable = weight dim = 1
#pragma HLS array_partition variable = weight dim = 2
#pragma HLS array_partition variable = intercept dim = 1
        if (RAMWeight == URAM) {
#pragma HLS bind_storage variable = weight type = ram_2p impl = uram
        } else if (RAMWeight == BRAM) {
#pragma HLS bind_storage variable = weight type = ram_2p impl = bram
        } else if (RAMWeight == LUTRAM) {
#pragma HLS bind_storage variable = weight type = ram_2p impl = lutram
        }

        if (RAMIntercept == URAM) {
#pragma HLS bind_storage variable = intercept type = ram_2p impl = uram
        } else if (RAMIntercept == BRAM) {
#pragma HLS bind_storage variable = intercept type = ram_2p impl = bram
        } else if (RAMIntercept == LUTRAM) {
#pragma HLS bind_storage variable = intercept type = ram_2p impl = lutram
        }
    }

   private:
    // Local buffer latency is longer than Latency of function B.
    // In current design is longer by 5.
    void processR(hls::stream<MType> opStrm[D],
                  hls::stream<bool>& eOpStrm,
                  hls::stream<MType> rOpStrm[D],
                  hls::stream<bool>& eROpStrm,
                  ap_uint<32> ws) {
        ap_uint<32> ws_batch = (ws + K - 1) / K;
        bool e = eOpStrm.read();
        ap_uint<32> counterWB = 0;
        MType opIn[D];
        while (!e) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 26000 max = 26000 avg = 26000
            if (counterWB == 0) {
                for (int i = 0; i < D; i++) {
#pragma HLS unroll
                    opIn[i] = opStrm[i].read();
                }
            }
            for (int i = 0; i < D; i++) {
#pragma HLS unroll
                rOpStrm[i].write(opIn[i]);
            }
            eROpStrm.write(false);

            counterWB++;
            if (counterWB == ws_batch) {
                counterWB = 0;
                e = eOpStrm.read();
            }
        }
        eROpStrm.write(true);
    }

    void processA(hls::stream<MType> rOpStrm[D],
                  hls::stream<bool>& eROpStrm,
                  ap_uint<32> cols,
                  ap_uint<32> ws,
                  hls::stream<MType> partMergeStrm[K],
                  hls::stream<bool>& ePartMergeStrm) {
        ap_uint<32> col_batch = (cols + D - 1) / D;
        ap_uint<32> ws_batch = (ws + K - 1) / K;

        ap_uint<32> ptrK = 0;
        ap_uint<32> ptrD = 0;
        while (!eROpStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 26000 max = 26000 avg = 26000

            MType tmp[K][D];
#pragma HLS array_partition variable = tmp dim = 1 complete
#pragma HLS array_partition variable = tmp dim = 2 complete

            for (int d = 0; d < D; d++) {
#pragma HLS unroll
                MType op = rOpStrm[d].read();
                for (int k = 0; k < K; k++) {
#pragma HLS unroll
                    tmp[k][d] = A(op, weight[k][d][ptrK * col_batch + ptrD]);
                }
            }

            for (int k = 0; k < K; k++) {
#pragma HLS unroll
                MType partMerge = 0;
                for (int d = 0; d < D; d++) {
                    if ((ptrD * D + d) < cols) {
                        B(partMerge, tmp[k][d]);
                    }
                }
                partMergeStrm[k].write(partMerge);
            }

            ePartMergeStrm.write(false);
            // pointer update
            ptrK++;
            if (ptrK == ws_batch) {
                ptrK = 0;
                ptrD++;
                if (ptrD == col_batch) {
                    ptrD = 0;
                }
            }
        }
        ePartMergeStrm.write(true);
    }

    void processB(hls::stream<MType> partMergeStrm[K],
                  hls::stream<bool>& ePartMergeStrm,
                  ap_uint<32> cols,
                  ap_uint<32> ws,
                  hls::stream<MType> fullMergeStrm[K][LatencyB],
                  hls::stream<bool>& eFullMergeStrm) {
        ap_uint<32> col_batch = (cols + D - 1) / D;
        ap_uint<32> ws_batch = (ws + K - 1) / K;

        MType buff[K][KMax][2][LatencyR][LatencyB];
#pragma HLS array_partition variable = buff dim = 1 complete
#pragma HLS array_partition variable = buff dim = 3 complete
#pragma HLS array_partition variable = buff dim = 5 complete

        ap_uint<1> ptrU = 0; // ping-pong in buff
        int ptrR = 0;        // row in buff
        int ptrB = 0;        // Col batch in buff
        int ptrK = 0;        // K batch in buff
        int ptrTR = 0;       // actual rows
        int ptrTB = 0;       // actual batch

        while (!ePartMergeStrm.read()) {
#pragma HLS dependence variable = buff inter false
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 26000 max = 26000 avg = 26000

            // load data
            MType partMerge[K];
#pragma HLS array_partition variable = partMerge dim = 1 complete
            MType reg[K];
#pragma HLS array_partition variable = reg dim = 1 complete

            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                partMerge[i] = partMergeStrm[i].read();
            }
            if (ptrU == 0) {
                for (int i = 0; i < K; i++) {
#pragma HLS unroll
                    reg[i] = buff[i][ptrK][0][ptrR][ptrB];
                    if (ptrTB < LatencyB) {
                        reg[i] = 0;
                    }
                }
            } else {
                for (int i = 0; i < K; i++) {
#pragma HLS unroll
                    reg[i] = buff[i][ptrK][1][ptrR][ptrB];
                    if (ptrTB < LatencyB) {
                        reg[i] = 0;
                    }
                }
            }
            // compute
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                B(reg[i], partMerge[i]);
            }
            // update buff
            if (ptrU == 0) {
                for (int i = 0; i < K; i++) {
#pragma HLS unroll
                    buff[i][ptrK][0][ptrR][ptrB] = reg[i];
                }
            } else {
                for (int i = 0; i < K; i++) {
#pragma HLS unroll
                    buff[i][ptrK][1][ptrR][ptrB] = reg[i];
                }
            }
            // write out
            if ((ptrTB == (col_batch - 1)) && (ptrTR >= LatencyR)) {
                if (ptrU == 0) {
                    for (int i = 0; i < K; i++) {
#pragma HLS unroll
                        for (int j = 0; j < LatencyB; j++) {
#pragma HLS unroll
                            MType tmp = buff[i][ptrK][1][ptrR][j];
                            if (j >= col_batch) {
                                tmp = 0;
                            }
                            fullMergeStrm[i][j].write(tmp);
                        }
                    }
                    eFullMergeStrm.write(false);
                } else {
                    for (int i = 0; i < K; i++) {
#pragma HLS unroll
                        for (int j = 0; j < LatencyB; j++) {
#pragma HLS unroll
                            MType tmp = buff[i][ptrK][0][ptrR][j];
                            if (j >= col_batch) {
                                tmp = 0;
                            }
                            fullMergeStrm[i][j].write(tmp);
                        }
                    }
                    eFullMergeStrm.write(false);
                }
            }
            // update ptr
            ptrK++;
            if (ptrK == ws_batch) {
                ptrK = 0;

                ptrB++;
                if ((ptrB == LatencyB) || (ptrTB == (col_batch - 1))) {
                    ptrB = 0;
                }
                ptrTB++;
                if (ptrTB == col_batch) {
                    ptrTB = 0;
                    ptrR++;
                    if (ptrR == LatencyR) {
                        ptrR = 0;
                        ptrU ^= ap_uint<1>(1);
                    }
                    ptrTR++;
                }
            }
        }

        int tailNum;
        if (ptrTR >= LatencyR) {
            tailNum = LatencyR;
        } else {
            tailNum = ptrTR;
            ptrU = 1;
            ptrR = 0;
        }

        for (int i = 0; i < tailNum; i++) {
#pragma HLS loop_tripcount min = 12 max = 12 avg = 12
            for (int j = 0; j < ws_batch; j++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 5 max = 5 avg = 5
                for (int k = 0; k < K; k++) {
#pragma HLS unroll
                    for (int l = 0; l < LatencyB; l++) {
#pragma HLS unroll
                        MType tmp = buff[k][j][1 - ptrU][ptrR][l];
                        if (l >= col_batch) {
                            tmp = 0;
                        }
                        fullMergeStrm[k][l].write(tmp);
                    }
                }
                eFullMergeStrm.write(false);
            }
            ptrR++;
            if (ptrR == LatencyR) {
                ptrR = 0;
                ptrU ^= ap_uint<1>(1);
            }
        }
        eFullMergeStrm.write(true);
    }

    void processC(hls::stream<MType> fullMergeStrm[K][LatencyB],
                  hls::stream<bool>& eFullMergeStrm,
                  ap_uint<32> cols,
                  ap_uint<32> ws,
                  hls::stream<MType> retStrm[K],
                  hls::stream<bool>& eRetStrm) {
        ap_uint<32> col_batch = (cols + D - 1) / D;
        ap_uint<32> ws_batch = (ws + K - 1) / K;

        int ptrK = 0;
        while (!eFullMergeStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 10000 max = 10000 avg = 10000
            // update ptr
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                MType ret = 0;
                for (int j = 0; j < LatencyB; j++) {
                    MType tmp = fullMergeStrm[i][j].read();
                    if (j < cols) {
                        B(ret, tmp);
                    }
                }
                B(ret, intercept[i][ptrK]);
                retStrm[i].write(C(ret));
            }
            eRetStrm.write(false);
            ptrK++;
            if (ptrK == ws_batch) {
                ptrK = 0;
            }
        }
        eRetStrm.write(true);
    }

   public:
    void initParams(ap_uint<32> cols, ap_uint<32> ws) {
        ap_uint<32> cols_batch = (cols + D - 1) / D;

        ap_uint<32> ptrK;
        ap_uint<32> ptrD;

        ptrK = 0;
        for (int i = 0; i < ws; i += K) {
#pragma HLS pipeline II = 1
            for (int j = 0; j < K; j++) {
#pragma HLS unroll
                intercept[j][ptrK] = 0;
            }
            ptrK++;
        }

        ptrK = 0;
        for (int i = 0; i < ws; i += K) {
            ptrD = 0;
            for (int j = 0; j < cols; j += D) {
#pragma HLS pipeline
                for (int k = 0; k < K; k++) {
#pragma HLS unroll
                    for (int l = 0; l < D; l++) {
#pragma HLS unroll
                        weight[k][l][ptrK * cols_batch + ptrD] = 0;
                    }
                }
                ptrD++;
            }
            ptrK++;
        }
    }

    void process(hls::stream<MType> opStrm[D],
                 hls::stream<bool>& eOpStrm,
                 hls::stream<MType> retStrm[K],
                 hls::stream<bool>& eRetStrm,
                 ap_uint<32> cols,
                 ap_uint<32> ws) {
#pragma HLS dataflow
        hls::stream<MType> rOpStrm[D];
        hls::stream<MType> partMergeStrm[K];
        hls::stream<MType> fullMergeStrm[K][LatencyB];
        hls::stream<bool> eROpStrm;
        hls::stream<bool> ePartMergeStrm;
        hls::stream<bool> eFullMergeStrm;
#pragma HLS stream variable = rOpStrm depth = localStrmDepth
#pragma HLS stream variable = partMergeStrm depth = localStrmDepth
#pragma HLS stream variable = fullMergeStrm depth = localStrmDepth
#pragma HLS stream variable = eROpStrm depth = localStrmDepth
#pragma HLS stream variable = ePartMergeStrm depth = localStrmDepth
#pragma HLS stream variable = eFullMergeStrm depth = localStrmDepth
#pragma HLS array_partition variable = rOpStrm dim = 1 complete
#pragma HLS array_partition variable = partMergeStrm dim = 1 complete
#pragma HLS array_partition variable = fullMergeStrm dim = 1 complete
#pragma HLS array_partition variable = fullMergeStrm dim = 2 complete
        processR(opStrm, eOpStrm, rOpStrm, eROpStrm, ws);
        processA(rOpStrm, eROpStrm, cols, ws, partMergeStrm, ePartMergeStrm);
        processB(partMergeStrm, ePartMergeStrm, cols, ws, fullMergeStrm, eFullMergeStrm);
        processC(fullMergeStrm, eFullMergeStrm, cols, ws, retStrm, eRetStrm);
    }

    void setWeight(MType inputW[K][D][KMax * DMax], ap_uint<32> cols, ap_uint<32> ws) {
        bool e = false;
        ap_uint<32> cols_batch = (cols + D - 1) / D;

        ap_uint<32> ptrK = 0;
        ap_uint<32> ptrTK = 0;
        ap_uint<32> ptrD = 0;
        ap_uint<32> ptrTD = 0;
        while (!e) {
#pragma HLS pipeline II = 1
            for (int i = 0; i < D; i++) {
#pragma HLS unroll
                for (int j = 0; j < K; j++) {
#pragma HLS unroll
                    weight[j][i][ptrK * cols_batch + ptrD] = inputW[j][i][ptrK * cols_batch + ptrD];
                }
            }
            // update ptr
            ptrK++;
            ptrTK += K;
            if (ptrTK >= ws) {
                ptrK = 0;
                ptrTK = 0;
                ptrD++;
                ptrTD += D;
                if (ptrTD >= cols) {
                    e = true;
                }
            }
        }
    }

    void setIntercept(MType inputI[K][KMax], ap_uint<32> ws) {
        ap_uint<32> ptrK = 0;
        for (ap_uint<32> i = 0; i < ws; i += K) {
#pragma HLS pipeline II = 1
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                intercept[i][ptrK] = inputI[i][ptrK];
            }
            ptrK++;
        }
    }

    // special case for K = 1, KMax = 1
    void setWeight(MType inputW[D][DMax], ap_uint<32> cols) {
        ap_uint<32> counter = 0;
        for (int i = 0; i < cols; i += D) {
#pragma HLS pipeline II = 1
            for (int j = 0; j < D; j++) {
#pragma HLS unroll
                weight[0][j][counter] = inputW[j][counter];
            }
            counter++;
        }
    }

    // special case for K = 1, KMax = 1
    void setIntercept(MType inputI) { intercept[0][0] = inputI; }
};

template <typename MType, int K, int KMax, void (*func)(MType& reg, MType op), int Latency>
class sl3 {
   public:
    sl3() {}

    static const int LatencyB = Latency + 5;
    static const int LatencyR = Latency + 5;

    MType callK(MType op[K], int eff) {
        MType reg = 0;
        for (int i = 0; i < K; i++) {
            if (i < eff) {
                func(reg, op[i]);
            }
        }
        return reg;
    }

    MType callB(MType op[LatencyB], int eff) {
        MType reg = 0;
        for (int i = 0; i < LatencyB; i++) {
            if (i < eff) {
                func(reg, op[i]);
            }
        }
        return reg;
    }

    void process(hls::stream<MType> opStrm[K],
                 hls::stream<bool>& eOpStrm,
                 const ap_uint<32> ws,
                 hls::stream<MType>& retStrm,
                 hls::stream<bool>& eRetStrm) {
        const ap_uint<32> ws_batch = (ws + K - 1) / K;

        MType buff[2][LatencyR][LatencyB];
#pragma HLS array_partition variable = buff dim = 1 complete

        int ptrU = 0;
        int ptrR = 0;
        int ptrB = 0;
        int ptrTR = 0;
        int ptrTB = 0;
        int ptrWS = 0;

        while (!eOpStrm.read()) {
#pragma HLS pipeline II = 1
            MType tmp[K];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                tmp[i] = opStrm[i].read();
            }
            int eff1;
            if (ptrWS + K >= ws) {
                eff1 = ws - ptrWS;
            } else {
                eff1 = K;
            }
            MType partSum = callK(tmp, eff1);
            // update buff
            MType reg;
            if (ptrU == 0) {
                reg = buff[0][ptrR][ptrB];
                if (ptrTB < LatencyB) {
                    reg = 0;
                }
            } else {
                reg = buff[1][ptrR][ptrB];
                if (ptrTB < LatencyB) {
                    reg = 0;
                }
            }
            func(reg, partSum);
            if (ptrU == 0) {
                buff[0][ptrR][ptrB] = reg;
            } else {
                buff[1][ptrR][ptrB] = reg;
            }
            // write out
            if ((ptrTB == (ws_batch - 1)) && (ptrTR >= LatencyR)) {
                MType buffTmp[LatencyB];
                if (ptrU == 0) {
                    for (int i = 0; i < LatencyB; i++) {
#pragma HLS unroll
                        buffTmp[i] = buff[1][ptrR][i];
                    }
                } else {
                    for (int i = 0; i < LatencyB; i++) {
#pragma HLS unroll
                        buffTmp[i] = buff[0][ptrR][i];
                    }
                }
#pragma HLS array_partition variable = buffTmp dim = 1 complete
                MType ret = callB(buffTmp, ws_batch);
                retStrm.write(ret);
                eRetStrm.write(false);
            }
            // update ptr
            ptrWS += K;
            if (ptrWS >= ws) {
                ptrWS = 0;
            }
            ptrB++;
            if (ptrB == LatencyB || ptrTB == ws_batch - 1) {
                ptrB = 0;
            }
            ptrTB++;
            if (ptrTB == ws_batch) {
                ptrTB = 0;
                ptrR++;
                ptrTR++;
                if (ptrR == LatencyR) {
                    ptrR = 0;
                    ptrU ^= ap_uint<1>(1);
                }
            }
        }

        int tailNum;
        if (ptrTR >= LatencyR) {
            tailNum = LatencyR;
        } else {
            tailNum = ptrTR;
            ptrU = 1;
            ptrR = 0;
        }

        for (int i = 0; i < tailNum; i++) {
            MType buffTmp[LatencyB];
#pragma HLS array_partition variable = buffTmp dim = 1 complete
            for (int i = 0; i < LatencyB; i++) {
#pragma HLS unroll
                buffTmp[i] = buff[1 - ptrU][ptrR][i];
            }
            MType ret = callB(buffTmp, ws_batch);
            retStrm.write(ret);
            eRetStrm.write(false);
            ptrR++;
            if (ptrR == LatencyR) {
                ptrR = 0;
                ptrU ^= ap_uint<1>(1);
            }
        }
        eRetStrm.write(true);
    }
};

template <typename MType, int K, int KMax, void (*B)(MType& reg, MType op), int Latency, RAMType RAMSum>
class s_aggr {
   public:
    static const int L = Latency + 5;

    MType sum[K][KMax * L];

    s_aggr() {
#pragma HLS inline
#pragma HLS array_partition variable = sum dim = 1
        if (RAMSum == URAM) {
#pragma HLS bind_storage variable = sum type = ram_2p impl = uram
        } else if (RAMSum == BRAM) {
#pragma HLS bind_storage variable = sum type = ram_2p impl = bram
        } else if (RAMSum == LUTRAM) {
#pragma HLS bind_storage variable = sum type = ram_2p impl = lutram
        }
    }

    void processAggr(hls::stream<MType> retStrm[K],
                     hls::stream<bool>& eRetStrm,
                     hls::stream<MType> aggrStrm[K],
                     hls::stream<bool>& eAggrStrm,
                     ap_uint<32> ws) {
        ap_uint<32> ptrK = 0;
        ap_uint<32> ptrTK = 0;
        ap_uint<32> ptrL = 0;
        ap_uint<32> ptrTL = 0;

        while (!eRetStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000 avg = 1000
#pragma HLS dependence variable = sum inter false
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                MType tmp = retStrm[i].read();
                MType reg = sum[i][ptrK * L + ptrL];
                if (ptrTL < L) {
                    reg = 0;
                }
                B(reg, tmp);
                sum[i][ptrK * L + ptrL] = reg;
            }
            // update ptr
            ptrK++;
            ptrTK += K;
            if (ptrTK >= ws) {
                ptrTK = 0;
                ptrK = 0;
                ptrL++;
                ptrTL++;
                if (ptrL == L) {
                    ptrL = 0;
                }
            }
        }
        ap_uint<32> batchCounter = 0;
        for (ap_uint<32> i = 0; i < ws; i += K) {
            MType tmp[K];
#pragma HLS loop_tripcount min = 5 max = 5 avg = 5
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (ap_uint<32> j = 0; j < L; j++) {
#pragma HLS pipeline
                for (ap_uint<32> k = 0; k < K; k++) {
#pragma HLS unroll
                    MType reg = tmp[k];
                    if (j == 0) {
                        reg = 0;
                    }
                    if (j < ptrTL) {
                        B(reg, sum[k][batchCounter + j]);
                    }
                    tmp[k] = reg;
                    if (j == (L - 1)) {
                        aggrStrm[k].write(reg);
                    }
                }
            }
            eAggrStrm.write(false);
            batchCounter += L;
        }
        eAggrStrm.write(true);
    }

    void processAggr(hls::stream<MType> retStrm[K], hls::stream<bool>& eRetStrm, ap_uint<32> ws) {
        ap_uint<32> ptrK = 0;
        ap_uint<32> ptrTK = 0;
        ap_uint<32> ptrL = 0;
        ap_uint<32> ptrTL = 0;

        while (!eRetStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000 avg = 1000
#pragma HLS dependence variable = sum inter false
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                MType tmp = retStrm[i].read();
                MType reg = sum[i][ptrK * L + ptrL];
                if (ptrTL < L) {
                    reg = 0;
                }
                B(reg, tmp);
                sum[i][ptrK * L + ptrL] = reg;
            }
            // update ptr
            ptrK++;
            ptrTK += K;
            if (ptrTK >= ws) {
                ptrTK = 0;
                ptrK = 0;
                ptrL++;
                ptrTL++;
                if (ptrL == L) {
                    ptrL = 0;
                }
            }
        }
        ap_uint<32> batchCounter = 0;
        ap_uint<32> resCounter = 0;
        for (ap_uint<32> i = 0; i < ws; i += K) {
            MType tmp[K];
#pragma HLS loop_tripcount min = 5 max = 5 avg = 5
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (ap_uint<32> j = 0; j < L; j++) {
#pragma HLS pipeline
                for (ap_uint<32> k = 0; k < K; k++) {
#pragma HLS unroll
                    MType reg = tmp[k];
                    if (j == 0) {
                        reg = 0;
                    }
                    if (j < ptrTL) {
                        B(reg, sum[k][batchCounter + j]);
                    }
                    tmp[k] = reg;
                    if (j == (L - 1)) {
                        sum[k][batchCounter] = reg;
                    }
                }
            }
            batchCounter += L;
            resCounter++;
        }
    }

    void processAggrAvg(hls::stream<MType> retStrm[K],
                        hls::stream<bool>& eRetStrm,
                        hls::stream<MType> aggrStrm[K],
                        hls::stream<MType> avgStrm[K],
                        hls::stream<bool>& eAggrStrm,
                        ap_uint<32> ws) {
        ap_uint<32> ptrK = 0;
        ap_uint<32> ptrTK = 0;
        ap_uint<32> ptrL = 0;
        ap_uint<32> ptrTL = 0;

        while (!eRetStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000 avg = 1000
#pragma HLS dependence variable = sum inter false
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                MType tmp = retStrm[i].read();
                MType reg = sum[i][ptrK * L + ptrL];
                if (ptrTL < L) {
                    reg = 0;
                }
                B(reg, tmp);
                sum[i][ptrK * L + ptrL] = reg;
            }
            // update ptr
            ptrK++;
            ptrTK += K;
            if (ptrTK >= ws) {
                ptrTK = 0;
                ptrK = 0;
                ptrL++;
                ptrTL++;
                if (ptrL == L) {
                    ptrL = 0;
                }
            }
        }
        ap_uint<32> batchCounter = 0;
        for (ap_uint<32> i = 0; i < ws; i += K) {
            MType tmp[K];
#pragma HLS loop_tripcount min = 5 max = 5 avg = 5
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (ap_uint<32> j = 0; j < L; j++) {
#pragma HLS pipeline
                for (ap_uint<32> k = 0; k < K; k++) {
#pragma HLS unroll
                    MType reg = tmp[k];
                    if (j == 0) {
                        reg = 0;
                    }
                    if (j < ptrTL) {
                        B(reg, sum[k][batchCounter + j]);
                    }
                    tmp[k] = reg;
                    if (j == (L - 1)) {
                        aggrStrm[k].write(reg);
                        avgStrm[k].write(reg / ptrTL);
                    }
                }
            }
            eAggrStrm.write(false);
            batchCounter += L;
        }
        eAggrStrm.write(true);
    }

    // MType sum[K][KMax * L];
    void processAggrAvg(hls::stream<MType> retStrm[K],
                        hls::stream<bool>& eRetStrm,
                        ap_uint<32> ws,
                        MType aggrRes[K][KMax],
                        MType avgRes[K][KMax]) {
        ap_uint<32> ptrK = 0;
        ap_uint<32> ptrTK = 0;
        ap_uint<32> ptrL = 0;
        ap_uint<32> ptrTL = 0;

        while (!eRetStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000 avg = 1000
#pragma HLS dependence variable = sum inter false
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                MType tmp = retStrm[i].read();
                MType reg = sum[i][ptrK * L + ptrL];
                if (ptrTL < L) {
                    reg = 0;
                }
                B(reg, tmp);
                sum[i][ptrK * L + ptrL] = reg;
            }
            // update ptr
            ptrK++;
            ptrTK += K;
            if (ptrTK >= ws) {
                ptrTK = 0;
                ptrK = 0;
                ptrL++;
                ptrTL++;
                if (ptrL == L) {
                    ptrL = 0;
                }
            }
        }
        ap_uint<32> batchCounter = 0;
        ap_uint<32> resCounter = 0;
        for (ap_uint<32> i = 0; i < ws; i += K) {
            MType tmp[K];
#pragma HLS loop_tripcount min = 5 max = 5 avg = 5
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (ap_uint<32> j = 0; j < L; j++) {
#pragma HLS pipeline
                for (ap_uint<32> k = 0; k < K; k++) {
#pragma HLS unroll
                    MType reg = tmp[k];
                    if (j == 0) {
                        reg = 0;
                    }
                    if (j < ptrTL) {
                        B(reg, sum[k][batchCounter + j]);
                    }
                    tmp[k] = reg;
                    if (j == (L - 1)) {
                        aggrRes[k][resCounter] = reg;
                        avgRes[k][resCounter] = reg / ptrTL;
                    }
                }
            }
            batchCounter += L;
            resCounter++;
        }
    }

    void processAvg(hls::stream<MType> retStrm[K],
                    hls::stream<bool>& eRetStrm,
                    hls::stream<MType> avgStrm[K],
                    hls::stream<bool>& eAvgStrm,
                    ap_uint<32> ws) {
        ap_uint<32> ptrK = 0;
        ap_uint<32> ptrTK = 0;
        ap_uint<32> ptrL = 0;
        ap_uint<32> ptrTL = 0;

        while (!eRetStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000 avg = 1000
#pragma HLS dependence variable = sum inter false
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                MType tmp = retStrm[i].read();
                MType reg = sum[i][ptrK * L + ptrL];
                if (ptrTL < L) {
                    reg = 0;
                }
                B(reg, tmp);
                sum[i][ptrK * L + ptrL] = reg;
            }
            // update ptr
            ptrK++;
            ptrTK += K;
            if (ptrTK >= ws) {
                ptrTK = 0;
                ptrK = 0;
                ptrL++;
                ptrTL++;
                if (ptrL == L) {
                    ptrL = 0;
                }
            }
        }
        ap_uint<32> batchCounter = 0;
        for (ap_uint<32> i = 0; i < ws; i += K) {
            MType tmp[K];
#pragma HLS loop_tripcount min = 5 max = 5 avg = 5
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (ap_uint<32> j = 0; j < L; j++) {
#pragma HLS pipeline
                for (ap_uint<32> k = 0; k < K; k++) {
#pragma HLS unroll
                    MType reg = tmp[k];
                    if (j == 0) {
                        reg = 0;
                    }
                    if (j < ptrTL) {
                        B(reg, sum[k][batchCounter + j]);
                    }
                    tmp[k] = reg;
                    if (j == (L - 1)) {
                        avgStrm[k].write(reg / ptrTL);
                    }
                }
            }
            eAvgStrm.write(false);
            batchCounter += L;
        }
        eAvgStrm.write(true);
    }

    void processAvg(hls::stream<MType> retStrm[K], hls::stream<bool>& eRetStrm, ap_uint<32> ws) {
        ap_uint<32> ptrK = 0;
        ap_uint<32> ptrTK = 0;
        ap_uint<32> ptrL = 0;
        ap_uint<32> ptrTL = 0;

        while (!eRetStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000 avg = 1000
#pragma HLS dependence variable = sum inter false
            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                MType tmp = retStrm[i].read();
                MType reg = sum[i][ptrK * L + ptrL];
                if (ptrTL < L) {
                    reg = 0;
                }
                B(reg, tmp);
                sum[i][ptrK * L + ptrL] = reg;
            }
            // update ptr
            ptrK++;
            ptrTK += K;
            if (ptrTK >= ws) {
                ptrTK = 0;
                ptrK = 0;
                ptrL++;
                ptrTL++;
                if (ptrL == L) {
                    ptrL = 0;
                }
            }
        }
        ap_uint<32> batchCounter = 0;
        ap_uint<32> resCounter = 0;
        for (ap_uint<32> i = 0; i < ws; i += K) {
            MType tmp[K];
#pragma HLS loop_tripcount min = 5 max = 5 avg = 5
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (ap_uint<32> j = 0; j < L; j++) {
#pragma HLS pipeline
                for (ap_uint<32> k = 0; k < K; k++) {
#pragma HLS unroll
                    MType reg = tmp[k];
                    if (j == 0) {
                        reg = 0;
                    }
                    if (j < ptrTL) {
                        B(reg, sum[k][batchCounter + j]);
                    }
                    tmp[k] = reg;
                    if (j == (L - 1)) {
                        sum[k][batchCounter] = reg / ptrTL;
                    }
                }
            }
            batchCounter += L;
            resCounter++;
        }
    }
};

template <int N, int _Depth, typename MType, typename TagType, RAMType RAMScaleBuff, RAMType RAMScaleFactor>
class scalingProcess {
   public:
    static const int L = AdditionLatency<MType>::value + 5;
    MType sum[N][_Depth * L];
    MType sqr[N][_Depth * L];
    MType scale[N][_Depth];

    scalingProcess() {
#pragma HLS inline
#pragma HLS array_partition variable = sum dim = 1 complete
#pragma HLS array_partition variable = sqr dim = 1 complete
#pragma HLS array_partition variable = scale dim = 1 complete
        if (RAMScaleBuff == URAM) {
#pragma HLS bind_storage variable = sum type = ram_2p impl = uram
#pragma HLS bind_storage variable = sqr type = ram_2p impl = uram
        } else if (RAMScaleBuff == BRAM) {
#pragma HLS bind_storage variable = sum type = ram_2p impl = bram
#pragma HLS bind_storage variable = sqr type = ram_2p impl = bram
        } else if (RAMScaleBuff == LUTRAM) {
#pragma HLS bind_storage variable = sum type = ram_2p impl = lutram
#pragma HLS bind_storage variable = sqr type = ram_2p impl = lutram
        }

        if (RAMScaleFactor == URAM) {
#pragma HLS bind_storage variable = scale type = ram_2p impl = uram
        } else if (RAMScaleFactor == BRAM) {
#pragma HLS bind_storage variable = scale type = ram_2p impl = bram
        } else if (RAMScaleFactor == LUTRAM) {
#pragma HLS bind_storage variable = scale type = ram_2p impl = lutram
        }
    }

    void scaling(hls::stream<MType> rawStrm[N],
                 hls::stream<bool>& eRawStrm,
                 hls::stream<TagType>& tagStrm,
                 hls::stream<bool>& eTagStrm,
                 const ap_uint<32> cols,
                 bool calcStdErr,
                 hls::stream<MType> outStrm[N],
                 hls::stream<bool>& eOutStrm,
                 hls::stream<TagType>& outTagStrm,
                 hls::stream<bool>& eOutTagStrm) {
        ap_uint<32> ptrC = 0;
        ap_uint<32> ptrTC = 0;
        ap_uint<32> ptrTL = 0;
        ap_uint<32> ptrR = 0;
        ap_uint<32> ptrTR = 0;

        eTagStrm.read();
        while (!eRawStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = sum inter false
#pragma HLS dependence variable = sqr inter false
            // load data
            MType dataIn[N];
#pragma HLS array_partition variable = dataIn dim = 1
            for (int i = 0; i < N; i++) {
#pragma HLS unroll
                dataIn[i] = rawStrm[i].read();
            }

            // process data based on clcStdErr
            MType dataOp[N];
#pragma HLS array_partition variable = dataOp dim = 1
            for (int i = 0; i < N; i++) {
#pragma HLS unroll
                if (calcStdErr) {
                    dataOp[i] = dataIn[i];
                } else {
                    dataOp[i] = scale[i][ptrC];
                }
            }

            MType dataOut[N];
#pragma HLS array_partition variable = dataOut dim = 1
            for (int i = 0; i < N; i++) {
#pragma HLS unroll
                dataOut[i] = dataIn[i] * dataOp[i];
            }

            if (calcStdErr) {
                for (int i = 0; i < N; i++) {
#pragma HLS unroll
                    MType tmpSum = sum[i][ptrTL + ptrR];
                    MType tmpSqr = sqr[i][ptrTL + ptrR];
                    if (ptrTR < L) {
                        tmpSum = 0;
                        tmpSqr = 0;
                    }
                    tmpSum += dataIn[i];
                    tmpSqr += dataOut[i];
                    sum[i][ptrTL + ptrR] = tmpSum;
                    sqr[i][ptrTL + ptrR] = tmpSqr;
                }
            }

            // write data
            for (int i = 0; i < N; i++) {
#pragma HLS unroll
                outStrm[i].write(dataOut[i]);
            }
            eOutStrm.write(false);
            if (ptrC == 0) {
                outTagStrm.write(tagStrm.read());
                eTagStrm.read();
                eOutTagStrm.write(false);
            }

            // update counter
            ptrC++;
            ptrTC += N;
            ptrTL += L;
            if (ptrTC >= cols) {
                ptrTC = 0;
                ptrTL = 0;
                ptrC = 0;
                ptrR++;
                ptrTR++;
                if (ptrR == L) {
                    ptrR = 0;
                }
            }
        }
        eOutStrm.write(true);
        eOutTagStrm.write(true);
        // posProcess for Scale factors
        if (calcStdErr) {
            ap_uint<32> batchCounter = 0;
            ap_uint<32> scaleCounter = 0;
            for (ap_uint<32> i = 0; i < cols; i += N) {
                MType tmpSum[N];
                MType tmpSqr[N];
#pragma HLS array_partition variable = tmpSum dim = 1 complete
#pragma HLS array_partition variable = tmpSqr dim = 1 complete
                for (ap_uint<32> j = 0; j < L; j++) {
#pragma HLS pipeline
                    for (ap_uint<32> k = 0; k < N; k++) {
#pragma HLS unroll
                        MType regSum = tmpSum[k];
                        MType regSqr = tmpSqr[k];
                        if (j == 0) {
                            regSum = 0;
                            regSqr = 0;
                        }
                        if (j < ptrTR) {
                            regSum += sum[k][batchCounter + j];
                            regSqr += sqr[k][batchCounter + j];
                        }
                        tmpSum[k] = regSum;
                        tmpSqr[k] = regSqr;
                        if (j == (L - 1)) {
                            MType tmp1 = regSqr * ptrTR;
                            MType tmp2 = regSum * regSum;
                            scale[k][scaleCounter] = ptrTR / xf::data_analytics::internal::m::sqrt(tmp1 - tmp2);
                        }
                    }
                }
                batchCounter += L;
                scaleCounter++;
            }
        }
    }

    void scaling(hls::stream<MType> rawStrm[N],
                 hls::stream<bool>& eRawStrm,
                 hls::stream<TagType>& tagStrm,
                 hls::stream<bool>& eTagStrm,
                 const ap_uint<32> cols,
                 bool calcStdErr,
                 hls::stream<MType> outStrm1[N],
                 hls::stream<bool>& eOutStrm1,
                 hls::stream<MType> outStrm2[N],
                 hls::stream<bool>& eOutStrm2,
                 hls::stream<TagType>& outTagStrm,
                 hls::stream<bool>& eOutTagStrm) {
        ap_uint<32> ptrC = 0;
        ap_uint<32> ptrTC = 0;
        ap_uint<32> ptrTL = 0;
        ap_uint<32> ptrR = 0;
        ap_uint<32> ptrTR = 0;

        eTagStrm.read();
        while (!eRawStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = sum inter false
#pragma HLS dependence variable = sqr inter false
            // load data
            MType dataIn[N];
#pragma HLS array_partition variable = dataIn dim = 1
            for (int i = 0; i < N; i++) {
#pragma HLS unroll
                dataIn[i] = rawStrm[i].read();
            }

            // process data based on clcStdErr
            MType dataOp[N];
#pragma HLS array_partition variable = dataOp dim = 1
            for (int i = 0; i < N; i++) {
#pragma HLS unroll
                if (calcStdErr) {
                    dataOp[i] = dataIn[i];
                } else {
                    dataOp[i] = scale[i][ptrC];
                }
            }

            MType dataOut[N];
#pragma HLS array_partition variable = dataOut dim = 1
            for (int i = 0; i < N; i++) {
#pragma HLS unroll
                dataOut[i] = dataIn[i] * dataOp[i];
            }

            if (calcStdErr) {
                for (int i = 0; i < N; i++) {
#pragma HLS unroll
                    MType tmpSum = sum[i][ptrTL + ptrR];
                    MType tmpSqr = sqr[i][ptrTL + ptrR];
                    if (ptrTR < L) {
                        tmpSum = 0;
                        tmpSqr = 0;
                    }
                    tmpSum += dataIn[i];
                    tmpSqr += dataOut[i];
                    sum[i][ptrTL + ptrR] = tmpSum;
                    sqr[i][ptrTL + ptrR] = tmpSqr;
                }
            }

            // write data
            for (int i = 0; i < N; i++) {
#pragma HLS unroll
                outStrm1[i].write(dataOut[i]);
                outStrm2[i].write(dataOut[i]);
            }
            eOutStrm1.write(false);
            eOutStrm2.write(false);
            if (ptrC == 0) {
                outTagStrm.write(tagStrm.read());
                eTagStrm.read();
                eOutTagStrm.write(false);
            }

            // update counter
            ptrC++;
            ptrTC += N;
            ptrTL += L;
            if (ptrTC >= cols) {
                ptrTC = 0;
                ptrTL = 0;
                ptrC = 0;
                ptrR++;
                ptrTR++;
                if (ptrR == L) {
                    ptrR = 0;
                }
            }
        }
        eOutStrm1.write(true);
        eOutStrm2.write(true);
        eOutTagStrm.write(true);
        // posProcess for Scale factors
        if (calcStdErr) {
            ap_uint<32> batchCounter = 0;
            ap_uint<32> scaleCounter = 0;
            for (ap_uint<32> i = 0; i < cols; i += N) {
                MType tmpSum[N];
                MType tmpSqr[N];
#pragma HLS array_partition variable = tmpSum dim = 1 complete
#pragma HLS array_partition variable = tmpSqr dim = 1 complete
                for (ap_uint<32> j = 0; j < L; j++) {
#pragma HLS pipeline
                    for (ap_uint<32> k = 0; k < N; k++) {
#pragma HLS unroll
                        MType regSum = tmpSum[k];
                        MType regSqr = tmpSqr[k];
                        if (j == 0) {
                            regSum = 0;
                            regSqr = 0;
                        }
                        if (j < ptrTR) {
                            regSum += sum[k][batchCounter + j];
                            regSqr += sqr[k][batchCounter + j];
                        }
                        tmpSum[k] = regSum;
                        tmpSqr[k] = regSqr;
                        if (j == (L - 1)) {
                            MType tmp1 = regSqr * ptrTR;
                            MType tmp2 = regSum * regSum;
                            scale[k][scaleCounter] = ptrTR / xf::data_analytics::internal::m::sqrt(tmp1 - tmp2);
                        }
                    }
                }
                batchCounter += L;
                scaleCounter++;
            }
        }
    }
};

template <typename MType, int K>
class pickMaxProcess {
   public:
    pickMaxProcess() {
#pragma HLS inline
    }

    static const int L = 2;
    static const int CompareLatency = L;
    static const int RowLatency = CompareLatency;

    void pickFromK(MType margin[K], ap_uint<32> counter, ap_uint<32> ws, MType& maxMargin, ap_uint<32>& maxIndex) {
        /*
        std::cout << "pickFromK: counter" << counter << " ws:" << ws << std::endl;
        for(int i = 0; i < K; i++) {
            std::cout << "margin[" << i << "]:" << margin[i] << std::endl;
        }
        */

        MType tmpMargin = margin[0];
        ap_uint<32> tmpIndex = 0;
        for (int i = 1; i < K; i++) {
#pragma HLS pipeline
            if ((counter + i) < ws) {
                if (margin[i] > tmpMargin) {
                    tmpMargin = margin[i];
                    tmpIndex = i;
                }
            }
        }
        maxMargin = tmpMargin;
        maxIndex = tmpIndex + counter;
        // std::cout << "maxMargin:" << maxMargin << "  maxIndex:" << maxIndex << std::endl << std::endl;
    }

    void pickFromL(MType margin[CompareLatency],
                   ap_uint<32> index[CompareLatency],
                   ap_uint<32> ws,
                   MType& maxMargin,
                   ap_uint<32>& maxIndex) {
        /*
        std::cout << "pickFromL: ws  " << ws << std::endl;
        for(int i = 0; i < L; i++) {
            std::cout << "margin[" << i << "]:" << margin[i] << " index:" << index[i] << std::endl << std::endl;
        }
        */
        ap_uint<32> counter = K;
        MType tmpMargin = margin[0];
        ap_uint<32> tmpIndex = index[0];
        for (int i = 1; i < CompareLatency; i++) {
#pragma HLS pipeline
            if (counter < ws) {
                if (tmpMargin < margin[i]) {
                    tmpMargin = margin[i];
                    tmpIndex = index[i];
                }
            }
            ws += K;
        }
        if (tmpMargin > 0) {
            tmpIndex++;
        } else {
            tmpMargin = 0;
            tmpIndex = 0;
        }
        maxMargin = tmpMargin;
        maxIndex = tmpIndex;
        // std::cout << "maxMargin:" << maxMargin << "  maxIndex:" << maxIndex << std::endl;
    }

    /*
        void pick(hls::stream<MType> inStrm[K],
                  hls::stream<bool>& eInStrm,
                  ap_uint<32> ws,
                  hls::stream<MType>& maxStrm,
                  hls::stream<ap_uint<32> >& maxIndexStrm,
                  hls::stream<bool>& eRetStrm) {
            MType buffMax[2][RowLatency][CompareLatency];
            MType buffIdx[2][RowLatency][CompareLatency];
    #pragma HLS array_partition variable = buffMax dim = 0 complete
    #pragma HLS array_partition variable = buffIdx dim = 0 complete
            ap_uint<1> ptrU = 0;
            ap_uint<32> ptrC = 0;
            ap_uint<32> ptrTC = 0;
            ap_uint<32> ptrR = 0;
            ap_uint<32> ptrTR = 0;
            ap_uint<32> counter = 0;
            while(!eInStrm.read()) {
    #pragma HLS dependence variable = buffMax inter false
    #pragma HLS dependence variable = buffIdx inter false
    #pragma HLS pipeline II = 1
                //load data
                MType oldMax;
                ap_uint<32> oldIdx;
                MType tmpIn[K];
    #pragma HLS array_partition variable = tmpIn dim = 1 complete
                for(int i = 0; i < K; i++) {
                    #pragma HLS unroll
                    tmpIn[i] = inStrm[i].read();
                }
                if(ptrU == 0) {
                    oldMax = buffMax[0][ptrR][ptrC];
                    oldIdx = buffIdx[0][ptrR][ptrC];
                } else {
                    oldMax = buffMax[1][ptrR][ptrC];
                    oldIdx = buffIdx[1][ptrR][ptrC];
                }
                if(ptrTC < CompareLatency) {
                    oldMax = 0;
                    oldIdx = 0;
                }
                //compute
                MType tmpMax;
                ap_uint<32> tmpIndex;
                pickFromK(tmpIn, counter, ws, tmpMax, tmpIndex);
                //update buff
                if(tmpMax < oldMax) {
                    tmpMax = oldMax;
                    tmpIndex = oldIdx;
                }
                if(ptrU == 0) {
                    buffMax[0][ptrR][ptrC] = tmpMax;
                    buffIdx[0][ptrR][ptrC] = tmpIndex;
                } else {
                    buffMax[1][ptrR][ptrC] = tmpMax;
                    buffIdx[1][ptrR][ptrC] = tmpIndex;
                }
                //write out
                if((ptrTR >= RowLatency) && (counter >= (ws - K))) {
                    MType maxChoice[CompareLatency];
                    ap_uint<32> idxChoice[CompareLatency];
    #pragma HLS array_partition variable = maxChoice dim = 1 complete
    #pragma HLS array_partition variable = idxChoice dim = 1 complete
                    if(ptrU == 0) {
                        for(int i = 0; i < CompareLatency; i++) {
                            #pragma HLS unroll
                            maxChoice[i] = buffMax[1][ptrR][i];
                            idxChoice[i] = buffIdx[1][ptrR][i];
                        }
                    } else {
                        for(int i = 0; i < CompareLatency; i++) {
                            #pragma HLS unroll
                            maxChoice[i] = buffMax[0][ptrR][i];
                            idxChoice[i] = buffIdx[0][ptrR][i];
                        }
                    }
                    MType finalMax;
                    ap_uint<32> finalIdx;
                    pickFromL(maxChoice, idxChoice, ws, finalMax, finalIdx);
                    maxStrm.write(finalMax);
                    maxIndexStrm.write(finalIdx);
                    eRetStrm.write(false)<< ptrU << std::endl;;
                }
                //update ptr
                counter++;
                ptrC++;
                ptrTC++;
                if(ptrC == CompareLatency) {
                    ptrC = 0;
                }
                if(counter >= ws) {
                    counter = 0;
                    ptrC = 0;
                    ptrTC = 0;
                    ptrR ++;
                    if(ptrR == RowLatency) {
                        ptrR = 0;
                        ptrU ^= ap_uint<1>(1);
                    }
                    ptrTR++;
                }
            }
            //write out tail
            int tailNum;
            if(ptrTR >= RowLatencyR) {
                tailNum = RowLatencyR;
            } else {
                tailNum = ptrTR;
                ptrU = 1;
                ptrR = 0;
            }
            for(int i = 0; i < tailNum; i++) {
                #pragma HLS pipeline
                MType finalMax;
                ap_uint<32> finalIdx;
                MType maxChoice[CompareLatency];
                ap_uint<32> idxChoice[CompareLatency];
    #pragma HLS array_partition variable = maxChoice dim = 1 complete
    #pragma HLS array_partition variable = idxChoice dim = 1 complete
                for(int i = 0; i < CompareLatency; i++) {
                    #pragma HLS unroll
                    maxChoice[i] = buffMax[1 - ptrU][ptrR][i];
                    idxChoice[i] = buffIdx[1 - ptrU][ptrR][i];
                }
                pickFromL(maxChoice, idxChoice, ws, finalMax, finalIdx);
                maxStrm.write(finalMax);
                maxIndexStrm.write(finalIdx);
                eRetStrm.write(false);
                ptrR++;
                if(ptrR == RowLatency) {
                    ptrR = 0;
                    ptrU ^= ap_uint<1>(1);
                }
            }
            eRetStrm.write(true);
        }
    */
    void pick(hls::stream<MType> inStrm[K],
              hls::stream<bool>& eInStrm,
              ap_uint<32> ws,
              hls::stream<ap_uint<32> >& maxIndexStrm,
              hls::stream<bool>& eRetStrm) {
        MType buffMax[2][RowLatency][CompareLatency];
        ap_uint<32> buffIdx[2][RowLatency][CompareLatency];
#pragma HLS array_partition variable = buffMax dim = 0 complete
#pragma HLS array_partition variable = buffIdx dim = 0 complete

        ap_uint<1> ptrU = 0;
        ap_uint<32> ptrC = 0;
        ap_uint<32> ptrTC = 0;
        ap_uint<32> ptrR = 0;
        ap_uint<32> ptrTR = 0;
        ap_uint<32> counter = 0;
        while (!eInStrm.read()) {
#pragma HLS dependence variable = buffMax inter false
#pragma HLS dependence variable = buffIdx inter false
#pragma HLS pipeline II = 1
            // load data
            MType oldMax;
            ap_uint<32> oldIdx;
            MType tmpIn[K];
#pragma HLS array_partition variable = tmpIn dim = 1 complete

            for (int i = 0; i < K; i++) {
#pragma HLS unroll
                tmpIn[i] = inStrm[i].read();
            }

            if (ptrU == 0) {
                oldMax = buffMax[0][ptrR][ptrC];
                oldIdx = buffIdx[0][ptrR][ptrC]; // XXX
            } else {
                oldMax = buffMax[1][ptrR][ptrC];
                oldIdx = buffIdx[1][ptrR][ptrC];
            }
            //            if(ptrTC < CompareLatency) {
            //                oldMax = 0;
            //                oldIdx = 0;
            //            }
            // compute
            MType tmpMax;
            ap_uint<32> tmpIndex;
            pickFromK(tmpIn, counter, ws, tmpMax, tmpIndex);
            // update buff
            if (tmpMax < oldMax && ptrTC >= CompareLatency) {
                tmpMax = oldMax;
                tmpIndex = oldIdx;
            }
            if (ptrU == 0) {
                buffMax[0][ptrR][ptrC] = tmpMax;
                buffIdx[0][ptrR][ptrC] = tmpIndex;
            } else {
                buffMax[1][ptrR][ptrC] = tmpMax; // XXX
                buffIdx[1][ptrR][ptrC] = tmpIndex;
            }
            // write out
            if ((ptrTR >= RowLatency) && (counter >= (ws - K))) {
                MType maxChoice[CompareLatency];
                ap_uint<32> idxChoice[CompareLatency];
#pragma HLS array_partition variable = maxChoice dim = 1 complete
#pragma HLS array_partition variable = idxChoice dim = 1 complete
                if (ptrU == 0) {
                    for (int i = 0; i < CompareLatency; i++) {
#pragma HLS unroll
                        maxChoice[i] = buffMax[1][ptrR][i];
                        idxChoice[i] = buffIdx[1][ptrR][i];
                    }
                } else {
                    for (int i = 0; i < CompareLatency; i++) {
#pragma HLS unroll
                        maxChoice[i] = buffMax[0][ptrR][i];
                        idxChoice[i] = buffIdx[0][ptrR][i];
                    }
                }

                MType finalMax;
                ap_uint<32> finalIdx;
                pickFromL(maxChoice, idxChoice, ws, finalMax, finalIdx);

                maxIndexStrm.write(finalIdx);
                eRetStrm.write(false);
            }
            // update ptr
            // std::cout << "counter:" << counter << " ptrC:" << ptrC << " ptrTC:" << ptrTC << " ptrR" << ptrR << "
            // ptrTR" << ptrTR << " ptrU"<< ptrU << std::endl;
            counter += K;
            ptrC++;
            ptrTC++;
            if (ptrC == CompareLatency) {
                ptrC = 0;
            }
            if (counter >= ws) {
                counter = 0;
                ptrC = 0;
                ptrTC = 0;
                ptrR++;
                if (ptrR == RowLatency) {
                    ptrR = 0;
                    ptrU ^= ap_uint<1>(1);
                }
                ptrTR++;
            }
        }
        // write out tail
        int tailNum;
        if (ptrTR >= RowLatency) {
            tailNum = RowLatency;
        } else {
            tailNum = ptrTR;
            ptrU = 1;
            ptrR = 0;
        }

        for (int i = 0; i < tailNum; i++) {
#pragma HLS pipeline
            MType finalMax;
            ap_uint<32> finalIdx;
            MType maxChoice[CompareLatency];
            ap_uint<32> idxChoice[CompareLatency];
#pragma HLS array_partition variable = maxChoice dim = 1 complete
#pragma HLS array_partition variable = idxChoice dim = 1 complete
            for (int i = 0; i < CompareLatency; i++) {
#pragma HLS unroll
                maxChoice[i] = buffMax[1 - ptrU][ptrR][i];
                idxChoice[i] = buffIdx[1 - ptrU][ptrR][i];
            }
            pickFromL(maxChoice, idxChoice, ws, finalMax, finalIdx);
            maxIndexStrm.write(finalIdx);
            eRetStrm.write(false);
            ptrR++;
            if (ptrR == RowLatency) {
                ptrR = 0;
                ptrU ^= ap_uint<1>(1);
            }
        }
        eRetStrm.write(true);
    }
};

} // namespace internal
} // namespace common
} // namespace data_analytics
} // namespace xf

#endif
