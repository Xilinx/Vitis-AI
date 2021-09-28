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
 * @file table_sample.hpp
 * @brief Table loading and sampling functions.
 *
 * This file is part of Vitis Data Analytics Library.
 */

#ifndef _XF_DATA_ANALYTICS_L1_TABLE_SAMPLE_HPP_
#define _XF_DATA_ANALYTICS_L1_TABLE_SAMPLE_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include <utils/x_hls_utils.h>
#include "xf_data_analytics/common/math_helper.hpp"
#include "xf_data_analytics/common/stream_local_processing.hpp"

namespace xf {
namespace data_analytics {
namespace common {
namespace internal {

template <typename MType>
union f_cast;

template <>
union f_cast<ap_uint<8> > {
    uint8_t f;
    uint8_t i;
};

template <>
union f_cast<ap_uint<32> > {
    uint32_t f;
    uint32_t i;
};

template <>
union f_cast<ap_uint<64> > {
    uint64_t f;
    uint64_t i;
};

template <>
union f_cast<double> {
    double f;
    uint64_t i;
};

template <>
union f_cast<float> {
    float f;
    uint32_t i;
};

template <int _WAxi, int _WData, int _BurstLen, typename MType>
class tableLoader {
   private:
    void readRaw(ap_uint<_WAxi>* ddr,
                 const ap_uint<32> offset,
                 const ap_uint<32> start,
                 const ap_uint<32> rows,
                 const ap_uint<32> cols,
                 hls::stream<ap_uint<_WAxi> >& vecStrm,
                 hls::stream<ap_uint<32> >& firstShiftStrm) {
        const ap_uint<64> passedElements = start * cols;
        const ap_uint<64> passedWAxi = passedElements / (_WAxi / _WData);
        const ap_uint<32> passedTail = passedElements % (_WAxi / _WData);
        const ap_uint<64> finishElements = (start + rows) * cols;
        const ap_uint<64> finishWAxi = (finishElements + ((_WAxi / _WData) - 1)) / (_WAxi / _WData);
        const ap_uint<64> nread = finishWAxi - passedWAxi;
        const ap_uint<64> realOffset = offset + passedWAxi;

        firstShiftStrm.write(passedTail);
    READ_RAW:
        for (ap_uint<64> i = 0; i < nread; i++) {
#pragma HLS loop_tripcount min = 300 max = 300 avg = 300
#pragma HLS pipeline II = 1
            vecStrm.write(ddr[realOffset + i]);
        }
    }

    void cage_shift_right(ap_uint<_WAxi * 2>& source, unsigned int s) {
#pragma HLS inline off
        if (s >= 0 && s <= _WAxi / _WData) {
            source >>= s * _WData;
        }
    }

    void varSplit(hls::stream<ap_uint<_WAxi> >& vecStrm,
                  hls::stream<ap_uint<32> >& firstShiftStrm,
                  const ap_uint<32> rows,
                  const ap_uint<32> cols,
                  hls::stream<MType> data[_WAxi / _WData],
                  hls::stream<bool>& eData) {
        const ap_uint<32> full_batch = (_WAxi / _WData);
        const ap_uint<32> tmp_tail_batch = cols % full_batch;
        const ap_uint<32> tail_batch = (tmp_tail_batch == 0) ? full_batch : tmp_tail_batch;
        const ap_uint<32> batch_num = (cols + full_batch - 1) / full_batch;
        const ap_uint<64> n_op = rows * batch_num;

        ap_uint<32> reserve = 0;
        ap_uint<_WAxi> inventory = 0;
        ap_uint<32> batch_counter = 0;

        ap_uint<32> firstShift = firstShiftStrm.read();
        if (firstShift != 0) {
            reserve = (_WAxi / _WData) - firstShift;
            inventory = vecStrm.read();
            inventory >>= firstShift * _WData;
        }
        for (ap_uint<64> i = 0; i < n_op; i++) {
#pragma HLS loop_tripcount min = 300 max = 300 avg = 300
#pragma HLS pipeline II = 1
            ap_uint<32> output;
            if (batch_counter == batch_num - 1) {
                output = tail_batch;
            } else {
                output = full_batch;
            }
            batch_counter++;
            if (batch_counter == batch_num) {
                batch_counter = 0;
            }

            ap_uint<_WAxi> new_come;
            ap_uint<32> tmp_reserve = reserve;
            if (reserve < output) {
                new_come = vecStrm.read();
                reserve += (full_batch - output);
            } else {
                new_come = 0;
                reserve -= output;
            }

            ap_uint<_WAxi* 2> cage = 0;
            cage.range(_WAxi * 2 - 1, _WAxi) = new_come;
            cage_shift_right(cage, full_batch - tmp_reserve);

            cage.range(_WAxi - 1, 0) = cage.range(_WAxi - 1, 0) ^ inventory.range(_WAxi - 1, 0);

            ap_uint<_WAxi> pre_local_output = cage.range(_WAxi - 1, 0);

            cage_shift_right(cage, output);
            inventory = cage.range(_WAxi - 1, 0);

            ap_uint<_WData> tmp[_WAxi / _WData];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int k = 0; k < full_batch; k++) {
#pragma HLS unroll
                if (k < output) {
                    tmp[k] = pre_local_output.range((k + 1) * _WData - 1, k * _WData);
                } else {
                    tmp[k] = 0;
                }
                f_cast<MType> cc;
                cc.i = tmp[k];
                data[k].write(cc.f);
            }
            eData.write(false);
        }
        eData.write(true);
    }

   public:
    void sample(ap_uint<_WAxi>* ddr,
                const ap_uint<32> offset,
                const ap_uint<32> start,
                const ap_uint<32> rows,
                const ap_uint<32> cols,
                hls::stream<MType> dataStrm[_WAxi / _WData],
                hls::stream<bool>& eDataStrm) {
        static const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<_WAxi> > vecStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
#pragma HLS bind_storage variable = vecStrm type = fifo impl = lutram
#pragma HLS stream variable = vecStrm depth = fifo_depth
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 2
#pragma HLS dataflow
        readRaw(ddr, offset, start, rows, cols, vecStrm, firstShiftStrm);
        varSplit(vecStrm, firstShiftStrm, rows, cols, dataStrm, eDataStrm);
    }
};

template <int _WAxi, int _WData, int _BurstLen, typename MType, typename TagType>
class tagTableLoader {
   private:
    void readRaw(ap_uint<_WAxi>* ddr,
                 const ap_uint<32> offset,
                 const ap_uint<32> start,
                 const ap_uint<32> rows,
                 const ap_uint<32> cols,
                 hls::stream<ap_uint<_WAxi> >& vecStrm,
                 hls::stream<ap_uint<32> >& firstShiftStrm) {
        const ap_uint<64> passedElements = start * cols;
        const ap_uint<64> passedWAxi = passedElements / (_WAxi / _WData);
        const ap_uint<32> passedTail = passedElements % (_WAxi / _WData);
        const ap_uint<64> finishElements = (start + rows) * cols;
        const ap_uint<64> finishWAxi = (finishElements + ((_WAxi / _WData) - 1)) / (_WAxi / _WData);
        const ap_uint<64> nread = finishWAxi - passedWAxi;
        const ap_uint<64> realOffset = offset + passedWAxi;

        firstShiftStrm.write(passedTail);
    READ_RAW:
        for (ap_uint<64> i = 0; i < nread; i++) {
#pragma HLS loop_tripcount min = 300 max = 300 avg = 300
#pragma HLS pipeline II = 1
            vecStrm.write(ddr[realOffset + i]);
        }
    }

    void cage_shift_right(ap_uint<_WAxi * 2>& source, unsigned int s) {
#pragma HLS inline off
        if (s >= 0 && s <= _WAxi / _WData) {
            source >>= s * _WData;
        }
    }

    void varSplit(hls::stream<ap_uint<_WAxi> >& vecStrm,
                  hls::stream<ap_uint<32> >& firstShiftStrm,
                  const ap_uint<32> rows,
                  const ap_uint<32> cols,
                  hls::stream<MType> data[_WAxi / _WData],
                  hls::stream<bool>& eData,
                  hls::stream<TagType>& tagStrm,
                  hls::stream<bool>& eTagStrm) {
        const ap_uint<32> full_batch = (_WAxi / _WData);
        const ap_uint<32> tmp_tail_batch = cols % full_batch;
        const ap_uint<32> tail_batch = (tmp_tail_batch == 0) ? full_batch : tmp_tail_batch;
        const ap_uint<32> batch_num = (cols + full_batch - 1) / full_batch;
        const ap_uint<64> n_op = rows * batch_num;

        ap_uint<32> reserve = 0;
        ap_uint<_WAxi> inventory = 0;
        ap_uint<32> batch_counter = 0;

        ap_uint<32> firstShift = firstShiftStrm.read();
        if (firstShift != 0) {
            reserve = (_WAxi / _WData) - firstShift;
            inventory = vecStrm.read();
            inventory >>= firstShift * _WData;
        }
        for (ap_uint<64> i = 0; i < n_op; i++) {
#pragma HLS loop_tripcount min = 300 max = 300 avg = 300
#pragma HLS pipeline II = 1
            ap_uint<32> output;
            if (batch_counter == batch_num - 1) {
                output = tail_batch;
            } else {
                output = full_batch;
            }

            ap_uint<_WAxi> new_come;
            ap_uint<32> tmp_reserve = reserve;
            if (reserve < output) {
                new_come = vecStrm.read();
                reserve += (full_batch - output);
            } else {
                new_come = 0;
                reserve -= output;
            }

            ap_uint<_WAxi* 2> cage = 0;
            cage.range(_WAxi * 2 - 1, _WAxi) = new_come;
            cage_shift_right(cage, full_batch - tmp_reserve);

            cage.range(_WAxi - 1, 0) = cage.range(_WAxi - 1, 0) ^ inventory.range(_WAxi - 1, 0);

            ap_uint<_WAxi> pre_local_output = cage.range(_WAxi - 1, 0);

            cage_shift_right(cage, output);
            inventory = cage.range(_WAxi - 1, 0);

            ap_uint<_WData> tmp[_WAxi / _WData];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int k = 0; k < full_batch; k++) {
#pragma HLS unroll
                if (k < output) {
                    tmp[k] = pre_local_output.range((k + 1) * _WData - 1, k * _WData);
                } else {
                    tmp[k] = 0;
                }
                f_cast<MType> cc;
                cc.i = tmp[k];
                data[k].write(cc.f);
            }
            if (batch_counter == batch_num - 1) {
                ap_uint<_WData> tmpTag = tmp[tail_batch - 1];
                f_cast<TagType> cc;
                cc.i = tmpTag;
                tagStrm.write(cc.f);
                eTagStrm.write(false);
            }
            eData.write(false);

            batch_counter++;
            if (batch_counter == batch_num) {
                batch_counter = 0;
            }
        }
        eData.write(true);
        eTagStrm.write(true);
    }

    void varSplit(hls::stream<ap_uint<_WAxi> >& vecStrm,
                  hls::stream<ap_uint<32> >& firstShiftStrm,
                  const ap_uint<32> rows,
                  const ap_uint<32> cols,
                  hls::stream<MType> data1[_WAxi / _WData],
                  hls::stream<bool>& eData1,
                  hls::stream<MType> data2[_WAxi / _WData],
                  hls::stream<bool>& eData2,
                  hls::stream<TagType>& tagStrm,
                  hls::stream<bool>& eTagStrm) {
        const ap_uint<32> full_batch = (_WAxi / _WData);
        const ap_uint<32> tmp_tail_batch = cols % full_batch;
        const ap_uint<32> tail_batch = (tmp_tail_batch == 0) ? full_batch : tmp_tail_batch;
        const ap_uint<32> batch_num = (cols + full_batch - 1) / full_batch;
        const ap_uint<64> n_op = rows * batch_num;

        ap_uint<32> reserve = 0;
        ap_uint<_WAxi> inventory = 0;
        ap_uint<32> batch_counter = 0;

        ap_uint<32> firstShift = firstShiftStrm.read();
        if (firstShift != 0) {
            reserve = (_WAxi / _WData) - firstShift;
            inventory = vecStrm.read();
            inventory >>= firstShift * _WData;
        }
        for (ap_uint<64> i = 0; i < n_op; i++) {
#pragma HLS loop_tripcount min = 300 max = 300 avg = 300
#pragma HLS pipeline II = 1
            ap_uint<32> output;
            if (batch_counter == batch_num - 1) {
                output = tail_batch;
            } else {
                output = full_batch;
            }

            ap_uint<_WAxi> new_come;
            ap_uint<32> tmp_reserve = reserve;
            if (reserve < output) {
                new_come = vecStrm.read();
                reserve += (full_batch - output);
            } else {
                new_come = 0;
                reserve -= output;
            }

            ap_uint<_WAxi* 2> cage = 0;
            cage.range(_WAxi * 2 - 1, _WAxi) = new_come;
            cage_shift_right(cage, full_batch - tmp_reserve);

            cage.range(_WAxi - 1, 0) = cage.range(_WAxi - 1, 0) ^ inventory.range(_WAxi - 1, 0);

            ap_uint<_WAxi> pre_local_output = cage.range(_WAxi - 1, 0);

            cage_shift_right(cage, output);
            inventory = cage.range(_WAxi - 1, 0);

            ap_uint<_WData> tmp[_WAxi / _WData];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int k = 0; k < full_batch; k++) {
#pragma HLS unroll
                if (k < output) {
                    tmp[k] = pre_local_output.range((k + 1) * _WData - 1, k * _WData);
                } else {
                    tmp[k] = 0;
                }
                f_cast<MType> cc;
                cc.i = tmp[k];
                data1[k].write(cc.f);
                data2[k].write(cc.f);
            }
            if (batch_counter == batch_num - 1) {
                ap_uint<_WData> tmpTag = tmp[tail_batch - 1];
                f_cast<TagType> cc;
                cc.i = tmpTag;
                tagStrm.write(cc.f);
                eTagStrm.write(false);
            }
            eData1.write(false);
            eData2.write(false);

            batch_counter++;
            if (batch_counter == batch_num) {
                batch_counter = 0;
            }
        }
        eData1.write(true);
        eData2.write(true);
        eTagStrm.write(true);
    }

   public:
    void sample(ap_uint<_WAxi>* ddr,
                const ap_uint<32> offset,
                const ap_uint<32> start,
                const ap_uint<32> rows,
                const ap_uint<32> cols,
                hls::stream<MType> dataStrm[_WAxi / _WData],
                hls::stream<bool>& eDataStrm,
                hls::stream<TagType>& tagStrm,
                hls::stream<bool>& eTagStrm) {
        static const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<_WAxi> > vecStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
#pragma HLS bind_storage variable = vecStrm type = fifo impl = lutram
#pragma HLS stream variable = vecStrm depth = fifo_depth
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 2
#pragma HLS dataflow
        readRaw(ddr, offset, start, rows, cols, vecStrm, firstShiftStrm);
        varSplit(vecStrm, firstShiftStrm, rows, cols, dataStrm, eDataStrm, tagStrm, eTagStrm);
    }

    void sample(ap_uint<_WAxi>* ddr,
                const ap_uint<32> offset,
                const ap_uint<32> start,
                const ap_uint<32> rows,
                const ap_uint<32> cols,
                hls::stream<MType> dataStrm1[_WAxi / _WData],
                hls::stream<bool>& eDataStrm1,
                hls::stream<MType> dataStrm2[_WAxi / _WData],
                hls::stream<bool>& eDataStrm2,
                hls::stream<TagType>& tagStrm,
                hls::stream<bool>& eTagStrm) {
        static const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<_WAxi> > vecStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
#pragma HLS bind_storage variable = vecStrm type = fifo impl = lutram
#pragma HLS stream variable = vecStrm depth = fifo_depth
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 2
#pragma HLS dataflow
        readRaw(ddr, offset, start, rows, cols, vecStrm, firstShiftStrm);
        varSplit(vecStrm, firstShiftStrm, rows, cols, dataStrm1, eDataStrm1, dataStrm2, eDataStrm2, tagStrm, eTagStrm);
    }
};

class MT19937 {
   private:
    /// Bit width of element in state vector
    static const int W = 32;
    /// Number of elements in state vector
    static const int N = 624;
    /// Bias in state vector to calculate new state element
    static const int M = 397;
    /// Bitmask for lower R bits
    static const int R = 31;
    /// First right shift length
    static const int U = 11;
    /// First left shift length
    static const int S = 7;
    /// Second left shift length
    static const int T = 15;
    /// Second right shift length
    static const int L = 18;
    /// Right shift length in seed initilization
    static const int S_W = 30;
    /// Address width of state vector
    static const int A_W = 10;
    /// Use 1024 depth array to hold a 624 state vector
    static const int N_W = 1024;

    /// Array to store state vector
    ap_uint<W> mt_odd_0[N_W / 2];
    /// Duplicate of array mt to solve limited memory port issue
    ap_uint<W> mt_odd_1[N_W / 2];
    /// Array to store state vector
    ap_uint<W> mt_even_0[N_W / 2];
    /// Duplicate of array mt to solve limited memory port issue
    ap_uint<W> mt_even_1[N_W / 2];
    /// Element 0 of state vector
    ap_uint<W> x_k_p_0;
    /// Element 1 of state vector
    ap_uint<W> x_k_p_1;
    /// Element 2 of state vector
    ap_uint<W> x_k_p_2;
    /// Element m of state vector
    ap_uint<W> x_k_p_m;
    /// Address of head of state vector in array mt/mt_1
    ap_uint<A_W> addr_head;

   public:
    /**
     * @brief initialize mt and mt_1 using seed
     *
     * @param seed initialization seed
     */
    void seedInitialization(ap_uint<W> seed) {
#pragma HLS inline off

        const ap_uint<W> start = 0xFFFFFFFF;
        const ap_uint<W> factor = 0x6C078965;
        const ap_uint<W> factor2 = 0x00010DCD;
        const ap_uint<W> factor3 = 0xffffffff - factor2;
        ap_uint<W * 2> tmp;
        ap_uint<W> mt_reg = seed;
        if (seed > factor3) {
            mt_reg = seed;
        } else {
            mt_reg = factor2 + seed;
        }
        mt_even_0[0] = mt_reg;
        mt_even_1[0] = mt_reg;
    SEED_INIT_LOOP:
        for (ap_uint<A_W> i = 1; i < N; i++) {
#pragma HLS pipeline II = 3
            tmp = factor * (mt_reg ^ (mt_reg >> S_W)) + i;
            mt_reg = tmp;
            ap_uint<A_W - 1> idx = i >> 1;
            if (i[0] == 0) {
                mt_even_0[idx] = tmp(W - 1, 0);
                mt_even_1[idx] = tmp(W - 1, 0);
            } else {
                mt_odd_0[idx] = tmp(W - 1, 0);
                mt_odd_1[idx] = tmp(W - 1, 0);
            }
        }
        x_k_p_0 = mt_even_0[0];
        x_k_p_1 = mt_odd_0[0];
        x_k_p_2 = mt_even_0[1];
        x_k_p_m = mt_odd_0[M / 2];
        addr_head = 0;
    }

    MT19937() {
#pragma HLS inline
#pragma HLS bind_storage variable = mt_even_0 type = ram_t2p impl = bram
#pragma HLS bind_storage variable = mt_even_1 type = ram_t2p impl = bram
#pragma HLS bind_storage variable = mt_odd_0 type = ram_t2p impl = bram
#pragma HLS bind_storage variable = mt_odd_1 type = ram_t2p impl = bram
    }

    /**
    * @brief Setup status
    *
    * @param data array to store the initialization data
    */
    void statusSetup(ap_uint<W> data[N]) {
        for (ap_uint<A_W> i = 0; i < N; i++) {
            ap_uint<A_W - 1> idx = i >> 1;
            if (i[0] == 0) {
                mt_even_0[idx] = data[i];
                mt_even_1[idx] = data[i];
            } else {
                mt_odd_0[idx] = data[i];
                mt_odd_1[idx] = data[i];
            }
        }
        // Initialize x_k_p_0, x_k_p_1 and head address of array mt
        x_k_p_0 = mt_even_0[0];
        x_k_p_1 = mt_odd_0[0];
        x_k_p_2 = mt_even_0[1];
        x_k_p_m = mt_odd_0[M / 2];
        addr_head = 0;
    }
    /**
     * @brief each call of next() generate a uniformly distributed random number
     *
     * @return a uniformly distributed random number
     */
    ap_ufixed<W, 0> next() {
#pragma HLS inline
#pragma HLS DEPENDENCE variable = mt_even_0 inter false
#pragma HLS DEPENDENCE variable = mt_even_1 inter false
#pragma HLS DEPENDENCE variable = mt_odd_0 inter false
#pragma HLS DEPENDENCE variable = mt_odd_1 inter false
        static const ap_uint<W> A = 0x9908B0DFUL;
        static const ap_uint<W> B = 0x9D2C5680UL;
        static const ap_uint<W> C = 0xEFC60000UL;
        ap_uint<W> tmp;
        ap_uint<1> tmp_0;
        ap_uint<A_W> addr_head_p_3;
        ap_uint<A_W> addr_head_p_m_p_1;
        ap_uint<A_W> addr_head_p_n;
        ap_uint<W> x_k_p_n;
        ap_uint<W> pre_result;
        ap_ufixed<W, 0> result;

        addr_head_p_3 = addr_head + 3;
        addr_head_p_m_p_1 = addr_head + M + 1;
        addr_head_p_n = addr_head + N;
        addr_head++;

        tmp(W - 1, R) = x_k_p_0(W - 1, R);
        tmp(R - 1, 0) = x_k_p_1(R - 1, 0);
        tmp_0 = tmp[0];

        tmp >>= 1;
        if (tmp_0) {
            tmp ^= A;
        }
        x_k_p_n = x_k_p_m ^ tmp;

        x_k_p_0 = x_k_p_1;
        x_k_p_1 = x_k_p_2;

        ap_uint<A_W - 1> rd_addr_0 = addr_head_p_3 >> 1;
        ap_uint<A_W - 1> rd_addr_1 = addr_head_p_m_p_1 >> 1;
        ap_uint<A_W - 1> wr_addr = addr_head_p_n >> 1;

        if (addr_head_p_3[0] == 0) {
            x_k_p_2 = mt_even_0[rd_addr_0];
            x_k_p_m = mt_odd_0[rd_addr_1];
            mt_odd_0[wr_addr] = x_k_p_n;
        } else {
            x_k_p_2 = mt_odd_0[rd_addr_0];
            x_k_p_m = mt_even_0[rd_addr_1];
            mt_even_0[wr_addr] = x_k_p_n;
        }

        pre_result = x_k_p_n;
        pre_result ^= (pre_result >> U);
        pre_result ^= (pre_result << S) & B;
        pre_result ^= (pre_result << T) & C;
        pre_result ^= (pre_result >> L);

        result(W - 1, 0) = pre_result(W - 1, 0);

        return result;
    }
};

template <int _WAxi, int _WData, int _BurstLen, typename MType>
class tableRandomLoader {
   private:
    MT19937 rng;

    void read_raw(ap_uint<_WAxi>* ddr,
                  const ap_uint<64> nread,
                  const ap_uint<64> realOffset,
                  hls::stream<ap_uint<_WAxi> >& vec_strm) {
        for (ap_uint<64> i = 0; i < nread; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 300 min = 300 avg = 300
            vec_strm.write(ddr[realOffset + i]);
        }
    }

    void cage_shift_right(ap_uint<_WAxi * 2>& source, unsigned int s) {
#pragma HLS inline off
        if (s >= 0 && s <= _WAxi / _WData) {
            source >>= s * _WData;
        }
    }

    void varSplit(hls::stream<ap_uint<_WAxi> >& vec_strm,
                  const ap_uint<32> firstShift,
                  const ap_uint<32> tail_batch,
                  const ap_uint<32> batch_num,
                  const ap_uint<64> n_op,
                  hls::stream<ap_uint<_WData> > dataStrm[_WAxi / _WData],
                  hls::stream<bool>& eDataStrm) {
        const ap_uint<32> full_batch = (_WAxi / _WData);

        ap_uint<32> reserve = 0;
        ap_uint<_WAxi> inventory = 0;
        ap_uint<32> batch_counter = 0;

        if (firstShift != 0) {
            reserve = (_WAxi / _WData) - firstShift;
            inventory = vec_strm.read();
            inventory >>= firstShift * _WData;
        }
        for (ap_uint<64> i = 0; i < n_op; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 300 min = 300 avg = 300
            ap_uint<32> output;
            if (batch_counter == batch_num - 1) {
                output = tail_batch;
            } else {
                output = full_batch;
            }
            batch_counter++;
            if (batch_counter == batch_num) {
                batch_counter = 0;
            }

            ap_uint<_WAxi> new_come;
            ap_uint<32> tmp_reserve = reserve;
            if (reserve < output) {
                new_come = vec_strm.read();
                reserve += (full_batch - output);
            } else {
                new_come = 0;
                reserve -= output;
            }

            ap_uint<_WAxi* 2> cage = 0;
            cage.range(_WAxi * 2 - 1, _WAxi) = new_come;
            cage_shift_right(cage, full_batch - tmp_reserve);

            cage.range(_WAxi - 1, 0) = cage.range(_WAxi - 1, 0) ^ inventory.range(_WAxi - 1, 0);

            ap_uint<_WAxi> pre_local_output = cage.range(_WAxi - 1, 0);

            cage_shift_right(cage, output);
            inventory = cage.range(_WAxi - 1, 0);

            for (int k = 0; k < full_batch; k++) {
#pragma HLS unroll
                ap_uint<_WData> tmp;
                if (k < output) {
                    tmp = pre_local_output.range((k + 1) * _WData - 1, k * _WData);
                } else {
                    tmp = 0;
                }
                dataStrm[k].write(tmp);
            }
            eDataStrm.write(false);
        }
    }

    void axiVarColToStreams(ap_uint<_WAxi>* ddr,
                            const ap_uint<64> nread,
                            const ap_uint<64> realOffset,
                            const ap_uint<32> firstShift,
                            const ap_uint<32> tail_batch,
                            const ap_uint<32> batch_num,
                            const ap_uint<64> n_op,
                            hls::stream<ap_uint<_WData> > dataStrm[_WAxi / _WData],
                            hls::stream<bool>& eDataStrm) {
        const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<_WAxi> > vec_strm;
#pragma HLS bind_storage variable = vec_strm type = fifo impl = lutram
#pragma HLS stream variable = vec_strm depth = fifo_depth
#pragma HLS dataflow
        read_raw(ddr, nread, realOffset, vec_strm);
        varSplit(vec_strm, firstShift, tail_batch, batch_num, n_op, dataStrm, eDataStrm);
    }

    void genRandom(const ap_uint<32> rows,
                   const ap_uint<32> cols,
                   const ap_uint<32> offset,
                   const ap_uint<32> bucketSize,
                   bool ifJump,
                   float fraction,
                   hls::stream<ap_uint<64> >& nreadStrm,
                   hls::stream<ap_uint<64> >& realOffsetStrm,
                   hls::stream<ap_uint<32> >& firstShiftStrm,
                   hls::stream<ap_uint<32> >& tailBatchStrm,
                   hls::stream<ap_uint<32> >& batchNumStrm,
                   hls::stream<ap_uint<64> >& nOpStrm,
                   hls::stream<bool>& eAxiCfgStrm,
                   hls::stream<bool>& ifDropStrm) {
        if (ifJump) {
            float log1MF = xf::data_analytics::internal::m::log(1 - fraction);

            bool e = false;
            ap_uint<32> start = 0;
            while (!e) {
#pragma HLS pipeline
#pragma HLS loop_tripcount max = 3 min = 3 avg = 3
                ap_ufixed<33, 0> ftmp = rng.next();
                ftmp[0] = 1;
                float tmp = ftmp;
                int jump = xf::data_analytics::internal::m::log(tmp) / log1MF;
                ap_uint<32> nextstart = start + jump * bucketSize;
                ap_uint<32> nextrows;

                if (nextstart < rows) {
                    if ((nextstart + bucketSize) > rows) {
                        nextrows = rows - nextstart;
                        e = true;
                    } else {
                        nextrows = bucketSize;
                    }

                    ap_uint<64> passedElements = nextstart * cols;
                    ap_uint<64> passedWAxi = passedElements / (_WAxi / _WData);
                    ap_uint<32> passedTail = passedElements % (_WAxi / _WData);
                    ap_uint<64> finishElements = (nextstart + nextrows) * cols;
                    ap_uint<64> finishWAxi = (finishElements + ((_WAxi / _WData) - 1)) / (_WAxi / _WData);
                    ap_uint<64> nread = finishWAxi - passedWAxi;
                    ap_uint<64> realOffset = offset + passedWAxi;

                    ap_uint<32> full_batch = (_WAxi / _WData);
                    ap_uint<32> tmp_tail_batch = cols % full_batch;
                    ap_uint<32> tail_batch = (tmp_tail_batch == 0) ? (full_batch) : tmp_tail_batch;
                    ap_uint<32> batch_num = (cols + full_batch - 1) / full_batch;
                    ap_uint<64> n_op = nextrows * batch_num;

                    nreadStrm.write(nread);
                    realOffsetStrm.write(realOffset);
                    firstShiftStrm.write(passedTail);
                    tailBatchStrm.write(tail_batch);
                    batchNumStrm.write(batch_num);
                    nOpStrm.write(n_op);
                    eAxiCfgStrm.write(false);
                } else {
                    e = true;
                }
                start += (1 + jump) * bucketSize;
            }
            eAxiCfgStrm.write(true);
        } else {
            ap_uint<64> finishElements = rows * cols;
            ap_uint<64> finishWAxi = (finishElements + ((_WAxi / _WData) - 1)) / (_WAxi / _WData);
            ap_uint<64> nread = finishWAxi;
            ap_uint<64> realOffset = offset;

            ap_uint<32> full_batch = (_WAxi / _WData);
            ap_uint<32> tmp_tail_batch = cols % full_batch;
            ap_uint<32> tail_batch = (tmp_tail_batch == 0) ? full_batch : tmp_tail_batch;
            ap_uint<32> batch_num = (cols + full_batch - 1) / full_batch;
            ap_uint<64> n_op = rows * batch_num;

            nreadStrm.write(nread);
            realOffsetStrm.write(realOffset);
            firstShiftStrm.write(0);
            tailBatchStrm.write(tail_batch);
            batchNumStrm.write(batch_num);
            nOpStrm.write(n_op);

            eAxiCfgStrm.write(false);
            eAxiCfgStrm.write(true);
            for (ap_uint<32> i = 0; i < rows; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 100 min = 100 avg = 100
                float tmpNext = rng.next();
                if (tmpNext < fraction) {
                    ifDropStrm.write(false);
                } else {
                    ifDropStrm.write(true);
                }
            }
        }
    }

    void jumpScan(ap_uint<_WAxi>* ddr,
                  hls::stream<ap_uint<64> >& nreadStrm,
                  hls::stream<ap_uint<64> >& realOffsetStrm,
                  hls::stream<ap_uint<32> >& firstShiftStrm,
                  hls::stream<ap_uint<32> >& tailBatchStrm,
                  hls::stream<ap_uint<32> >& batchNumStrm,
                  hls::stream<ap_uint<64> >& nOpStrm,
                  hls::stream<bool>& eAxiCfgStrm,
                  hls::stream<ap_uint<_WData> > dataStrm[_WAxi / _WData],
                  hls::stream<bool>& eDataStrm) {
        while (!eAxiCfgStrm.read()) {
#pragma HLS loop_tripcount max = 1 min = 1 avg = 1
            const ap_uint<64> nread = nreadStrm.read();
            const ap_uint<64> realOffset = realOffsetStrm.read();
            const ap_uint<32> firstShift = firstShiftStrm.read();
            const ap_uint<32> tail_batch = tailBatchStrm.read();
            const ap_uint<32> batch_num = batchNumStrm.read();
            const ap_uint<64> n_op = nOpStrm.read();
            axiVarColToStreams(ddr, nread, realOffset, firstShift, tail_batch, batch_num, n_op, dataStrm, eDataStrm);
        }
        eDataStrm.write(true);
    }

    void dropControl(const bool ifJump,
                     const ap_uint<32> cols,
                     hls::stream<bool>& ifDropStrm,
                     hls::stream<ap_uint<_WData> > inDataStrm[_WAxi / _WData],
                     hls::stream<bool>& eInDataStrm,
                     hls::stream<MType> outDataStrm[_WAxi / _WData],
                     hls::stream<bool>& eOutDataStrm) {
        ap_uint<32> colCounter = 0;
        bool drop = false;

        while (!eInDataStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 300 min = 300 avg = 300
            if (!ifJump) {
                if (colCounter == 0) {
                    drop = ifDropStrm.read();
                }
            }

            ap_uint<_WData> tmp[_WAxi / _WData];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int i = 0; i < (_WAxi / _WData); i++) {
#pragma HLS unroll
                tmp[i] = inDataStrm[i].read();
            }
            if (ifJump || !drop) {
                for (int i = 0; i < (_WAxi / _WData); i++) {
#pragma HLS unroll
                    f_cast<MType> cc;
                    cc.i = tmp[i];
                    outDataStrm[i].write(cc.f);
                }
                eOutDataStrm.write(false);
            }

            colCounter += (_WAxi / _WData);
            if (colCounter >= cols) {
                colCounter = 0;
            }
        }
        eOutDataStrm.write(true);
    }

   public:
    void seedInitialization(ap_uint<32> seed) { rng.seedInitialization(seed); }

    void sample(ap_uint<_WAxi>* ddr,
                const ap_uint<32> offset,
                const ap_uint<32> rows,
                const ap_uint<32> cols,
                const float fraction,
                const bool ifJump,
                const ap_uint<32> bucketSize,
                hls::stream<MType> dataStrm[_WAxi / _WData],
                hls::stream<bool>& eDataStrm) {
        const int fifo_depth = _BurstLen * 2;
#pragma HLS dataflow
        hls::stream<ap_uint<64> > nreadStrm;
        hls::stream<ap_uint<64> > realOffsetStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
        hls::stream<ap_uint<32> > tailBatchStrm;
        hls::stream<ap_uint<32> > batchNumStrm;
        hls::stream<ap_uint<64> > nOpStrm;
        hls::stream<bool> eAxiCfgStrm;
        hls::stream<bool> ifDropStrm;
        hls::stream<ap_uint<_WData> > interDataStrm[_WAxi / _WData];
        hls::stream<bool> eInterDataStrm;
#pragma HLS bind_storage variable = nreadStrm type = fifo impl = lutram
#pragma HLS stream variable = nreadStrm depth = 4
#pragma HLS bind_storage variable = realOffsetStrm type = fifo impl = lutram
#pragma HLS stream variable = realOffsetStrm depth = 4
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 4
#pragma HLS bind_storage variable = tailBatchStrm type = fifo impl = lutram
#pragma HLS stream variable = tailBatchStrm depth = 4
#pragma HLS bind_storage variable = batchNumStrm type = fifo impl = lutram
#pragma HLS stream variable = batchNumStrm depth = 4
#pragma HLS bind_storage variable = nOpStrm type = fifo impl = lutram
#pragma HLS stream variable = nOpStrm depth = 4
#pragma HLS bind_storage variable = eAxiCfgStrm type = fifo impl = lutram
#pragma HLS stream variable = eAxiCfgStrm depth = 4

#pragma HLS bind_storage variable = ifDropStrm type = fifo impl = lutram
#pragma HLS stream variable = ifDropStrm depth = fifo_depth
#pragma HLS bind_storage variable = interDataStrm type = fifo impl = lutram
#pragma HLS stream variable = interDataStrm depth = fifo_depth
#pragma HLS bind_storage variable = eInterDataStrm type = fifo impl = lutram
#pragma HLS stream variable = eInterDataStrm depth = fifo_depth

        genRandom(rows, cols, offset, bucketSize, ifJump, fraction, nreadStrm, realOffsetStrm, firstShiftStrm,
                  tailBatchStrm, batchNumStrm, nOpStrm, eAxiCfgStrm, ifDropStrm);
        jumpScan(ddr, nreadStrm, realOffsetStrm, firstShiftStrm, tailBatchStrm, batchNumStrm, nOpStrm, eAxiCfgStrm,
                 interDataStrm, eInterDataStrm);
        dropControl(ifJump, cols, ifDropStrm, interDataStrm, eInterDataStrm, dataStrm, eDataStrm);
    }
};

template <int _WAxi, int _WData, int _BurstLen, typename MType, typename TagType>
class tagTableRandomLoader {
   private:
    MT19937 rng;

    void read_raw(ap_uint<_WAxi>* ddr,
                  const ap_uint<64> nread,
                  const ap_uint<64> realOffset,
                  hls::stream<ap_uint<_WAxi> >& vec_strm) {
        for (ap_uint<64> i = 0; i < nread; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 300 min = 300 avg = 300
            vec_strm.write(ddr[realOffset + i]);
        }
    }

    void cage_shift_right(ap_uint<_WAxi * 2>& source, unsigned int s) {
#pragma HLS inline off
        if (s >= 0 && s <= _WAxi / _WData) {
            source >>= s * _WData;
        }
    }

    void varSplit(hls::stream<ap_uint<_WAxi> >& vec_strm,
                  const ap_uint<32> firstShift,
                  const ap_uint<32> tail_batch,
                  const ap_uint<32> batch_num,
                  const ap_uint<64> n_op,
                  hls::stream<ap_uint<_WData> > dataStrm[_WAxi / _WData],
                  hls::stream<bool>& eDataStrm) {
        const ap_uint<32> full_batch = (_WAxi / _WData);

        ap_uint<32> reserve = 0;
        ap_uint<_WAxi> inventory = 0;
        ap_uint<32> batch_counter = 0;

        if (firstShift != 0) {
            reserve = (_WAxi / _WData) - firstShift;
            inventory = vec_strm.read();
            inventory >>= firstShift * _WData;
        }
        for (ap_uint<64> i = 0; i < n_op; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 300 min = 300 avg = 300
            ap_uint<32> output;
            if (batch_counter == batch_num - 1) {
                output = tail_batch;
            } else {
                output = full_batch;
            }
            batch_counter++;
            if (batch_counter == batch_num) {
                batch_counter = 0;
            }

            ap_uint<_WAxi> new_come;
            ap_uint<32> tmp_reserve = reserve;
            if (reserve < output) {
                new_come = vec_strm.read();
                reserve += (full_batch - output);
            } else {
                new_come = 0;
                reserve -= output;
            }

            ap_uint<_WAxi* 2> cage = 0;
            cage.range(_WAxi * 2 - 1, _WAxi) = new_come;
            cage_shift_right(cage, full_batch - tmp_reserve);

            cage.range(_WAxi - 1, 0) = cage.range(_WAxi - 1, 0) ^ inventory.range(_WAxi - 1, 0);

            ap_uint<_WAxi> pre_local_output = cage.range(_WAxi - 1, 0);

            cage_shift_right(cage, output);
            inventory = cage.range(_WAxi - 1, 0);

            for (int k = 0; k < full_batch; k++) {
#pragma HLS unroll
                ap_uint<_WData> tmp;
                if (k < output) {
                    tmp = pre_local_output.range((k + 1) * _WData - 1, k * _WData);
                } else {
                    tmp = 0;
                }
                dataStrm[k].write(tmp);
            }
            eDataStrm.write(false);
        }
    }

    void axiVarColToStreams(ap_uint<_WAxi>* ddr,
                            const ap_uint<64> nread,
                            const ap_uint<64> realOffset,
                            const ap_uint<32> firstShift,
                            const ap_uint<32> tail_batch,
                            const ap_uint<32> batch_num,
                            const ap_uint<64> n_op,
                            hls::stream<ap_uint<_WData> > dataStrm[_WAxi / _WData],
                            hls::stream<bool>& eDataStrm) {
        const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<_WAxi> > vec_strm;
#pragma HLS bind_storage variable = vec_strm type = fifo impl = lutram
#pragma HLS stream variable = vec_strm depth = fifo_depth
#pragma HLS dataflow
        read_raw(ddr, nread, realOffset, vec_strm);
        varSplit(vec_strm, firstShift, tail_batch, batch_num, n_op, dataStrm, eDataStrm);
    }

    void genRandom(const ap_uint<32> rows,
                   const ap_uint<32> cols,
                   const ap_uint<32> offset,
                   const ap_uint<32> bucketSize,
                   bool ifJump,
                   float fraction,
                   hls::stream<ap_uint<64> >& nreadStrm,
                   hls::stream<ap_uint<64> >& realOffsetStrm,
                   hls::stream<ap_uint<32> >& firstShiftStrm,
                   hls::stream<ap_uint<32> >& tailBatchStrm1,
                   hls::stream<ap_uint<32> >& tailBatchStrm2,
                   hls::stream<ap_uint<32> >& batchNumStrm,
                   hls::stream<ap_uint<64> >& nOpStrm,
                   hls::stream<bool>& eAxiCfgStrm,
                   hls::stream<bool>& ifDropStrm) {
        if (ifJump) {
            float log1MF = xf::data_analytics::internal::m::log(1 - fraction);
            bool e = false;
            ap_uint<32> start = 0;
            ap_uint<32> full_batch = (_WAxi / _WData);
            ap_uint<32> tmp_tail_batch = cols % full_batch;
            ap_uint<32> tail_batch = (tmp_tail_batch == 0) ? (full_batch) : tmp_tail_batch;
            ap_uint<32> batch_num = (cols + full_batch - 1) / full_batch;
            tailBatchStrm2.write(tail_batch);

            while (!e) {
#pragma HLS pipeline
#pragma HLS loop_tripcount max = 3 min = 3 avg = 3
                ap_ufixed<33, 0> ftmp = rng.next();
                ftmp[0] = 1;
                float tmp = ftmp;
                int jump = xf::data_analytics::internal::m::log(tmp) / log1MF;
                ap_uint<32> nextstart = start + jump * bucketSize;
                ap_uint<32> nextrows;

                if (nextstart < rows) {
                    if ((nextstart + bucketSize) > rows) {
                        nextrows = rows - nextstart;
                        e = true;
                    } else {
                        nextrows = bucketSize;
                    }

                    ap_uint<64> passedElements = nextstart * cols;
                    ap_uint<64> passedWAxi = passedElements / (_WAxi / _WData);
                    ap_uint<32> passedTail = passedElements % (_WAxi / _WData);
                    ap_uint<64> finishElements = (nextstart + nextrows) * cols;
                    ap_uint<64> finishWAxi = (finishElements + ((_WAxi / _WData) - 1)) / (_WAxi / _WData);
                    ap_uint<64> nread = finishWAxi - passedWAxi;
                    ap_uint<64> realOffset = offset + passedWAxi;

                    ap_uint<64> n_op = nextrows * batch_num;

                    nreadStrm.write(nread);
                    realOffsetStrm.write(realOffset);
                    firstShiftStrm.write(passedTail);
                    tailBatchStrm1.write(tail_batch);
                    batchNumStrm.write(batch_num);
                    nOpStrm.write(n_op);
                    eAxiCfgStrm.write(false);
                } else {
                    e = true;
                }
                start += (1 + jump) * bucketSize;
            }
            eAxiCfgStrm.write(true);
        } else {
            ap_uint<64> finishElements = rows * cols;
            ap_uint<64> finishWAxi = (finishElements + ((_WAxi / _WData) - 1)) / (_WAxi / _WData);
            ap_uint<64> nread = finishWAxi;
            ap_uint<64> realOffset = offset;

            ap_uint<32> full_batch = (_WAxi / _WData);
            ap_uint<32> tmp_tail_batch = cols % full_batch;
            ap_uint<32> tail_batch = (tmp_tail_batch == 0) ? full_batch : tmp_tail_batch;
            ap_uint<32> batch_num = (cols + full_batch - 1) / full_batch;
            ap_uint<64> n_op = rows * batch_num;

            nreadStrm.write(nread);
            realOffsetStrm.write(realOffset);
            firstShiftStrm.write(0);
            tailBatchStrm1.write(tail_batch);
            tailBatchStrm2.write(tail_batch);
            batchNumStrm.write(batch_num);
            nOpStrm.write(n_op);

            eAxiCfgStrm.write(false);
            eAxiCfgStrm.write(true);
            for (ap_uint<32> i = 0; i < rows; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 100 min = 100 avg = 100
                float tmpNext = rng.next();
                if (tmpNext < fraction) {
                    ifDropStrm.write(false);
                } else {
                    ifDropStrm.write(true);
                }
            }
        }
    }

    void jumpScan(ap_uint<_WAxi>* ddr,
                  hls::stream<ap_uint<64> >& nreadStrm,
                  hls::stream<ap_uint<64> >& realOffsetStrm,
                  hls::stream<ap_uint<32> >& firstShiftStrm,
                  hls::stream<ap_uint<32> >& tailBatchStrm,
                  hls::stream<ap_uint<32> >& batchNumStrm,
                  hls::stream<ap_uint<64> >& nOpStrm,
                  hls::stream<bool>& eAxiCfgStrm,
                  hls::stream<ap_uint<_WData> > dataStrm[_WAxi / _WData],
                  hls::stream<bool>& eDataStrm) {
        while (!eAxiCfgStrm.read()) {
#pragma HLS loop_tripcount max = 1 min = 1 avg = 1
            const ap_uint<64> nread = nreadStrm.read();
            const ap_uint<64> realOffset = realOffsetStrm.read();
            const ap_uint<32> firstShift = firstShiftStrm.read();
            const ap_uint<32> tail_batch = tailBatchStrm.read();
            const ap_uint<32> batch_num = batchNumStrm.read();
            const ap_uint<64> n_op = nOpStrm.read();
            axiVarColToStreams(ddr, nread, realOffset, firstShift, tail_batch, batch_num, n_op, dataStrm, eDataStrm);
        }
        eDataStrm.write(true);
    }

    void dropControl(const bool ifJump,
                     const ap_uint<32> cols,
                     hls::stream<ap_uint<32> >& tailBatchStrm,
                     hls::stream<bool>& ifDropStrm,
                     hls::stream<ap_uint<_WData> > inDataStrm[_WAxi / _WData],
                     hls::stream<bool>& eInDataStrm,
                     hls::stream<MType> outDataStrm[_WAxi / _WData],
                     hls::stream<bool>& eOutDataStrm,
                     hls::stream<TagType>& tagStrm,
                     hls::stream<bool>& eTagStrm) {
        ap_uint<32> colCounter = 0;
        bool drop = false;
        ap_uint<32> tail_batch = tailBatchStrm.read();

        while (!eInDataStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 300 min = 300 avg = 300
            if (!ifJump) {
                if (colCounter == 0) {
                    drop = ifDropStrm.read();
                }
            }

            ap_uint<_WData> tmp[_WAxi / _WData];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int i = 0; i < (_WAxi / _WData); i++) {
#pragma HLS unroll
                tmp[i] = inDataStrm[i].read();
            }
            if (ifJump || !drop) {
                if (colCounter + 1 != cols) {
                    for (int i = 0; i < (_WAxi / _WData); i++) {
#pragma HLS unroll
                        f_cast<MType> cc;
                        cc.i = tmp[i];
                        outDataStrm[i].write(cc.f);
                    }
                    eOutDataStrm.write(false);
                }
            }

            colCounter += (_WAxi / _WData);
            if (colCounter >= cols) {
                colCounter = 0;
                if (ifJump || !drop) {
                    ap_uint<_WData> tmpTag = tmp[tail_batch - 1];
                    f_cast<TagType> cc;
                    cc.i = tmpTag;
                    tagStrm.write(cc.f);
                    eTagStrm.write(false);
                }
            }
        }
        eOutDataStrm.write(true);
        eTagStrm.write(true);
    }

    void dropControl(const bool ifJump,
                     const ap_uint<32> cols,
                     hls::stream<ap_uint<32> >& tailBatchStrm,
                     hls::stream<bool>& ifDropStrm,
                     hls::stream<ap_uint<_WData> > inDataStrm[_WAxi / _WData],
                     hls::stream<bool>& eInDataStrm,
                     hls::stream<MType> outDataStrm1[_WAxi / _WData],
                     hls::stream<bool>& eOutDataStrm1,
                     hls::stream<MType> outDataStrm2[_WAxi / _WData],
                     hls::stream<bool>& eOutDataStrm2,
                     hls::stream<TagType>& tagStrm,
                     hls::stream<bool>& eTagStrm) {
        ap_uint<32> colCounter = 0;
        bool drop = false;
        ap_uint<32> tail_batch = tailBatchStrm.read();

        while (!eInDataStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 300 min = 300 avg = 300
            if (!ifJump) {
                if (colCounter == 0) {
                    drop = ifDropStrm.read();
                }
            }

            ap_uint<_WData> tmp[_WAxi / _WData];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int i = 0; i < (_WAxi / _WData); i++) {
#pragma HLS unroll
                tmp[i] = inDataStrm[i].read();
            }
            if (ifJump || !drop) {
                if (colCounter + 1 != cols) {
                    for (int i = 0; i < (_WAxi / _WData); i++) {
#pragma HLS unroll
                        f_cast<MType> cc;
                        cc.i = tmp[i];
                        outDataStrm1[i].write(cc.f);
                        outDataStrm2[i].write(cc.f);
                    }
                    eOutDataStrm1.write(false);
                    eOutDataStrm2.write(false);
                }
            }

            colCounter += (_WAxi / _WData);
            if (colCounter >= cols) {
                colCounter = 0;
                if (ifJump || !drop) {
                    ap_uint<_WData> tmpTag = tmp[tail_batch - 1];
                    f_cast<TagType> cc;
                    cc.i = tmpTag;
                    tagStrm.write(cc.f);
                    eTagStrm.write(false);
                }
            }
        }
        eOutDataStrm1.write(true);
        eOutDataStrm2.write(true);
        eTagStrm.write(true);
    }

   public:
    tagTableRandomLoader() {
#pragma HLS inline
    }

    void seedInitialization(ap_uint<32> seed) { rng.seedInitialization(seed); }

    void sample(ap_uint<_WAxi>* ddr,
                const ap_uint<32> offset,
                const ap_uint<32> rows,
                const ap_uint<32> cols,
                const float fraction,
                const bool ifJump,
                const ap_uint<32> bucketSize,
                hls::stream<MType> dataStrm[_WAxi / _WData],
                hls::stream<bool>& eDataStrm,
                hls::stream<TagType>& tagStrm,
                hls::stream<bool>& eTagStrm) {
        const int fifo_depth = _BurstLen * 2;
#pragma HLS dataflow
        hls::stream<ap_uint<64> > nreadStrm;
        hls::stream<ap_uint<64> > realOffsetStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
        hls::stream<ap_uint<32> > tailBatchStrm1;
        hls::stream<ap_uint<32> > tailBatchStrm2;
        hls::stream<ap_uint<32> > batchNumStrm;
        hls::stream<ap_uint<64> > nOpStrm;
        hls::stream<bool> eAxiCfgStrm;
        hls::stream<bool> ifDropStrm;
        hls::stream<ap_uint<_WData> > interDataStrm[_WAxi / _WData];
        hls::stream<bool> eInterDataStrm;
#pragma HLS bind_storage variable = nreadStrm type = fifo impl = lutram
#pragma HLS stream variable = nreadStrm depth = 4
#pragma HLS bind_storage variable = realOffsetStrm type = fifo impl = lutram
#pragma HLS stream variable = realOffsetStrm depth = 4
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 4
#pragma HLS bind_storage variable = tailBatchStrm1 type = fifo impl = lutram
#pragma HLS stream variable = tailBatchStrm1 depth = 4
#pragma HLS bind_storage variable = tailBatchStrm2 type = fifo impl = lutram
#pragma HLS stream variable = tailBatchStrm2 depth = 4
#pragma HLS bind_storage variable = batchNumStrm type = fifo impl = lutram
#pragma HLS stream variable = batchNumStrm depth = 4
#pragma HLS bind_storage variable = nOpStrm type = fifo impl = lutram
#pragma HLS stream variable = nOpStrm depth = 4
#pragma HLS bind_storage variable = eAxiCfgStrm type = fifo impl = lutram
#pragma HLS stream variable = eAxiCfgStrm depth = 4

#pragma HLS bind_storage variable = ifDropStrm type = fifo impl = lutram
#pragma HLS stream variable = ifDropStrm depth = fifo_depth
#pragma HLS array_partition variable = interDataStrm dim = 1 complete
#pragma HLS bind_storage variable = interDataStrm type = fifo impl = lutram
#pragma HLS stream variable = interDataStrm depth = fifo_depth
#pragma HLS bind_storage variable = eInterDataStrm type = fifo impl = lutram
#pragma HLS stream variable = eInterDataStrm depth = fifo_depth

        genRandom(rows, cols, offset, bucketSize, ifJump, fraction, nreadStrm, realOffsetStrm, firstShiftStrm,
                  tailBatchStrm1, tailBatchStrm2, batchNumStrm, nOpStrm, eAxiCfgStrm, ifDropStrm);
        jumpScan(ddr, nreadStrm, realOffsetStrm, firstShiftStrm, tailBatchStrm1, batchNumStrm, nOpStrm, eAxiCfgStrm,
                 interDataStrm, eInterDataStrm);
        dropControl(ifJump, cols, tailBatchStrm2, ifDropStrm, interDataStrm, eInterDataStrm, dataStrm, eDataStrm,
                    tagStrm, eTagStrm);
    }

    void sample(ap_uint<_WAxi>* ddr,
                const ap_uint<32> offset,
                const ap_uint<32> rows,
                const ap_uint<32> cols,
                const float fraction,
                const bool ifJump,
                const ap_uint<32> bucketSize,
                hls::stream<MType> dataStrm1[_WAxi / _WData],
                hls::stream<bool>& eDataStrm1,
                hls::stream<MType> dataStrm2[_WAxi / _WData],
                hls::stream<bool>& eDataStrm2,
                hls::stream<TagType>& tagStrm,
                hls::stream<bool>& eTagStrm) {
        const int fifo_depth = _BurstLen * 2;
#pragma HLS dataflow
        hls::stream<ap_uint<64> > nreadStrm;
        hls::stream<ap_uint<64> > realOffsetStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
        hls::stream<ap_uint<32> > tailBatchStrm1;
        hls::stream<ap_uint<32> > tailBatchStrm2;
        hls::stream<ap_uint<32> > batchNumStrm;
        hls::stream<ap_uint<64> > nOpStrm;
        hls::stream<bool> eAxiCfgStrm;
        hls::stream<bool> ifDropStrm;
        hls::stream<ap_uint<_WData> > interDataStrm[_WAxi / _WData];
        hls::stream<bool> eInterDataStrm;
#pragma HLS bind_storage variable = nreadStrm type = fifo impl = lutram
#pragma HLS stream variable = nreadStrm depth = 4
#pragma HLS bind_storage variable = realOffsetStrm type = fifo impl = lutram
#pragma HLS stream variable = realOffsetStrm depth = 4
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 4
#pragma HLS bind_storage variable = tailBatchStrm1 type = fifo impl = lutram
#pragma HLS stream variable = tailBatchStrm1 depth = 4
#pragma HLS bind_storage variable = tailBatchStrm2 type = fifo impl = lutram
#pragma HLS stream variable = tailBatchStrm2 depth = 4
#pragma HLS bind_storage variable = batchNumStrm type = fifo impl = lutram
#pragma HLS stream variable = batchNumStrm depth = 4
#pragma HLS bind_storage variable = nOpStrm type = fifo impl = lutram
#pragma HLS stream variable = nOpStrm depth = 4
#pragma HLS bind_storage variable = eAxiCfgStrm type = fifo impl = lutram
#pragma HLS stream variable = eAxiCfgStrm depth = 4

#pragma HLS bind_storage variable = ifDropStrm type = fifo impl = lutram
#pragma HLS stream variable = ifDropStrm depth = fifo_depth
#pragma HLS array_partition variable = interDataStrm dim = 1 complete
#pragma HLS bind_storage variable = interDataStrm type = fifo impl = lutram
#pragma HLS stream variable = interDataStrm depth = fifo_depth
#pragma HLS bind_storage variable = eInterDataStrm type = fifo impl = lutram
#pragma HLS stream variable = eInterDataStrm depth = fifo_depth

        genRandom(rows, cols, offset, bucketSize, ifJump, fraction, nreadStrm, realOffsetStrm, firstShiftStrm,
                  tailBatchStrm1, tailBatchStrm2, batchNumStrm, nOpStrm, eAxiCfgStrm, ifDropStrm);
        jumpScan(ddr, nreadStrm, realOffsetStrm, firstShiftStrm, tailBatchStrm1, batchNumStrm, nOpStrm, eAxiCfgStrm,
                 interDataStrm, eInterDataStrm);
        dropControl(ifJump, cols, tailBatchStrm2, ifDropStrm, interDataStrm, eInterDataStrm, dataStrm1, eDataStrm1,
                    dataStrm2, eDataStrm2, tagStrm, eTagStrm);
    }
};

template <int W, int N, int _BurstLen>
class tupleRandomLoader {
   private:
    MT19937 rng;

   public:
    void seedInitialization(ap_uint<32> seed) { rng.seedInitialization(seed); }

    void genRandom(const ap_uint<64> nums,
                   const ap_uint<64> offset,
                   const ap_uint<32> bucketSize,
                   const bool ifJump,
                   float fraction,
                   hls::stream<ap_uint<64> >& nreadStrm,
                   hls::stream<ap_uint<64> >& realOffsetStrm,
                   hls::stream<bool>& eAxiCfgStrm,
                   hls::stream<bool>& ifDropStrm) {
        ap_uint<64> totalRead = (nums + N - 1) / N;

        if (ifJump) {
            float log1MF = xf::data_analytics::internal::m::log(1 - fraction);

            bool e = false;
            ap_uint<64> start = 0;

            while (!e) {
#pragma HLS pipeline
                ap_ufixed<33, 0> ftmp = rng.next();
                ftmp[0] = 1;
                float tmp = ftmp;
                int jump = xf::data_analytics::internal::m::log(tmp) / log1MF;

                ap_uint<64> nextOffset = start + jump * bucketSize;
                ap_uint<64> nextrows;
                ap_uint<64> realOffset = nextOffset + offset;

                if (nextOffset < totalRead) {
                    if ((nextOffset + bucketSize) > totalRead) {
                        nextrows = totalRead - nextOffset;
                        e = true;
                    } else {
                        nextrows = bucketSize;
                    }
                    nreadStrm.write(nextrows);
                    realOffsetStrm.write(realOffset);
                    eAxiCfgStrm.write(false);
                } else {
                    e = true;
                }
                start += (1 + jump) * bucketSize;
            }
            eAxiCfgStrm.write(true);
        } else {
            ap_uint<64> nread = totalRead;
            ap_uint<64> realOffset = offset;
            nreadStrm.write(nread);
            realOffsetStrm.write(realOffset);
            eAxiCfgStrm.write(false);
            eAxiCfgStrm.write(true);
            for (ap_uint<64> i = 0; i < nread; i++) {
#pragma HLS pipeline II = 1
                float tmpNext = rng.next();
                if (tmpNext < fraction) {
                    ifDropStrm.write(false);
                } else {
                    ifDropStrm.write(true);
                }
            }
        }
    }

    void readRaw(ap_uint<W * 3 * N>* ddr,
                 const ap_uint<64> nread,
                 const ap_uint<64> realOffset,
                 hls::stream<ap_uint<W> > xRawStrm[N],
                 hls::stream<ap_uint<W> > yRawStrm[N],
                 hls::stream<ap_uint<W> > zRawStrm[N],
                 hls::stream<bool>& eStrm) {
        for (ap_uint<64> i = 0; i < nread; i++) {
#pragma HLS pipeline II = 1
            ap_uint<W* 3 * N> tmp = ddr[realOffset + i];
            for (int i = 0; i < N; i++) {
#pragma HLS unroll
                ap_uint<W> x = tmp.range(i * W * 3 + W - 1, i * W * 3);
                ap_uint<W> y = tmp.range(i * W * 3 + W * 2 - 1, i * W * 3 + W);
                ap_uint<W> z = tmp.range(i * W * 3 + W * 3 - 1, i * W * 3 + W * 2);
                xRawStrm[i].write(x);
                yRawStrm[i].write(y);
                zRawStrm[i].write(z);
            }
            eStrm.write(false);
        }
    }

    void jumpScan(ap_uint<W * 3 * N>* ddr,
                  hls::stream<ap_uint<64> >& nreadStrm,
                  hls::stream<ap_uint<64> >& realOffsetStrm,
                  hls::stream<bool>& eAxiCfgStrm,
                  hls::stream<ap_uint<W> > xRawStrm[N],
                  hls::stream<ap_uint<W> > yRawStrm[N],
                  hls::stream<ap_uint<W> > zRawStrm[N],
                  hls::stream<bool>& eRawStrm) {
        while (!eAxiCfgStrm.read()) {
            const ap_uint<64> nread = nreadStrm.read();
            const ap_uint<64> realOffset = realOffsetStrm.read();
            readRaw(ddr, nread, realOffset, xRawStrm, yRawStrm, zRawStrm, eRawStrm);
        }
        eRawStrm.write(true);
    }

    void dropControl(const bool ifJump,
                     hls::stream<ap_uint<W> > xRawStrm[N],
                     hls::stream<ap_uint<W> > yRawStrm[N],
                     hls::stream<ap_uint<W> > zRawStrm[N],
                     hls::stream<bool>& eRawStrm,
                     hls::stream<bool>& ifDropStrm,
                     hls::stream<ap_uint<W> > xStrm[N],
                     hls::stream<ap_uint<W> > yStrm[N],
                     hls::stream<ap_uint<W> > zStrm[N],
                     hls::stream<bool>& eStrm) {
        while (!eRawStrm.read()) {
#pragma HLS pipeline II = 1
            bool ifDrop;
            if (!ifJump) {
                ifDrop = ifDropStrm.read();
            }

            for (int i = 0; i < N; i++) {
#pragma HLS unroll
                ap_uint<W> x = xRawStrm[i].read();
                ap_uint<W> y = yRawStrm[i].read();
                ap_uint<W> z = zRawStrm[i].read();

                if (ifJump || !ifDrop) {
                    xStrm[i].write(x);
                    yStrm[i].write(y);
                    zStrm[i].write(z);
                }
            }
            if (ifJump || !ifDrop) {
                eStrm.write(false);
            }
        }
        eStrm.write(true);
    }

    void sample(ap_uint<W * 3 * N>* ddr,
                const ap_uint<64> offset,
                const ap_uint<64> nums,
                const float fraction,
                const bool ifJump,
                const ap_uint<32> bucketSize,
                hls::stream<ap_uint<W> > xStrm[N],
                hls::stream<ap_uint<W> > yStrm[N],
                hls::stream<ap_uint<W> > zStrm[N],
                hls::stream<bool>& eStrm) {
        const int fifo_depth = _BurstLen * 2;
#pragma HLS dataflow
        hls::stream<ap_uint<64> > nreadStrm;
        hls::stream<ap_uint<64> > realOffsetStrm;
        hls::stream<bool> eAxiCfgStrm;
        hls::stream<bool> ifDropStrm;
        hls::stream<ap_uint<W> > xRawStrm[N];
        hls::stream<ap_uint<W> > yRawStrm[N];
        hls::stream<ap_uint<W> > zRawStrm[N];
        hls::stream<bool> eRawStrm;
#pragma HLS stream variable = nreadStrm depth = 4
#pragma HLS stream variable = realOffsetStrm depth = 4
#pragma HLS stream variable = eAxiCfgStrm depth = 4
#pragma HLS stream variable = ifDropStrm depth = fifo_depth
#pragma HLS stream variable = xRawStrm depth = fifo_depth
#pragma HLS stream variable = yRawStrm depth = fifo_depth
#pragma HLS stream variable = zRawStrm depth = fifo_depth
#pragma HLS stream variable = eRawStrm depth = fifo_depth
        genRandom(nums, offset, bucketSize, ifJump, fraction, nreadStrm, realOffsetStrm, eAxiCfgStrm, ifDropStrm);
        jumpScan(ddr, nreadStrm, realOffsetStrm, eAxiCfgStrm, xRawStrm, yRawStrm, zRawStrm, eRawStrm);
        dropControl(ifJump, xRawStrm, yRawStrm, zRawStrm, eRawStrm, ifDropStrm, xStrm, yStrm, zStrm, eStrm);
    }
};

template <int _WAxi, int _WData>
class splitTag {
   public:
    void split(hls::stream<ap_uint<_WData> > inStrm[_WAxi / _WData],
               hls::stream<bool>& eInStrm,
               const ap_uint<32> cols,
               hls::stream<ap_uint<_WData> > outStrm[_WAxi / _WData],
               hls::stream<ap_uint<_WData> >& tagStrm,
               hls::stream<bool>& eOutStrm) {
        ap_uint<32> ptr = cols % (_WAxi / _WData);
        if (ptr == 0) {
            ptr = _WAxi / _WData - 1;
        } else {
            ptr = ptr - 1;
        }

        ap_uint<32> colCounter = 0;
        while (!eInStrm.read()) {
#pragma HLS pipeline II = 1
            ap_uint<_WData> tmp[_WAxi / _WData];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int i = 0; i < _WAxi / _WData; i++) {
#pragma HLS unroll
                tmp[i] = inStrm[i].read();
                outStrm[i].write(tmp[i]);
            }
            eOutStrm.write(false);

            colCounter += (_WAxi / _WData);
            if (colCounter >= cols) {
                tagStrm.write(tmp[ptr]);
                colCounter = 0;
            }
        }
        eOutStrm.write(true);
    }
};

} // namespace internal
} // namespace common
} // namespace data_analytics
} // namespace xf

#endif
