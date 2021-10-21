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

#include "kernel3/build_table_encode_histo.hpp"

inline void hls_SmallestIncrement(int count, int& inc) {
    int bits = (count == 0) ? -1 : (31 ^ __builtin_clz((uint32_t)count)); // logcount
    int drop_bits = bits - ((bits + 1) >> 1);                             // GetPopulationCountPrecision(bits);
    inc = (1 << drop_bits);
}

void hls_ANSBuildInfoTable_syn(const int counts[MAX_ALPHABET_SIZE],
                               const int flat_counts[MAX_ALPHABET_SIZE],
                               bool use_flat,
                               int alphabet_size,
                               uint16_t max_nz_symbol,

                               int histogram[MAX_ALPHABET_SIZE],
                               hls_ANSEncSymbolInfo* info) {
#pragma HLS INLINE OFF

    int Fs_start = 0;

    if (use_flat) {
        _XF_IMAGE_PRINT("--5 ANSBuildInfoTable - BuildAndStoreANS\n");
        _XF_IMAGE_PRINT("--6 RewindStorage - BuildAndStoreANS\n");
        _XF_IMAGE_PRINT("--7 EncodeFlatHistogram - BuildAndStoreANS\n");
    }
    for (int s = 0; s < alphabet_size; ++s) {
#pragma HLS PIPELINE II = 1
        histogram[s] = 0; // clean to 0 for next table

        int tmp = use_flat ? flat_counts[s] : counts[s];
        const uint32_t freq = (s < max_nz_symbol) ? tmp : 0;
        info[s].freq_ = freq;
        info[s].start_ = Fs_start;
        Fs_start += freq;
        // zyl:use ifreq_
        //#ifdef USE_MULT_BY_RECIPROCAL
        if (freq != 0) {
            info[s].ifreq_ = ((1ull << hls_RECIPROCAL_PRECISION) + info[s].freq_ - 1) / info[s].freq_;
        } else {
            info[s].ifreq_ = 1; // shouldn't matter (symbol shouldn't occur), but...
        }
        //#endif
    }
}

// ------------------------------------------------------------

// ------------------------------------------------------------
// 1. count the symbol to num_symbols
// 2. jadge the max_symbol
// 3. rebalance
void CountNZSymbol(const uint32_t histogram[MAX_ALPHABET_SIZE], // output
                   uint8_t max_nz_symbol[2],
                   const uint16_t max_loop,

                   uint16_t& total,
                   int hls_counts[MAX_ALPHABET_SIZE],
                   int hls_counts2[MAX_ALPHABET_SIZE],
                   int hls_counts3[MAX_ALPHABET_SIZE],
                   int hls_countFlat[MAX_ALPHABET_SIZE],
                   int& num_symbols,                    // output
                   int scode_symbols[MAX_ALPHABET_SIZE] // output
                   ) {
#pragma HLS INLINE OFF
    // const int table_size = 1 << ANS_LOG_TAB_SIZE;  // target sum / table size
    // uint16_t total = 0;
    total = 0; // change from 64 to 32 to 16 because there is max 2^16 tockens
    int max_symbol = 0;
    int symbol_count = 0;
    const int flat_cnt = hls_ANS_TAB_SIZE / max_loop;
    const int re_add = hls_ANS_TAB_SIZE - flat_cnt * max_loop;

    // 1. test if symbol_count > precision_table_size
    // cnt = sym_cnt + 0_cnt
    // total of the all the tockens
    for (int n = 0; n < max_loop; ++n) { // 0~255
#pragma HLS PIPELINE II = 1
        total += histogram[n];
        hls_counts[n] = histogram[n];
        hls_counts2[n] = histogram[n];
        hls_counts3[n] = histogram[n];
        if (n < re_add) {
            hls_countFlat[n] = flat_cnt + 1;
        } else {
            hls_countFlat[n] = flat_cnt;
        }

        if (histogram[n] > 0) { // the front 4 non-zero cnt is record
            if (symbol_count < hls_kMaxNumSymbolsForSmallCode) {
                scode_symbols[symbol_count] = n;
            }
            ++symbol_count; // sym_cnt is non-z cnt
            max_symbol = n + 1;
            _XF_IMAGE_PRINT("--historgrams[%d] = %d\n", n, (int)histogram[n]);
        }
    }

    max_nz_symbol[0] = max_symbol;
    max_nz_symbol[1] = max_symbol;
    // count the symbol to num_symbols
    num_symbols = symbol_count;
}

void ADD_FP_strm(const int num_in,
                 hls::stream<float>& strm_in, // max 256 input

                 hls::stream<float>& strm_sum) {
#pragma HLS INLINE OFF

    ap_uint<4> idx = 0;
    float sum[16];
    const int DEP = 16;
    const int line = (num_in + 15) >> 4; //(num_in+15)/16

    if (num_in != 0) {
    INIT_ACC:
        for (int t = 0; t < DEP; t++) {
#pragma HLS UNROLL
            sum[t] = 0.0f;
        }

    CALC_ELEMENTS:
        for (int i = 0; i < line; i++) {
            for (int j = 0; j < DEP; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = DEP max = DEP
#pragma HLS DEPENDENCE variable = sum inter false
                float in = 0.0f;
                if ((i << 4) + j < num_in) {
                    in = strm_in.read();
                }
                sum[j] += in;
            }
        }

        // sum 16 data to 8
        float alpha_sum_tmp0[8] = {0};
        for (int k = 0; k < 8; k++) {
#pragma HLS PIPELINE
            alpha_sum_tmp0[k] = sum[2 * k] + sum[2 * k + 1];
        }

        // sum 8 data to 4
        float alpha_sum_tmp1[4] = {0};
        for (int k = 0; k < 4; k++) {
#pragma HLS PIPELINE
            alpha_sum_tmp1[k] = alpha_sum_tmp0[2 * k] + alpha_sum_tmp0[2 * k + 1];
        }
        // sum 4 data to 2
        float alpha_sum_tmp2[2] = {0};
        for (int k = 0; k < 2; k++) {
#pragma HLS PIPELINE
            alpha_sum_tmp2[k] = alpha_sum_tmp1[2 * k] + alpha_sum_tmp1[2 * k + 1];
        }
        // sum 2 data to 1
        float sum_out = 0.0f;
        sum_out = alpha_sum_tmp2[0] + alpha_sum_tmp2[1];

        strm_sum.write(sum_out);

    } // endif
}

void ADD_FP(const int num_in,
            float block_in[MAX_ALPHABET_SIZE], // max 256 input

            int& num_out,
            float& strm_sum) {
#pragma HLS INLINE OFF

    ap_uint<4> idx = 0;
    float sum[16];
    const int DEP = 16;
    const int line = (num_in + 15) >> 4; //(num_in+15)/16

    if (num_in != 0) {
    INIT_ACC:
        for (int t = 0; t < DEP; t++) {
#pragma HLS UNROLL
            sum[t] = 0.0f;
        }

    CALC_ELEMENTS:
        for (int i = 0; i < line; i++) {
            for (int j = 0; j < DEP; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = DEP max = DEP
#pragma HLS DEPENDENCE variable = sum inter false
                float in = 0.0f;
                if ((i << 4) + j < num_in) {
                    in = block_in[i * DEP + j];
                }
                sum[j] += in;
            }
        }

        // sum 16 data to 8
        float alpha_sum_tmp0[8] = {0};
        for (int k = 0; k < 8; k++) {
#pragma HLS PIPELINE
            alpha_sum_tmp0[k] = sum[2 * k] + sum[2 * k + 1];
        }

        // sum 8 data to 4
        float alpha_sum_tmp1[4] = {0};
        for (int k = 0; k < 4; k++) {
#pragma HLS PIPELINE
            alpha_sum_tmp1[k] = alpha_sum_tmp0[2 * k] + alpha_sum_tmp0[2 * k + 1];
        }
        // sum 4 data to 2
        float alpha_sum_tmp2[2] = {0};
        for (int k = 0; k < 2; k++) {
#pragma HLS PIPELINE
            alpha_sum_tmp2[k] = alpha_sum_tmp1[2 * k] + alpha_sum_tmp1[2 * k + 1];
        }
        // sum 2 data to 1
        float sum_out = 0.0f;
        sum_out = alpha_sum_tmp2[0] + alpha_sum_tmp2[1];

        strm_sum = sum_out;

    } // endif
    num_out = num_in;
}

void hls_Rebalance_by_sum(const float targets[MAX_ALPHABET_SIZE],
                          const int max_symbol,
                          const int table_size, // 1024
                          const int num_symbols,

                          // hls::stream<float>& strm_sum_nonrounded,
                          const float add_nround,
                          int re_cnt1[MAX_ALPHABET_SIZE],
                          int sum,

                          int& omit_pos,
                          int re_counts1[MAX_ALPHABET_SIZE],
                          int re_counts2[MAX_ALPHABET_SIZE],
                          int re_counts3[MAX_ALPHABET_SIZE],
                          bool& mode) {
#pragma HLS INLINE OFF

    if (num_symbols > 1 && (num_symbols <= hls_ANS_TAB_SIZE)) {
        float sum_nonrounded = 0.0f;
        if (sum != 0) {
            sum_nonrounded = add_nround; // strm_sum_nonrounded.read();
        }
        const float discount_ratio = (hls_ANS_TAB_SIZE - sum) / (hls_ANS_TAB_SIZE - sum_nonrounded);
        assert(discount_ratio > 0);
        assert(discount_ratio <= 1.0);
        int remainder_pos = 0; // if all of them are handled in first loop
        int remainder_log = -1;
        int count;

        // Invariant for minimize_error_of_sum == true:
        // abs(sum - sum_nonrounded)
        //   <= SmallestIncrement(max(targets[])) + max_symbol
        for (int n = 0; n < max_symbol; ++n) { // 33
#pragma HLS PIPELINE II = 1
            if (targets[n] >= 1.0) {
                // sum_nonrounded += targets[n];
                count = static_cast<uint32_t>(targets[n] * discount_ratio); // truncate

                // round
                if (count == 0) count = 1;
                if (count == table_size) count = table_size - 1;
                // Round the count to the closest nonzero multiple of SmallestIncrement
                // (when minimize_error_of_sum is false) or one of two closest so as to
                // keep the sum as close as possible to sum_nonrounded.

                int inc;
                hls_SmallestIncrement(count, inc);

                count -= count & (inc - 1);
                // TODO(robryk): Should we rescale targets[n]?

                const float target = targets[n];
                if (count == 0 || (target > count + (inc >> 1) && (count + inc < table_size))) {
                    count += inc;
                }
                sum += count;

                re_counts1[n] = count; // duplicate
                re_counts2[n] = count; // duplicate
                re_counts3[n] = count; // duplicate

                const int count_log = (31 ^ __builtin_clz((uint32_t)count)); // hls_Log2FloorNonZero_32b(re_counts1[n]);
                if (count_log > remainder_log) {
                    remainder_pos = n;
                    remainder_log = count_log;
                }

            } else {
                int tmp = re_cnt1[n];
                re_counts1[n] = tmp; // duplicate
                re_counts2[n] = tmp; // duplicate
                re_counts3[n] = tmp; // duplicate
            }
        }

        for (int j = 0; j < max_symbol; ++j) {
            _XF_IMAGE_PRINT("--historgrams_norm[%d] = %d\n", j, (int)re_counts3[j]);
        }

        assert(remainder_pos != -1);
        int tmp = re_counts1[remainder_pos] - sum + table_size;

        _XF_IMAGE_PRINT("--remainder_pos = %d, tmp=%d \n", remainder_pos, (int)tmp);

        re_counts1[remainder_pos] = tmp; // dup
        re_counts2[remainder_pos] = tmp; // dup
        re_counts3[remainder_pos] = tmp; // dup
        omit_pos = remainder_pos;
        mode = tmp > 0;

    } else {
        // strm_sum_nonrounded.read();//no read and no used
        for (int n = 0; n < max_symbol; ++n) { // 33
#pragma HLS PIPELINE II = 1
            int tmp = re_cnt1[n];
            re_counts1[n] = tmp; // dup
            re_counts2[n] = tmp; // dup
            re_counts3[n] = tmp; // dup
        }
    }
}

void hls_RebalanceHistogram_minture(const float targets[MAX_ALPHABET_SIZE],
                                    const int max_symbol,
                                    const int table_size, // 1024
                                    const int num_symbols,
                                    // const ap_uint<16> total,
                                    int sum,
                                    float sum_nonrounded,
                                    const float discount_ratio,
                                    int& omit_pos,
                                    int re_counts1[MAX_ALPHABET_SIZE],
                                    int re_counts2[MAX_ALPHABET_SIZE],
                                    int re_counts3[MAX_ALPHABET_SIZE],
                                    bool& mode) {
    if (num_symbols > 1 && (num_symbols <= hls_ANS_TAB_SIZE)) {
        const float discount_ratio = (hls_ANS_TAB_SIZE - sum) / (hls_ANS_TAB_SIZE - sum_nonrounded);
        assert(discount_ratio > 0);
        assert(discount_ratio <= 1.0);
        // the input count is reblance once to form 303 to 72, 3 to 1
        // the target[] not change but the fianl targets change to sum-sum_round.

        int remainder_pos = 0; // if all of them are handled in first loop
        int remainder_log = -1;

        // Invariant for minimize_error_of_sum == true:
        // abs(sum - sum_nonrounded)
        //   <= SmallestIncrement(max(targets[])) + max_symbol
        for (int n = 0; n < max_symbol; ++n) { // 33
#pragma HLS PIPELINE II = 1
            if (targets[n] >= 1.0) {
                sum_nonrounded += targets[n];
                re_counts1[n] = static_cast<uint32_t>(targets[n] * discount_ratio); // truncate

                // round
                if (re_counts1[n] == 0) re_counts1[n] = 1;
                if (re_counts1[n] == table_size) re_counts1[n] = table_size - 1;
                // Round the count to the closest nonzero multiple of SmallestIncrement
                // (when minimize_error_of_sum is false) or one of two closest so as to
                // keep the sum as close as possible to sum_nonrounded.

                int inc;
                hls_SmallestIncrement(re_counts1[n], inc);

                re_counts1[n] -= re_counts1[n] & (inc - 1);
                // TODO(robryk): Should we rescale targets[n]?

                const float target = (sum_nonrounded - sum);
                if (re_counts1[n] == 0 || (target > re_counts1[n] + (inc >> 1) && (re_counts1[n] + inc < table_size))) {
                    re_counts1[n] += inc;
                }
                sum += re_counts1[n];
                re_counts2[n] = re_counts1[n];
                re_counts3[n] = re_counts1[n];

                const int count_log =
                    (31 ^ __builtin_clz((uint32_t)re_counts1[n])); // hls_Log2FloorNonZero_32b(re_counts1[n]);
                if (count_log > remainder_log) {
                    remainder_pos = n;
                    remainder_log = count_log;
                }
            }
        }

        //    for (int j = 0; j < 100; ++j) {
        //    	_XF_IMAGE_PRINT("--historgrams_norm[%d] = %d\n", j,
        //    (int)re_counts1[j] );
        //    }

        assert(remainder_pos != -1);
        re_counts1[remainder_pos] -= sum - table_size;
        re_counts2[remainder_pos] = re_counts1[remainder_pos];
        re_counts3[remainder_pos] = re_counts1[remainder_pos];
        omit_pos = remainder_pos;
        mode = re_counts1[remainder_pos] > 0;

    } // endif
}

void ComputeTarget(int hls_counts[MAX_ALPHABET_SIZE],
                   uint8_t max_nz_symbol,

                   uint16_t total,
                   int& num_symbols,
                   const int scode_symbol_0,

                   int& sum,
                   // hls::stream<float> &strm_target,
                   float add_targets[MAX_ALPHABET_SIZE],
                   float targets[MAX_ALPHABET_SIZE],
                   int re_cnt1[MAX_ALPHABET_SIZE] // output
                   ) {
#pragma HLS INLINE OFF

    const float norm = 1.f * hls_ANS_TAB_SIZE / total;
    ap_uint<16> total_ap = total;
    uint16_t remd = total_ap(9, 0);
    bool is_remd = total_ap(9, 0) != 0;
    uint8_t diff = total_ap(15, 10);

    for (int n = 0; n < max_nz_symbol; ++n) { // round the <1 to 1
#pragma HLS PIPELINE II = 1

        if (num_symbols == 0) {
            re_cnt1[n] = 0;
        } else if (num_symbols == 1) {
            int tmp_counts = (n == scode_symbol_0) ? hls_ANS_TAB_SIZE : 0;
            re_cnt1[n] = tmp_counts;
        } else if (num_symbols > hls_ANS_TAB_SIZE) {
            _XF_IMAGE_PRINT("Too many entries in an ANS histogram");
        } else {
            int tmp = hls_counts[n];
            float target_tmp = norm * tmp;
            targets[n] = target_tmp;

            if (tmp != 0 && tmp < (is_remd ? (diff + 1) : diff)) {
                re_cnt1[n] = 1; // round count which is small than norm
                add_targets[sum] = target_tmp;
                sum += 1; // add to sum
            } else if (tmp == 0) {
                re_cnt1[n] = 0; // clean the ram for the 0 num in the org_count
            }
        }
    }
}

void CopyRams(uint8_t max_nz_symbol,
              float targets[MAX_ALPHABET_SIZE],
              int re_cnt[MAX_ALPHABET_SIZE],

              float targets_sub[MAX_ALPHABET_SIZE],
              int re_cnt_sub[MAX_ALPHABET_SIZE]) {
    for (int n = 0; n < max_nz_symbol; ++n) { // round the <1 to 1
#pragma HLS PIPELINE II = 1
        targets_sub[n] = targets[n];
        re_cnt_sub[n] = re_cnt[n];
    }
}

void SetTargetsWithRebalance(bool& mode, // todo
                             int hls_counts[MAX_ALPHABET_SIZE],
                             int& omit_pos,
                             uint8_t max_nz_symbol,

                             uint16_t total,
                             int& num_symbols,
                             const int scode_symbol_0,

                             int re_counts1[MAX_ALPHABET_SIZE], // output
                             int re_counts2[MAX_ALPHABET_SIZE], // output
                             int re_counts3[MAX_ALPHABET_SIZE]  // output
                             ) {
// round to ANS_TAB_SIZE==1024
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW
    // for the loop is small, no need to dataflow and pingpang

    // 1. two rebalence mechanism to rebalence the histogram

    float targets[MAX_ALPHABET_SIZE]; // hard code
    float add_targets[MAX_ALPHABET_SIZE];
#pragma HLS RESOURCE variable = add_targets core = RAM_2P_BRAM

    float add_nround;
    int sum_sub;
    //#pragma HLS ARRAY_PARTITION variable=targets complete
    int sum = 0;
    int re_cnt[MAX_ALPHABET_SIZE];        // pingpang ram
    float targets_sub[MAX_ALPHABET_SIZE]; //
    int re_cnt_sub[MAX_ALPHABET_SIZE];

    hls::stream<float> strm_target;
#pragma HLS RESOURCE variable = strm_target core = FIFO_BRAM
#pragma HLS STREAM variable = strm_target depth = 1024

    // this module is sequencial because of discount_ratio to iterate re_counts
    ComputeTarget(hls_counts, max_nz_symbol, total, num_symbols, scode_symbol_0,

                  sum, add_targets, targets, re_cnt); // strm_target,

    hls::stream<float> strm_sum_nonrounded;
#pragma HLS RESOURCE variable = strm_sum_nonrounded core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_sum_nonrounded depth = 32

    // ADD_FP(sum, strm_target, strm_sum_nonrounded);//once for one func
    ADD_FP(sum, add_targets, sum_sub, add_nround); // once for one func

    CopyRams(max_nz_symbol, targets, re_cnt, targets_sub, re_cnt_sub);

    hls_Rebalance_by_sum(targets_sub, max_nz_symbol, hls_ANS_TAB_SIZE, num_symbols, add_nround, re_cnt_sub,
                         sum_sub, // strm_sum_nonrounded,
                         omit_pos, re_counts1, re_counts2, re_counts3, mode);

    // debug
    for (int j = 0; j < max_nz_symbol; ++j) {
        _XF_IMAGE_PRINT("--re_historgrams[%d] = %d\n", j, (int)re_counts3[j]);
    }
}

void hls_StoreVarLenUint16_build(uint32_t n,
                                 // size_t* storage_ix, uint8_t* storage,
                                 int& num_bits,
                                 int& num,
                                 hls::stream<nbits_t>& strm_nbits,
                                 hls::stream<uint16_t>& strm_bits) {
#pragma HLS INLINE OFF
    if (n == 0) {
        hls_WriteBits_strm(1, 0, num_bits, num, strm_nbits, strm_bits);
    } else {
        int nbits;
        for (int i = 0; i < 3; i++) {
#pragma HLS PIPELINE
            if (i == 0) {
                hls_WriteBits_strm_nodepend(1, 1, strm_nbits, strm_bits);
                nbits = hls_Log2FloorNonZero_32b(n);
            } else if (i == 1) {
                hls_WriteBits_strm_nodepend(4, nbits, strm_nbits, strm_bits);
            } else {
                hls_WriteBits_strm_nodepend(nbits, n - (1ULL << nbits), strm_nbits, strm_bits);

                num_bits += 1 + 4 + nbits;
                num += 2 + ((nbits == 0) ? 0 : 1);
            }
        }
    }
}

// encode module ii will affected, because the use_flat flag!
void hls_EncodeCounts(const int counts[MAX_ALPHABET_SIZE],
                      const int alphabet_size,
                      const int omit_pos,
                      const int num_symbols,
                      const int symbols[MAX_ALPHABET_SIZE],
                      const uint8_t max_nz_symbol,
                      bool do_encode,

                      int& histo_bits,
                      int& num,
                      hls::stream<nbits_t>& strm_nbits,
                      hls::stream<uint16_t>& strm_bits) { // encode module ii is not affected!

#pragma HLS INLINE OFF

    histo_bits = 0;
    num = 0;
    int same[MAX_ALPHABET_SIZE];
#pragma HLS RESOURCE variable = same core = RAM_2P_LUTRAM

    if (do_encode) {
        _XF_IMAGE_PRINT("--tree size(num_symbols) = %d\n", (int)num_symbols);
        if (num_symbols <= 2) { // Small tree

            hls_WriteBits_strm(1, 1, histo_bits, num, strm_nbits, strm_bits); // Small tree marker to encode 1-2
                                                                              // symbols.

            if (num_symbols == 0) {
                hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
                hls_StoreVarLenUint16_build(0, histo_bits, num, strm_nbits, strm_bits);

            } else {
                hls_WriteBits_strm(1, num_symbols - 1, histo_bits, num, strm_nbits, strm_bits);
                for (int i = 0; i < num_symbols; ++i) {
                    // hls_StoreVarLenUint16_build(symbols[i], histo_bits,  num,
                    // strm_nbits, strm_bits);
                    if (symbols[i] == 0) {
                        hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
                    } else {
                        hls_StoreVarLenUint16_build(symbols[i], histo_bits, num, strm_nbits, strm_bits);
                    }
                }
            }
            if (num_symbols == 2) {
                hls_WriteBits_strm(hls_ANS_LOG_TAB_SIZE, counts[symbols[0]], histo_bits, num, strm_nbits, strm_bits);
            }

        } else { // non-small tree
                 //		  ---W--- n_bits=1, bits=0
                 //		  ---W--- n_bits=1, bits=0
                 // hls_StoreVarLenUint16_build
            //		  ---W--- n_bits=1, bits=1
            //		  ---W--- n_bits=4, bits=0  =
            // hls_Log2FloorNonZero_32b(max_nz_symbol - 3)
            //		  ---W--- n_bits=3, bits=1  , pos=908

            //		  ---W--- n_bits=3, bits=1, pos=911
            //		  ---W--- n_bits=3, bits=2
            //		  ---W--- n_bits=4, bits=3
            //		  ---W--- n_bits=3, bits=1
            //		  ---W--- n_bits=2, bits=0
            //		  ---W--- n_bits=1, bits=0
            //		  ---W--- n_bits=1, bits=1
            //		  ---W--- n_bits=10, bits=256

            // Mark non-small tree.
            hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
            // Mark non-flat histogram.
            hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
            // Since num_symbols >= 3, we know that length >= 3, therefore we encode
            // length - 3.
            _XF_IMAGE_PRINT("max_nz_symbol = %d\n", max_nz_symbol);
            hls_StoreVarLenUint16_build(max_nz_symbol - 3, histo_bits, num, strm_nbits, strm_bits);

            // todo Merge the first two loop to syn
            // ------------------------------------------------------------
            // init a RAM
            // Precompute sequences for RLE encoding. Contains the number of identical
            // values starting at a given index. Only contains the value at the first
            // element of the series.
            // std::vector<int> same(alphabet_size, 0);

            same[0] = 0;
            same[1] = 0;
            //    for(int i = 0; i < max_nz_symbol; i++){
            //    	same[i] = 0;
            //    }
            //    i = 1, 2, 3, 4, 5, 6, 7
            // last = 0, 0, 3, 4, 4, 6, 7
            int last = 0;
            for (int i = 1; i < max_nz_symbol; i++) {
#pragma HLS DEPENDENCE variable = same inter false
#pragma HLS PIPELINE
                // Store the sequence length once different symbol reached, or we're at
                // the end, or the length is longer than we can encode, or we are at
                // the omit_pos. We don't support including the omit_pos in an RLE
                // sequence because this value may use a different amoung of log2 bits
                // than standard, it is too complex to handle in the decoder.

                same[i] = 0; // two port bram
                if (counts[i] != counts[last] || i + 1 == alphabet_size || (i - last) >= 255 || i == omit_pos ||
                    i == omit_pos + 1) {
                    same[last] = (i - last);
                    last = i + 1;
                }
            }

            for (int j = 0; j < max_nz_symbol; ++j) {
                _XF_IMAGE_PRINT("--same[%d] = %d\n", j, (int)same[j]);
            }

            // ------------------------------------------------------------
            // init a RAM
            uint8_t logcounts[MAX_ALPHABET_SIZE];
#pragma HLS RESOURCE variable = logcounts core = RAM_2P_LUTRAM
            uint8_t omit_log = 0;

            for (int i = 0; i < max_nz_symbol; ++i) {
#pragma HLS PIPELINE II = 1
                assert(counts[i] <= hls_ANS_TAB_SIZE);
                assert(counts[i] >= 0);
                if (i == omit_pos) {
                } else if (counts[i] > 0) {
                    logcounts[i] = (31 ^ __builtin_clz((uint32_t)counts[i])) + 1;
                    if (i < omit_pos) {
                        omit_log = (omit_log > logcounts[i] + 1) ? omit_log : (logcounts[i] + 1);
                    } else {
                        omit_log = (omit_log > logcounts[i]) ? omit_log : (logcounts[i]);
                    }
                } else {
                    logcounts[i] = 0;
                }
            }
            logcounts[omit_pos] = omit_log;

            // The logcount values are encoded with a static Huffman code.
            static const int kMinReps = 4;
            int rep = hls_ANS_LOG_TAB_SIZE + 1;

        logcount_loop1:
            for (int i = 0; i < max_nz_symbol; ++i) {
#pragma HLS PIPELINE II = 4
                if (i > 0 && same[i - 1] > kMinReps) {
                    // Encode the RLE symbol and skip the repeated ones.
                    hls_WriteBits_strm(hls_kLogCountBitLengths[rep], hls_kLogCountSymbols[rep],
                                       // pos,storage,
                                       histo_bits, num, strm_nbits, strm_bits);
                    hls_WriteBits_strm(8, same[i - 1], histo_bits, num, strm_nbits, strm_bits);
                    i += same[i - 1] - 2;

                } else {
                    hls_WriteBits_strm(hls_kLogCountBitLengths[logcounts[i]], hls_kLogCountSymbols[logcounts[i]],
                                       histo_bits, num, strm_nbits, strm_bits);
                }
            }

        logcount_loop2:
            for (int i = 0; i < max_nz_symbol; ++i) {
#pragma HLS PIPELINE II = 3
                if (i > 0 && same[i - 1] > kMinReps) {
                    // Skip symbols encoded by RLE.
                    i += same[i - 1] - 2;

                } else if (logcounts[i] > 1 && i != omit_pos) {
                    int bitcount = (logcounts[i]) >> 1;
                    int drop_bits = logcounts[i] - 1 - bitcount;
                    hls_WriteBits_strm(bitcount, (counts[i] >> drop_bits) - (1 << bitcount), histo_bits, num,
                                       strm_nbits, strm_bits);
                }
            }
        } // end num_symbol if
    }     // end pos if
}

void hls_EstimateDataBits( // debug
    bool do_estimate,
    const int* histogram,
    const int counts[MAX_ALPHABET_SIZE],
    const short num_symbol,
    const short len, // alphabet_size
    int& Estimate) {
#pragma HLS INLINE OFF

    float sum = 0.0f;
    int total_histogram = 0;
    int total_counts = 0;
    Estimate = 0;

    int num = 0;

    hls::stream<float> strm_fsub;
#pragma HLS RESOURCE variable = strm_fsub core = FIFO_BRAM
#pragma HLS STREAM variable = strm_fsub depth = 1024
    hls::stream<float> strm_sum_nonrounded;
#pragma HLS RESOURCE variable = strm_sum_nonrounded core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_sum_nonrounded depth = 32

    if (do_estimate) {
        _XF_IMAGE_PRINT("--4 EstimateDataBits - BuildAndStoreANS\n");

#pragma HLS DATAFLOW

        for (int i = 0; i < len; ++i) { // alphabet_size
#pragma HLS PIPELINE II = 1

            if (num_symbol == 0) {
                //}else if (num_symbol == 1){

            } else {
                total_histogram += histogram[i]; // already have 4096
                total_counts += counts[i];       // already have 1024, rebalance from 4096 to 1024

                if (histogram[i] > 0) {
                    assert(counts[i] > 0);
                    float tmp = hls_FastLog2(counts[i]); //
                    float fsub = histogram[i] * tmp;
                    strm_fsub.write((-1 * fsub));

                    num++;
                }
            }
        }

        ADD_FP_strm(num, strm_fsub, strm_sum_nonrounded);

        for (int i = 0; i < 1; ++i) {
#pragma HLS PIPELINE II = 1
            if (num != 0) {
                sum = strm_sum_nonrounded.read();
            }

            if (num_symbol == 1) total_counts = hls_ANS_TAB_SIZE;

            _XF_IMAGE_PRINT("-- total_histogram = %d, max_symb=%d, sum=%f\n", total_histogram, len, sum);
            if (total_histogram > 0) {
                assert(total_counts == hls_ANS_TAB_SIZE);
                sum += total_histogram * hls_ANS_LOG_TAB_SIZE;
            }
            Estimate = static_cast<int>(sum + 1.0f);
        }
    } // endif
}

inline int hls_EstimateDataBitsFlat(const short len, const uint16_t total_histogram) {
    const float flat_bits = hls_FastLog2(len); // 8
    return static_cast<int>(total_histogram * flat_bits + 1.0);
}

void hls_WriteBitStreamWithConsume(hls::stream<int>& strm_num_pair,
                                   hls::stream<nbits_t>& strm_nbits1,
                                   hls::stream<uint16_t>& strm_bits1,

                                   hls::stream<bool>& strm_use_flat,
                                   hls::stream<int>& strm_num_fpair,
                                   hls::stream<nbits_t>& strm_nbits2,
                                   hls::stream<uint16_t>& strm_bits2,

                                   int& pos,
                                   uint8_t& byte_tail,
                                   hls::stream<uint8_t>& strm_byte,
                                   hls::stream<bool>& strm_histo_e) {
#pragma HLS INLINE OFF

    uint8_t ntail = pos & 7;
    uint8_t n_byte = 0; // n bytes to be write out
    nbits_t pair_nbits = 0;
    uint16_t pair_bits;
    nbits_t nbits1 = 0;
    uint16_t bits1;
    nbits_t nbits2 = 0;
    uint16_t bits2;
    ap_uint<32> buffer = byte_tail;
    int cnt_pair = 0;
    int cnt_cnsm = 0;
    int cnt_max = 0;

    // init
    int num_pair1 = strm_num_pair.read();
    int num_pair2 = strm_num_fpair.read();
    const bool use_pair2 = strm_use_flat.read();

    _XF_IMAGE_PRINT("--byte_tail = %d , pos=%d\n", byte_tail, pos);
    _XF_IMAGE_PRINT("--num_pair = %d , num_fpair=%d\n", num_pair1, num_pair2);

    int num_pair = use_pair2 ? num_pair2 : num_pair1;
    int num_cnsm = use_pair2 ? num_pair1 : num_pair2;
    int num_max = (num_pair1 > num_pair2) ? num_pair1 : num_pair2;

    while (cnt_max < num_max + 1) { // loopn and tail// ii=2 is not affected
#pragma HLS PIPELINE II = 1

        if (n_byte == 0) { // update num to write

            if (cnt_max < num_pair1) {
                nbits1 = strm_nbits1.read();
                bits1 = strm_bits1.read();
            }
            if (cnt_max < num_pair2) {
                nbits2 = strm_nbits2.read();
                bits2 = strm_bits2.read();
            }

            if (use_pair2) {
                pair_nbits = nbits2;
                pair_bits = bits2;
            } else {
                pair_nbits = nbits1;
                pair_bits = bits1;
            }

            if (cnt_pair < num_pair) {
                pos += pair_nbits;
                n_byte = (ntail + pair_nbits) >> 3;
                buffer(ntail + 16, ntail) = pair_bits;
                ntail = (ntail + pair_nbits) & 7;
            }

            // n_byte = (ntail+pair_nbits)>>3;
            cnt_pair++; // end here
            cnt_max++;

        } else { // write out

            uint8_t byte = buffer(7, 0);

            buffer = buffer >> 8;
            strm_byte.write(byte);
            strm_histo_e.write(false);
            // num_byte++;
            n_byte--;
        }
    } // end while

    byte_tail = buffer(7, 0);
}

inline void hls_EncodeFlatHistogram(const int alphabet_size,
                                    bool do_encode,
                                    int& num_fpair,
                                    hls::stream<nbits_t>& strm_flat_nbits,
                                    hls::stream<uint16_t>& strm_flat_bits) {
#pragma HLS INLINE OFF
    int tmp = 0;
    num_fpair = 0;
    if (do_encode) {
        // Mark non-small tree.
        hls_WriteBits_strm(1, 0, tmp, num_fpair, strm_flat_nbits, strm_flat_bits);
        // Mark uniform histogram.
        hls_WriteBits_strm(1, 1, tmp, num_fpair, strm_flat_nbits, strm_flat_bits);
        // Encode alphabet size.
        hls_WriteBits_strm(hls_ANS_LOG_TAB_SIZE, alphabet_size, tmp, num_fpair, strm_flat_nbits, strm_flat_bits);
    }
}

// ------------------------------------------------------------
void XAcc_BuildAndStoreANSEncodingData3(int histogram1[MAX_ALPHABET_SIZE],
                                        int histogram2[MAX_ALPHABET_SIZE],
                                        // const uint32_t histogram2[MAX_ALPHABET_SIZE],
                                        const uint16_t alphabet_size,
                                        // ANSEncSymbolInfo ans_table[MAX_ALPHABET_SIZE],
                                        bool do_encode, // tmp cache int64

                                        uint8_t& max_nz_symbol,
                                        uint16_t& total,
                                        int& num_symbols,
                                        int scode_symbol[MAX_ALPHABET_SIZE],
                                        int re_counts1[MAX_ALPHABET_SIZE],
                                        // int hls_count_flat[MAX_ALPHABET_SIZE],

                                        bool& use_flat,
                                        hls::stream<bool>& strm_use_flat,

                                        hls::stream<int>& strm_num_pair,
                                        hls::stream<nbits_t>& strm_nbits,
                                        hls::stream<uint16_t>& strm_bits,

                                        hls::stream<int>& strm_num_fpair,
                                        hls::stream<nbits_t>& strm_flat_nbits,
                                        hls::stream<uint16_t>& strm_flat_bits) {
#pragma HLS INLINE OFF
    // because the use_flat is a conditional execution
    int num_pair = 0;
    int num_fpair = 0;

    assert(alphabet_size <= hls_ANS_TAB_SIZE); //<1024 = (1<<ANS_LOG_TAB_SIZE)

    bool mode = false;

    bool do_estimate = do_encode && (alphabet_size > hls_kMaxNumSymbolsForSmallCode); //

    _XF_IMAGE_PRINT("--1 NormalizeCounts - BuildAndStoreANS\n");

    int omit_pos = 0;
    int re_counts2[MAX_ALPHABET_SIZE];
    int re_counts3[MAX_ALPHABET_SIZE];
    // loop max_nz_symbol*3  dataflow
    SetTargetsWithRebalance(mode, histogram1, omit_pos, max_nz_symbol, total, num_symbols, scode_symbol[0],

                            re_counts1, re_counts2, re_counts3);

    // prepare for flat
    const int histo_bits_flat = hls_ANS_LOG_TAB_SIZE + 2;

    const int data_bits_flat = hls_EstimateDataBitsFlat(alphabet_size, total);

    // const int storage_ix0 = *pos;

    _XF_IMAGE_PRINT("--3 EncodeCounts - BuildAndStoreANS\n");

    int histo_bits = 0;
    // sequential
    hls_EncodeCounts(re_counts2, alphabet_size, omit_pos, num_symbols, scode_symbol, max_nz_symbol, do_encode,

                     histo_bits, num_pair, strm_nbits, strm_bits);

    hls_EncodeFlatHistogram(alphabet_size, do_encode, num_fpair, strm_flat_nbits, strm_flat_bits);

    // Let's see if we can do better in terms of histogram size + data size.
    // const int histo_bits = hls_pos - storage_ix0;
    int data_bits;
    hls_EstimateDataBits(do_estimate, histogram2, re_counts3, num_symbols, max_nz_symbol, data_bits);

    use_flat = do_estimate && (histo_bits_flat + data_bits_flat < histo_bits + data_bits);
    _XF_IMAGE_PRINT(
        "--histo_bits_flat = %d , data_bits_flat=%d, histo_bits = %d "
        ", data_bits=%d\n",
        histo_bits_flat, data_bits_flat, histo_bits, data_bits);

    strm_num_pair.write(num_pair);
    strm_num_fpair.write(num_fpair);
    strm_use_flat.write(use_flat);
}

// hls_histo_bitstream_top(cluster_size, num_pair, strm_nbits, strm_bits,
// use_flat, num_fpair, strm_flat_nbits,
//                        strm_flat_bits, pos, tail_bits, strm_histo,
//                        strm_histo_e);//);
void hls_histo_bitstream_top(const int cluster_size,
                             hls::stream<int>& num_pair,
                             hls::stream<nbits_t>& strm_nbits,
                             hls::stream<uint16_t>& strm_bits,

                             hls::stream<bool>& use_flat,
                             hls::stream<int>& num_fpair,
                             hls::stream<nbits_t>& strm_flat_nbits,
                             hls::stream<uint16_t>& strm_flat_bits,

                             int& pos, // tmp cache int64
                             uint8_t& tail_bits,
                             hls::stream<uint8_t>& strm_histo,
                             hls::stream<bool>& strm_histo_e) {
#pragma HLS INLINE OFF
    for (int c = 0; c < cluster_size; ++c) {
        hls_WriteBitStreamWithConsume(num_pair, strm_nbits, strm_bits, use_flat, num_fpair, strm_flat_nbits,
                                      strm_flat_bits, pos, tail_bits, strm_histo, strm_histo_e);
    }
}

void hls_build_ans_encode_histo(const bool is_dc,
                                uint32_t histogram[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                                const uint16_t alphabet_size,
                                const int cluster_size,
                                const uint16_t alphabet_size_dc[MAX_NUM_COLOR],
                                const bool do_encode,

                                hls::stream<int>& strm_num_pair,
                                hls::stream<nbits_t>& strm_nbits,
                                hls::stream<uint16_t>& strm_bits,

                                hls::stream<bool>& strm_use_flat,
                                hls::stream<int>& strm_num_fpair,
                                hls::stream<nbits_t>& strm_flat_nbits,
                                hls::stream<uint16_t>& strm_flat_bits,
                                hls_ANSEncSymbolInfo ans_table[hls_kNumStaticContexts][MAX_ALPHABET_SIZE]

                                ) {
#pragma HLS INLINE OFF

    for (int c = 0; c < cluster_size; ++c) {
        //#pragma HLS DATAFLOW

        uint16_t total = 0;
        uint8_t max_nz_symbol[2]; // for dataflow
        max_nz_symbol[0] = 0;
        max_nz_symbol[1] = 0;
        int num_symbols = 0;
        int num_byte;

        int scode_symbol[MAX_ALPHABET_SIZE]; // output
        int hls_counts1[MAX_ALPHABET_SIZE];
        int hls_counts2[MAX_ALPHABET_SIZE];
        int hls_counts3[MAX_ALPHABET_SIZE];
        int hls_count_flat[MAX_ALPHABET_SIZE];

        // loop max_loop = alphabet_size
        CountNZSymbol(histogram[c], max_nz_symbol, (is_dc ? alphabet_size_dc[c] : alphabet_size),

                      total, hls_counts1, hls_counts2, hls_counts3, hls_count_flat, num_symbols, scode_symbol);

        // count rebalance and estimate
        bool use_flat = false;
        int re_counts1[MAX_ALPHABET_SIZE];
        XAcc_BuildAndStoreANSEncodingData3(hls_counts1, hls_counts2, (is_dc ? alphabet_size_dc[c] : alphabet_size),

                                           do_encode, max_nz_symbol[0], total, num_symbols, scode_symbol,
                                           // output
                                           re_counts1, use_flat, strm_use_flat, strm_num_pair, strm_nbits, strm_bits,
                                           strm_num_fpair, strm_flat_nbits, strm_flat_bits);

        _XF_IMAGE_PRINT("-- ANSBuildInfoTable - BuildAndStoreANS\n");
        hls_ANSBuildInfoTable_syn(re_counts1, hls_count_flat, use_flat, (is_dc ? alphabet_size_dc[c] : alphabet_size),
                                  max_nz_symbol[1],

                                  hls_counts3, ans_table[c]);

// for print
#ifndef __SYNTHESIS__
        _XF_IMAGE_PRINT("c=%d \n", (int)c);
        _XF_IMAGE_PRINT("alphabet_size=%d \n", (int)(is_dc ? alphabet_size_dc[c] : alphabet_size));
        for (int i = 0; i < (is_dc ? alphabet_size_dc[c] : alphabet_size); ++i) { // 0~255
            hls_ANSEncSymbolInfo info = ans_table[c][i];
            if (info.freq_ > 0) _XF_IMAGE_PRINT("info.freq=%d, info.start=%d \n", (int)info.freq_, (int)info.start_);
        }
#endif

    } // end loop
}

void hls_build_and_encode_top(const bool is_dc,
                              uint32_t histogram[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                              const uint16_t alphabet_size,
                              const int cluster_size,
                              const uint16_t alphabet_size_dc[MAX_NUM_COLOR],

                              hls_ANSEncSymbolInfo ans_table[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                              int& pos, // tmp cache int64
                              const bool do_encode,
                              // uint8_t* storage
                              hls::stream<uint8_t>& strm_histo,
                              hls::stream<bool>& strm_histo_e,
                              uint8_t& tail_bits) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

    // clang-format off
	    hls::stream<int> num_pair;
#pragma HLS RESOURCE  	  variable = num_pair core = FIFO_BRAM
#pragma HLS STREAM    	  variable = num_pair depth = 2048
		hls::stream<nbits_t> strm_nbits;
#pragma HLS RESOURCE  	  variable = strm_nbits core = FIFO_BRAM
#pragma HLS STREAM    	  variable = strm_nbits depth = 2048
		hls::stream<uint16_t> strm_bits("strm_bits");
#pragma HLS RESOURCE  	  variable = strm_bits core = FIFO_BRAM
#pragma HLS STREAM    	  variable = strm_bits depth = 2048

		 hls::stream<bool>    use_flat;
#pragma HLS RESOURCE  	  variable = use_flat core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = use_flat depth = 32
		hls::stream<int>     num_fpair;
#pragma HLS RESOURCE  	  variable = num_fpair core = FIFO_BRAM
#pragma HLS STREAM    	  variable = num_fpair depth = 2048
	    hls::stream<nbits_t> strm_flat_nbits;
#pragma HLS RESOURCE  	  variable = strm_flat_nbits core = FIFO_BRAM
#pragma HLS STREAM    	  variable = strm_flat_nbits depth = 2048
	    hls::stream<uint16_t> strm_flat_bits("strm_flat_bits");
#pragma HLS RESOURCE  	  variable = strm_flat_bits core = FIFO_BRAM
#pragma HLS STREAM    	  variable = strm_flat_bits depth = 2048
    // clang-format on

    hls_build_ans_encode_histo(is_dc, histogram, alphabet_size, cluster_size, alphabet_size_dc, do_encode, num_pair,
                               strm_nbits, strm_bits, use_flat, num_fpair, strm_flat_nbits, strm_flat_bits, ans_table);

    hls_histo_bitstream_top(cluster_size, num_pair, strm_nbits, strm_bits, use_flat, num_fpair, strm_flat_nbits,
                            strm_flat_bits, pos, tail_bits, strm_histo, strm_histo_e);
}

void build_historgram_syn(hls::stream<ap_uint<13> >& strm_token_addr,
                          hls::stream<bool>& strm_e_addr,
                          hist_t total[hls_kMinClustersForHistogramRemap],
                          hist_t hls_histograms[hls_NumHistograms],
                          hist_t hls_histograms2[hls_kNumStaticContexts][MAX_ALPHABET_SIZE]

                          ) {
#pragma HLS INLINE OFF
#pragma HLS RESOURCE variable = hls_histograms core = RAM_2P_BRAM

    // init addr
    ap_uint<13> addr_c, addr_r0, addr_r1, addr_r2, addr_r3, addr_r4;
    addr_r0 = addr_r1 = addr_r2 = addr_r3 = addr_r4 = 0x1fff; // The 0x7ff should never be accessed
    addr_c = 0;
    // init reg

    hist_t cnt;
    hist_t cnt_r0 = 0; // max 4096?
    hist_t cnt_r1 = 0;
    hist_t cnt_r2 = 0;
    hist_t cnt_r3 = 0;
    hist_t cnt_r4 = 0;

    bool e = strm_e_addr.read();

    // 1.init
    if (!e) {
        for (int i = 0; i < hls_kNumStaticContexts; ++i) {
            for (int j = 0; j < MAX_ALPHABET_SIZE; ++j) {
#pragma HLS PIPELINE II = 1
                hls_histograms[i * MAX_ALPHABET_SIZE + j] = 0;
                hls_histograms2[i][j] = 0;
            }
        }
        for (int i = 0; i < hls_kMinClustersForHistogramRemap; ++i) {
            total[i] = 0;
        }
    }

AGGREGATE_TOKEN:
    while (!e) {
#pragma HLS dependence variable = hls_histograms inter false
#pragma HLS PIPELINE II = 1

        // 1)Get data's address
        // addr_c = ac_static_context_map[ac_token.context]<<8 + ac_token.symbol;
        addr_c = strm_token_addr.read();
        e = strm_e_addr.read();

        total[addr_c >> 8]++;

        // 2)Read RAM and select the cnt based on the addr, addr0 and addr1
        if (addr_c == addr_r0) { //&& cur_key == key_r0
            cnt = cnt_r0;
        } else if (addr_c == addr_r1) {
            cnt = cnt_r1;
        } else if (addr_c == addr_r2) {
            cnt = cnt_r2;
        } else if (addr_c == addr_r3) { // pass the cosim of 2019.1
            cnt = cnt_r3;
        } else if (addr_c == addr_r4) { // must be use for the cosim of 2018.3
            cnt = cnt_r4;
        } else {
            cnt = hls_histograms[addr_c];
        }
        // IMBS
        cnt = cnt + 1;
        // 3)Write back to RAM
        hls_histograms[addr_c] = cnt;
        hls_histograms2[addr_c >> 8][addr_c(7, 0)] = cnt;

        // 4)shift the whole data line 1 cycle for RAM content( state) and ADDRESS
        cnt_r4 = cnt_r3;
        cnt_r3 = cnt_r2;
        cnt_r2 = cnt_r1;
        cnt_r1 = cnt_r0;
        cnt_r0 = cnt;

        addr_r4 = addr_r3;
        addr_r3 = addr_r2;
        addr_r2 = addr_r1;
        addr_r1 = addr_r0;
        addr_r0 = addr_c;
    }
}

void XAcc_EncodeStaticContextMap(int& pos,
                                 hls::stream<uint8_t>& strm_histo,
                                 hls::stream<bool>& strm_histo_e,
                                 uint8_t& tail_cxt_bits) {
#pragma HLS INLINE OFF

    pos += 901;
    static const uint16_t static_cxt_short[56] = {
        0x8379, 0x2028, 0x7776, 0x6cdb, 0x557d, 0x3000, 0x0027, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x73a0, 0x97de,
        0x8470, 0xeeeb, 0x3434, 0x7774, 0x1a1a, 0x496e, 0xee10, 0xd3bb, 0xddd0, 0x69dd, 0xeae8, 0x34ee, 0x8f1c, 0xee79,
        0xf097, 0xde1a, 0xa776, 0xa1a1, 0xd3bb, 0xf0d0, 0x049e, 0x63e1, 0x1a77, 0xbbba, 0x0d3b, 0xdd5d, 0x869d, 0x73e3,
        0xf6be, 0xf877, 0xbf09, 0xd3bb, 0xd0d0, 0x69dd, 0xf868, 0x824b, 0xbff0, 0x0d3b, 0xdddd, 0x869d, 0xeeae, 0xc34e};
    int tail_cxt_pos = 5;
    // static const uint8_t
    tail_cxt_bits = 0x11;

    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 2; j++) {
#pragma HLS PIPELINE II = 1
            ap_uint<16> shortInt;
            if (j == 0) {
                shortInt = static_cxt_short[i];
            }

            uint8_t tmp = (j == 0) ? shortInt(7, 0) : shortInt(15, 8);
            strm_histo.write(tmp);
            strm_histo_e.write(false);
        }
    }
}

// ------------------------------------------------------------
void hls_StoreVarLenUint8_build(
    uint32_t n, int& num_bits, int& num, hls::stream<nbits_t>& strm_nbits, hls::stream<uint16_t>& strm_bits) {
#pragma HLS INLINE
    if (n == 0) {
        hls_WriteBits_strm(1, 0, num_bits, num, strm_nbits, strm_bits);
    } else {
        int nbits;
        for (int i = 0; i < 3; i++) {
#pragma HLS PIPELINE
            if (i == 0) {
                hls_WriteBits_strm_nodepend(1, 1, strm_nbits, strm_bits);
                nbits = hls_Log2FloorNonZero_32b(n);
            } else if (i == 1) {
                hls_WriteBits_strm_nodepend(3, nbits, strm_nbits, strm_bits);
            } else {
                hls_WriteBits_strm_nodepend(nbits, n - (1ULL << nbits), strm_nbits, strm_bits);
                num_bits += 1 + 3 + nbits;
                num += 2 + ((nbits == 0) ? 0 : 1);
            }
        }
    }
}

// sequential
void hls_EncodeContextMap(const uint8_t context_map[MAX_NUM_COLOR],
                          const int num_histograms,

                          int& num,
                          hls::stream<nbits_t>& strm_nbits,
                          hls::stream<uint16_t>& strm_bits) {
#pragma HLS INLINE OFF

    _XF_IMAGE_PRINT("---start EncodeContextMap:\n");

    int histo_bits = 0;
    // 1. write to storage_ix
    _XF_IMAGE_PRINT("--1 StoreVarLenUint8 - start EncodeContextMap\n");
    // hls_StoreVarLenUint8(num_histograms - 1, storage_ix, storage);
    hls_StoreVarLenUint8_build(num_histograms - 1, histo_bits, num, strm_nbits, strm_bits);

    // context_map = 000 will not go into the if
    if (num_histograms != 1) {
        // Alphabet size is 256 + 16 = 272. (We can have 256 clusters and 16 run
        // length codes).
        static const int kAlphabetSize = 272;

        // 2. sort
        _XF_IMAGE_PRINT("--2 MoveToFrontTransform - start EncodeContextMap\n");
        // 3. encode runlength:  input [dc, 0*63] return v_out = [dc, prefix_code] ,
        // extra = [64 - 1 - length of
        // prefix_code]
        _XF_IMAGE_PRINT("--3 RunLengthCodeZeros - start EncodeContextMap\n");
        uint32_t max_run_length_prefix = 0;
        // RunLengthCodeZeros(transformed_symbols, &max_run_length_prefix,
        // &rle_symbols,
        //                     &extra_bits);
        if (context_map[0] == 0 && (context_map[1] == 0) && (context_map[2] == 1)) {
            max_run_length_prefix = 1;
        }
        // 4. write a length
        _XF_IMAGE_PRINT("--4 use_rle - start EncodeContextMap\n");

        bool use_rle = max_run_length_prefix > 0;
        hls_WriteBits_strm(1, use_rle, histo_bits, num, strm_nbits, strm_bits);
        if (use_rle) {
            hls_WriteBits_strm(4, max_run_length_prefix - 1, histo_bits, num, strm_nbits, strm_bits);
        }
        // 5. write huffman tree
        _XF_IMAGE_PRINT("--5 BuildAndStoreHuffmanTree - start EncodeContextMap\n");
        if (context_map[0] == 0 && (context_map[1] == 1) && (context_map[2] == 0)) {
            hls_WriteBits_strm(2, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(2, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(4, 0, histo_bits, num, strm_nbits, strm_bits);
        } else if (context_map[0] == 0 && (context_map[1] == 1) && (context_map[2] == 2)) {
            hls_WriteBits_strm(2, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(2, 2, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(4, 0, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(4, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
        } else if (context_map[0] == 0 && (context_map[1] == 0) && (context_map[2] == 1)) {
            hls_WriteBits_strm(2, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(2, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(4, 0, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(4, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
        } else {
            _XF_IMAGE_PRINT("--ERROR : new status to be added !!!\n");
        }

        // 6. move storage to storage_ix
        _XF_IMAGE_PRINT("--6 move storage to storage_ix - start EncodeContextMap\n");
        if (context_map[0] == 0 && (context_map[1] == 1) && (context_map[2] == 0)) {
            hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 1, histo_bits, num, strm_nbits, strm_bits);
        } else if (context_map[0] == 0 && (context_map[1] == 1) && (context_map[2] == 2)) {
            hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(2, 1, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(2, 3, histo_bits, num, strm_nbits, strm_bits);
        } else if (context_map[0] == 0 && (context_map[1] == 0) && (context_map[2] == 1)) {
            hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 0, histo_bits, num, strm_nbits, strm_bits);
            hls_WriteBits_strm(1, 1, histo_bits, num, strm_nbits, strm_bits);
        } else {
            _XF_IMAGE_PRINT("--ERROR : new status to be added !!!\n");
        }
        hls_WriteBits_strm(1, 1, histo_bits, num, strm_nbits, strm_bits); // use move-to-front
    }                                                                     // end if (num_histograms != 1)
}

void hls_EncodeContextMapByte(const uint8_t context_map[MAX_NUM_COLOR],
                              const int num_histograms,
                              uint8_t& tail_bits,
                              int& pos,
                              hls::stream<uint8_t>& strm_histo_byte,
                              hls::stream<bool>& strm_histo_e) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

    int num = 0;

    // clang-format off
		 hls::stream<nbits_t> strm_nbits("strm_nbits_cluster");
#pragma HLS RESOURCE  	  variable = strm_nbits core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_nbits depth = 32
		hls::stream<uint16_t> strm_bits("strm_bits_cluster");
#pragma HLS RESOURCE  	  variable = strm_bits core = FIFO_LUTRAM
#pragma HLS STREAM    	  variable = strm_bits depth = 32
    // clang-format on

    hls_EncodeContextMap(context_map, num_histograms, num, strm_nbits, strm_bits);

    hls_WriteBitToStream(num, tail_bits, strm_nbits, strm_bits, pos, strm_histo_byte, strm_histo_e);
}

inline double CrossEntropy(const uint32_t* counts, const int counts_len, const uint32_t* codes, const int codes_len) {
    double sum = 0.0f;
    uint32_t total_count = 0;
    uint32_t total_codes = 0;
    for (int i = 0; i < codes_len; ++i) {
#pragma HLS PIPELINE
        if (codes[i] > 0) {
            if (i < counts_len && counts[i] > 0) {
                sum -= counts[i] * hls_FastLog2(codes[i]);
                total_count += counts[i];
            }
            total_codes += codes[i];
        }
    }
    if (total_codes > 0) {
        sum += total_count * hls_FastLog2(total_codes);
    }
    return sum;
}

inline double hls_ShannonEntropy(const uint32_t* data, const int data_size) {
    return CrossEntropy(data, data_size, data, data_size);
}

void CopyStaticHisto(const hist_t hls_histograms2[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                     hist_t hls_histograms_out[hls_kNumStaticContexts][MAX_ALPHABET_SIZE]) {
#pragma HLS INLINE OFF

    for (int i = 0; i < hls_kNumStaticContexts; ++i) {
        for (int j = 0; j < MAX_ALPHABET_SIZE; ++j) { // 0~255
#pragma HLS PIPELINE II = 1
            hls_histograms_out[i][j] = hls_histograms2[i][j];
        }
    }
}

#define DEBUG_CLUSTER (1)
// sequential
void hls_encode_histo_context(const bool is_dc,
                              const hist_t hls_histograms2[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],

                              hls_ANSEncSymbolInfo hls_codes[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                              uint8_t dc_context_map[MAX_NUM_COLOR],
                              int& pos,
                              hls::stream<uint8_t>& strm_histo_byte,
                              hls::stream<bool>& strm_histo_e) {
#pragma HLS INLINE OFF

    // prepare buffers
    // const int max_out_size = hls_kNumStaticContexts * 1024;//24K
    uint8_t tail_bits;

    int num_clusters = 0;
    int cluster_dcac;

    uint16_t alphabet_size_dc[MAX_NUM_COLOR];
    hist_t hls_histograms_out[hls_kNumStaticContexts][MAX_ALPHABET_SIZE];
    // 3. build and store ANS from histograms/counts to ans_table and storage

    if (is_dc) {
#ifdef DEBUG_DC_HISTO
        _XF_IMAGE_PRINT("---org historgram:\n");
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 256; i++) {
                uint32_t tmp = hls_histograms2[c][i];
                _XF_IMAGE_PRINT("%d,", (int)(tmp));
            }
            _XF_IMAGE_PRINT("\n");
        }
#endif

        _XF_IMAGE_PRINT("---start ClusterHistograms:\n");

        uint8_t max_nz_symbol[MAX_NUM_COLOR];

        hls_ClusterHistograms_top(hls_histograms2, hls_kClustersLimit, num_clusters, max_nz_symbol, hls_histograms_out,
                                  dc_context_map);

#ifdef DEBUG_CLUSTER

        _XF_IMAGE_PRINT("---remap historgram:\n");
        for (int c = 0; c < num_clusters; c++) {
            for (int i = 0; i < max_nz_symbol[c]; i++) {
                uint32_t tmp = hls_histograms_out[c][i];
                _XF_IMAGE_PRINT("%d,", (int)(tmp));
            }
            _XF_IMAGE_PRINT("\n");
        }
        _XF_IMAGE_PRINT("---dc / ctrl context_map:\n");
        for (int c = 0; c < MAX_NUM_COLOR; ++c) {
            _XF_IMAGE_PRINT("%d,", dc_context_map[c]);
        }
        _XF_IMAGE_PRINT("\n");

        _XF_IMAGE_PRINT("-alphabet_size = %d, pos=%d\n", (int)max_nz_symbol[0], (int)(pos));
        _XF_IMAGE_PRINT("-alphabet_size = %d, pos=%d\n", (int)max_nz_symbol[1], (int)(pos));
        _XF_IMAGE_PRINT("-alphabet_size = %d, pos=%d\n", (int)max_nz_symbol[2], (int)(pos));

#endif

        tail_bits = 0;
        hls_EncodeContextMapByte(dc_context_map, num_clusters, tail_bits, pos, strm_histo_byte, strm_histo_e);

        bool do_encode = (pos != 0);
        for (int c = 0; c < MAX_NUM_COLOR; c++) {
            alphabet_size_dc[c] = max_nz_symbol[c];
        }
        cluster_dcac = num_clusters;
        _XF_IMAGE_PRINT("---end EncodeContextMap\n");

    } else {
        cluster_dcac = hls_kNumStaticContexts;

        CopyStaticHisto(hls_histograms2, hls_histograms_out);
        // 3. Encode the ContextMap.
        XAcc_EncodeStaticContextMap(pos, strm_histo_byte, strm_histo_e, tail_bits);
    }

    bool do_encode = (pos != 0);
    _XF_IMAGE_PRINT("do_encode=%d \n", (int)do_encode);
    hls_build_and_encode_top(is_dc, hls_histograms_out, 256, cluster_dcac,
                             alphabet_size_dc, // use when dc
                             hls_codes, pos, do_encode, strm_histo_byte, strm_histo_e, tail_bits);

    if (pos & (7)) {
        strm_histo_byte.write(tail_bits);
        strm_histo_e.write(false);
    }
    strm_histo_e.write(true);

    // 4. Close the histogram bit stream.
    // WriteZeroesToByteBoundary(&storage_ix, storage);
    _XF_IMAGE_PRINT("storage_ix=%d \n", (int)pos);

    hls_WriteZeroesToByteBoundary(&pos);

    _XF_IMAGE_PRINT("storage_ix=%d \n", (int)pos);
}

void XAcc_EncodeHistogramsFast_top(const bool is_dc,
                                   uint8_t dc_context_map[MAX_NUM_COLOR],
                                   hls::stream<ap_uint<13> >& strm_token_addr,
                                   hls::stream<bool>& strm_e_addr,

                                   // hls_PikImageSizeInfo info,
                                   hls_ANSEncSymbolInfo hls_codes[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                                   hist_t hls_histograms[hls_NumHistograms],
                                   int pos,
                                   int& len_histo,
                                   hls::stream<uint8_t>& strm_histo_byte,
                                   hls::stream<bool>& strm_histo_e) {
#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

    hist_t hls_histograms2[hls_kNumStaticContexts][MAX_ALPHABET_SIZE];
    hist_t total[hls_kMinClustersForHistogramRemap];
    // todo: just init once

    // 2.build
    build_historgram_syn(strm_token_addr, strm_e_addr, total, hls_histograms, hls_histograms2);

    for (int j = 0; j < (10); ++j) {
        _XF_IMAGE_PRINT("--historgrams[%d] = %d\n", j, (int)hls_histograms[j]);
    }

    for (int j = 256; j < (256 + 10); ++j) {
        _XF_IMAGE_PRINT("--historgrams[%d] = %d\n", j, (int)hls_histograms[j]);
    }
    for (int j = 512; j < (512 + 10); ++j) {
        _XF_IMAGE_PRINT("--historgrams[%d] = %d\n", j, (int)hls_histograms[j]);
    }

    // encode
    hls_encode_histo_context(is_dc, hls_histograms2, hls_codes, dc_context_map, pos, strm_histo_byte, strm_histo_e); //

    int storage_ix = pos;
    len_histo = (storage_ix + 7) >> 3;
}
