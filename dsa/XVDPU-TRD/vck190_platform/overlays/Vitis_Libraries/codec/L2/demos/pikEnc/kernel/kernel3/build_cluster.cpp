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

#include "kernel3/build_cluster.hpp"

struct hls_HistogramPair {
    uint32_t idx1;
    uint32_t idx2;
    double cost_combo;
    double cost_diff;
};

float hls_ANSPopulationCost(const hist_t* data, int alphabet_size, int total_count) {
#pragma HLS INLINE OFF

    static hls::stream<float> strm_fsub;
#pragma HLS RESOURCE variable = strm_fsub core = FIFO_BRAM
#pragma HLS STREAM variable = strm_fsub depth = 1024
    static hls::stream<float> strm_entropy_bits;
#pragma HLS RESOURCE variable = strm_entropy_bits core = FIFO_LUTRAM
#pragma HLS STREAM variable = strm_entropy_bits depth = 32
    int num = 0;
    float fsub;

    if (total_count == 0) {
        return 7;
    }

    float entropy_bits = total_count * hls_ANS_LOG_TAB_SIZE;
    int histogram_bits = 0;
    int count = 0;
    int length = 0;

    if (total_count > hls_ANS_TAB_SIZE) {
        uint64_t total = total_count;
        for (int i = 0; i < alphabet_size; ++i) {
            if (data[i] > 0) {
                ++count;    // 25    num_nz
                length = i; // 24 max_nz_symbol
            }
        }
        if (count == 1) {
            return 7;
        }
        ++length;                                                                // 25
        const uint64_t max0 = (total * length) >> hls_ANS_LOG_TAB_SIZE;          // 184
        const uint64_t max1 = (max0 * length) >> hls_ANS_LOG_TAB_SIZE;           // 4
        const uint32_t min_base = (total + max0 + max1) >> hls_ANS_LOG_TAB_SIZE; // 7
        total += min_base * count;                                               // 7735
        const int64_t kFixBits = 32;
        const int64_t kFixOne = 1LL << kFixBits;                      // 4294967296
        const int64_t kDescaleBits = kFixBits - hls_ANS_LOG_TAB_SIZE; // 22
        const int64_t kDescaleOne = 1LL << kDescaleBits;              // 4194304
        const int64_t kDescaleMask = kDescaleOne - 1;
        const uint32_t mult = kFixOne / total;  /// 555264
        const uint32_t error = kFixOne % total; // 256
        uint32_t cumul = error;
        if (error < kDescaleOne) {
            cumul += (kDescaleOne - error) >> 1; // 2097280
        }

        _XF_IMAGE_PRINT("---cobo:  total = %d \n", total_count);

        if (data[0] > 0) { // 870
            uint64_t c = (uint64_t)(data[0] + min_base) * mult + cumul;
            _XF_IMAGE_PRINT("data[0]= %d, c = %ld\n", (int)(data[0]), (c >> kDescaleBits));
            float log2count = hls_FastLog2(c >> kDescaleBits);
            // entropy_bits -= data[0] * log2count;
            fsub = data[0] * log2count;
            strm_fsub.write((-1 * fsub));

            cumul = c & kDescaleMask; // 2524544

            num++;
        }
        _XF_IMAGE_PRINT("\n : entropy_bits = %f\n", (entropy_bits));

        for (int i = 1; i < length; ++i) {
#pragma HLS PIPELINE II = 1
            if (data[i] > 0) {
                uint64_t c = (uint64_t)(data[i] + min_base) * mult + cumul;
                _XF_IMAGE_PRINT("data[%d]= %d, c = %ld\n", i, (int)(data[i]), (c >> kDescaleBits));

                float log2count = hls_FastLog2(c >> kDescaleBits); // 6.49
                int log2floor = static_cast<int>(log2count);       // 6
                // entropy_bits -= data[i] * log2count;//65277
                fsub = data[i] * log2count;
                strm_fsub.write((-1 * fsub));
                num++;

                // when use ap_uint<16>
                // log2count from 9.27 to 9 make the fsub is smaller, then the return is
                // bigger to all
                // which is misable
                _XF_IMAGE_PRINT("log2count= %f, log2floor = %d\n", log2count, (log2floor));

                histogram_bits += log2floor;                              // 6
                histogram_bits += hls_kLogCountBitLengths[log2floor + 1]; // 6+3
                cumul = c & kDescaleMask;

            } else {
                histogram_bits += hls_kLogCountBitLengths[0];
            }
        } // end for

        _XF_IMAGE_PRINT("\n");

    } else {
        float log2norm = hls_ANS_LOG_TAB_SIZE - hls_FastLog2(total_count);
        if (data[0] > 0) {
            float log2count = hls_FastLog2(data[0]) + log2norm;
            // entropy_bits -= data[0] * log2count;
            fsub = data[0] * log2count;
            strm_fsub.write((-1 * fsub));
            num++;

            length = 0;
            ++count;
        }
        for (int i = 1; i < alphabet_size; ++i) {
#pragma HLS PIPELINE II = 1
            if (data[i] > 0) {
                float log2count = hls_FastLog2(data[i]) + log2norm;
                int log2floor = static_cast<int>(log2count);
                // entropy_bits -= data[i] * log2count;
                fsub = data[i] * log2count;
                strm_fsub.write((-1 * fsub));
                num++;

                if (log2floor >= hls_ANS_LOG_TAB_SIZE) {
                    log2floor = hls_ANS_LOG_TAB_SIZE - 1;
                }
                histogram_bits += (log2floor + 1) >> 1; // GetPopulationCountPrecision(log2floor);
                histogram_bits += hls_kLogCountBitLengths[log2floor + 1];
                length = i;
                ++count;
            } else {
                histogram_bits += hls_kLogCountBitLengths[0];
            }
        }
        ++length;
    }

    if (num != 0) {
        num++;
        fsub = total_count * hls_ANS_LOG_TAB_SIZE;
        strm_fsub.write((fsub));
    } else {
        num = 0;
    }

    ADD_FP_strm(num, strm_fsub, strm_entropy_bits);

    if (num != 0) {
        entropy_bits = strm_entropy_bits.read();
    }

    if (count == 1) {
        return 7;
    }

    if (count == 2) {
        return static_cast<int>(entropy_bits) + 1 + 12 + hls_ANS_LOG_TAB_SIZE;
    }

    uint8_t tmp = alphabet_size - 1;
    int max_bits = 1 + (tmp == 0 ? -1 : hls_Log2FloorNonZero_32b((int)tmp));
    histogram_bits += max_bits;

    _XF_IMAGE_PRINT("\n histogram_bits= %d, entropy_bits = %d\n", (int)(histogram_bits),
                    (static_cast<int>(entropy_bits)));

    return histogram_bits + static_cast<int>(entropy_bits) + 1;
}

inline float hls_ClusterCostDiff(int size_a, int size_b) {
    int size_c = size_a + size_b;
    return size_a * hls_FastLog2(size_a) + size_b * hls_FastLog2(size_b) - size_c * hls_FastLog2(size_c);
}

inline bool comparePair(const hls_HistogramPair& p1, const hls_HistogramPair& p2) {
    if (p1.cost_diff != p2.cost_diff) {
        return p1.cost_diff > p2.cost_diff;
    }
#ifndef __SYNTHESIS__
    return std::abs(p1.idx1 - p1.idx2) > std::abs(p2.idx1 - p2.idx2);
#else
    return hls::abs(p1.idx1 - p1.idx2) > hls::abs(p2.idx1 - p2.idx2);
#endif
}

void hls_CompareAndPushToQueue(const hist_t hls_clustgrams[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                               const uint16_t total_count[MAX_NUM_COLOR],
                               const int max_loop,
                               const uint8_t max_nz_symbol[MAX_NUM_COLOR],

                               const int cluster_size[MAX_NUM_COLOR],
                               const float bit_cost[MAX_NUM_COLOR],
                               int idx1,
                               int idx2,
                               bool& pair_is_empty,
                               int& cnt,
                               hls_HistogramPair pairs[MAX_NUM_COLOR] // for {01}{02}{12}
                               ) {
#pragma HLS INLINE OFF

    if (idx1 == idx2) {
        return;
    }
    if (idx2 < idx1) {
        int t = idx2;
        idx2 = idx1;
        idx1 = t;
    }
    bool store_pair = false;
    hls_HistogramPair p;
    p.idx1 = idx1;
    p.idx2 = idx2;
    _XF_IMAGE_PRINT("------cluster_size[idx1]=%d, cluster_size[idx2]=%d\n", cluster_size[idx1], cluster_size[idx2]);
    p.cost_diff = 0.5f * hls_ClusterCostDiff(cluster_size[idx1], cluster_size[idx2]);
    _XF_IMAGE_PRINT("------cost_diff[%d]=%f\n", p.cost_diff);
    p.cost_diff -= bit_cost[idx1];
    p.cost_diff -= bit_cost[idx2];

    _XF_IMAGE_PRINT("-bit_cost[%d]=%f, bit_cost[%d]=%f\n", idx1, bit_cost[idx1], idx2, bit_cost[idx2]);

    if (total_count[idx1] == 0) {
        p.cost_combo = bit_cost[idx2];
        store_pair = true;
    } else if (total_count[idx2] == 0) {
        p.cost_combo = bit_cost[idx1];
        store_pair = true;
    } else {
        double zero = 0.0f;

        float threshold;
#ifndef __SYNTHESIS__
        threshold = pair_is_empty ? std::numeric_limits<float>::max() : hls::max(zero, pairs[0].cost_diff);
#else
        threshold = hls::max(zero, pairs[0].cost_diff);
#endif
        // Histogram combo = out[idx1];
        // combo.AddHistogram(out[idx2]);// total = 3780*2
        // float cost_combo = combo.hls_PopulationCost();//31914  27268  31977 46566
        hist_t hls_clustgrams12[MAX_ALPHABET_SIZE];

        for (int n = 0; n < max_loop; ++n) { // 0~255
#pragma HLS PIPELINE II = 1
            hls_clustgrams12[n] = hls_clustgrams[idx1][n] + hls_clustgrams[idx2][n];
        }

        uint16_t total_count12 = total_count[idx1] + total_count[idx2];
        int combo_size = hls::max(max_nz_symbol[idx1], max_nz_symbol[idx2]);
        _XF_IMAGE_PRINT("-combo_size=%d, total_count12=%d\n", combo_size, (int)total_count12);
        float cost_combo = hls_ANSPopulationCost(hls_clustgrams12, combo_size, (int)total_count12);

        _XF_IMAGE_PRINT("-cost_combo=%f, p.cost_diff=%f, threshold=%f\n", cost_combo, p.cost_diff, threshold);
        if (pair_is_empty || (cost_combo + p.cost_diff < threshold)) { // 2116  -25   -29266
            // threshold max  max?    0
            p.cost_combo = cost_combo;
            store_pair = true;
            pair_is_empty = false;
        }
    }
    if (store_pair) {
        p.cost_diff += p.cost_combo;

        if (cnt == 0) {
            pairs[0].idx1 = p.idx1;
            pairs[0].idx2 = p.idx2;
            pairs[0].cost_diff = p.cost_diff;
            pairs[0].cost_combo = p.cost_combo;
            cnt++;
        } else {
            bool smaller = comparePair(pairs[cnt - 1], p);
            if (smaller) {
                if (cnt == 1) {
                    pairs[cnt].idx1 = pairs[cnt - 1].idx1;
                    pairs[cnt].idx2 = pairs[cnt - 1].idx2;
                    pairs[cnt].cost_diff = pairs[cnt - 1].cost_diff;
                    pairs[cnt].cost_combo = pairs[cnt - 1].cost_combo;
                } else {
                    pairs[cnt].idx1 = pairs[cnt - 1].idx1;
                    pairs[cnt].idx2 = pairs[cnt - 1].idx2;
                    pairs[cnt].cost_diff = pairs[cnt - 1].cost_diff;
                    pairs[cnt].cost_combo = pairs[cnt - 1].cost_combo;
                    pairs[cnt - 1].idx1 = pairs[cnt - 2].idx1;
                    pairs[cnt - 1].idx2 = pairs[cnt - 2].idx2;
                    pairs[cnt - 1].cost_diff = pairs[cnt - 2].cost_diff;
                    pairs[cnt - 1].cost_combo = pairs[cnt - 2].cost_combo;
                }
                pairs[0].idx1 = p.idx1;
                pairs[0].idx2 = p.idx2;
                pairs[0].cost_diff = p.cost_diff;
                pairs[0].cost_combo = p.cost_combo;
            } else {
                pairs[cnt].idx1 = p.idx1;
                pairs[cnt].idx2 = p.idx2;
                pairs[cnt].cost_diff = p.cost_diff;
                pairs[cnt].cost_combo = p.cost_combo;
            }
            cnt++;
        }
    }
}

void smallCase_AddHistogram(const hist_t hls_clustgrams[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                            uint16_t total_count[MAX_NUM_COLOR],
                            const int max_symbol,
                            hist_t hls_clustgrams_out[hls_kNumStaticContexts][MAX_ALPHABET_SIZE]) {
#pragma HLS INLINE OFF

#pragma HLS ARRAY_PARTITION variable = hls_clustgrams complete dim = 1
#pragma HLS ARRAY_PARTITION variable = total_count complete dim = 1

    for (int i = 0; i < max_symbol; i++) {
#pragma HLS PIPELINE II = 1
        hist_t tmp0 = hls_clustgrams[0][i];
        hist_t tmp1 = hls_clustgrams[1][i];
        hist_t tmp2 = hls_clustgrams[2][i];

        hls_clustgrams_out[0][i] = tmp0 + tmp1 + tmp2;

        if (i == 0) {
            uint16_t tmp3 = total_count[0];
            uint16_t tmp4 = total_count[1];
            uint16_t tmp5 = total_count[2];
            total_count[0] = tmp3 + tmp4 + tmp5;
        }

    } // end for
}

void hls_HistogramCombine(const hist_t hls_clustgrams[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                          uint16_t total_count[MAX_NUM_COLOR],
                          const int max_loop,
                          uint8_t max_nz_symbol[MAX_NUM_COLOR], // rewrite by combo

                          int cluster_size[MAX_NUM_COLOR], // 1,1,1
                          float bit_cost[MAX_NUM_COLOR],
                          uint8_t dc_context_map[MAX_NUM_COLOR],
                          int symbols_size, // 3
                          int max_clusters, // 64
                          hist_t hls_clustgrams_out[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],

                          int& num_clusters) {
#pragma HLS INLINE OFF

    float cost_diff_threshold = 0.0f;
    int min_cluster_size = 1;

    // Uniquify the list of symbols after merging empty clusters.

    int clusters[MAX_NUM_COLOR];
    int cnt = 0;

    int sum_of_totals = 0;
    int first_zero_pop_count_symbol = -1;
    for (int i = 0; i < MAX_NUM_COLOR; ++i) { // 3
#pragma HLS PIPELINE II = 1
        if (total_count[dc_context_map[i]] == 0) { // 3780
            // Merge the zero pop count histograms into one.
            if (first_zero_pop_count_symbol == -1) {
                first_zero_pop_count_symbol = dc_context_map[i];
                clusters[cnt] = dc_context_map[i];
                cnt++;
            } else {
                dc_context_map[i] = first_zero_pop_count_symbol;
            }
        } else {
            // Insert all histograms with non-zero pop counts.
            clusters[cnt] = dc_context_map[i];
            sum_of_totals += total_count[dc_context_map[i]]; // 11340 = total_token
            cnt++;
        }
    }

    if (sum_of_totals < 160) { // 12288

        *cluster_size = 1;
        smallCase_AddHistogram(hls_clustgrams, total_count, max_loop, hls_clustgrams_out);
        // Histogram combo = hls_clustgrams[dc_context_map[0]];
        // for (int i = 1; i < MAX_NUM_COLOR; ++i) {
        // combo.AddHistogram(hls_clustgrams[dc_context_map[i]]);
        //}
        // cluster_histograms[dc_context_map[0]] = combo;// todo
        for (int i = 1; i < MAX_NUM_COLOR; ++i) { // 1,2
#pragma HLS UNROLL
            dc_context_map[i] = dc_context_map[0];
        }
        // return 1;
        num_clusters = 1;
        max_nz_symbol[0] = max_loop;

    } else {
        // cpoy
        for (int c = 0; c < MAX_NUM_COLOR; ++c) {
            // for (int n = 0; n < max_nz_symbol[c]; ++n) { // 0~255 // csim
            for (int n = 0; n < 256; ++n) { // 0~255   //syn
#pragma HLS PIPELINE II = 1
                hls_clustgrams_out[c][n] = hls_clustgrams[c][n];
            }
        }

        bool pair_is_empty = true;
        int pairs_size = 0;
        hls_HistogramPair pairs[MAX_NUM_COLOR];
        // sequential
        // std::vector<hls_HistogramPair> pairs;
        for (int idx1 = 0; idx1 < cnt; ++idx1) {
            for (int idx2 = idx1 + 1; idx2 < cnt; ++idx2) {
                _XF_IMAGE_PRINT("org compair p.cost_diff to check (%d,%d)\n", idx1, idx2);
                hls_CompareAndPushToQueue(hls_clustgrams, total_count, max_loop, max_nz_symbol, cluster_size, bit_cost,
                                          clusters[idx1], clusters[idx2], pair_is_empty, pairs_size, pairs);
            }
        }

        // debug
        for (int i = 0; i < pairs_size; ++i) {
            _XF_IMAGE_PRINT("\n debug : %d-pairs.cost_diff=%f, p.cost_combo=%f,idx1=%d,idx2=%d\n", i,
                            pairs[i].cost_diff, pairs[i].cost_combo, pairs[i].idx1, pairs[i].idx2);
        }

        while (num_clusters > min_cluster_size) {
            _XF_IMAGE_PRINT("pairs[0].cost_diff=%f, cost_diff_threshold=%f\n", pairs[0].cost_diff, cost_diff_threshold);
            if (pairs[0].cost_diff < cost_diff_threshold) {
                int best_idx1 = pairs[0].idx1; // 0
                int best_idx2 = pairs[0].idx2; // 2
                // cluster_histograms[best_idx1].AddHistogram(cluster_histograms[best_idx2]);

                int hls_clustgrams12[MAX_ALPHABET_SIZE];

                for (int n = 0; n < max_loop; ++n) { // 0~255 ii=n or unroll the 256*3 lut
#pragma HLS PIPELINE
                    hls_clustgrams_out[best_idx1][n] =
                        hls_clustgrams_out[best_idx1][n] + hls_clustgrams_out[best_idx2][n];
                }
                if (max_nz_symbol[best_idx1] < max_nz_symbol[best_idx2]) {
                    max_nz_symbol[best_idx1] = max_nz_symbol[best_idx2];
                }
                total_count[best_idx1] = total_count[best_idx1] + total_count[best_idx2];
                bit_cost[best_idx1] = pairs[0].cost_combo; // 27268//28093

                cluster_size[best_idx1] += cluster_size[best_idx2];
                for (int i = 0; i < MAX_NUM_COLOR; ++i) {
#pragma HLS PIPELINE II = 1
                    if (dc_context_map[i] == best_idx2) { // 0,1,2->0,1,0
                        dc_context_map[i] = best_idx1;
                    }
                }

                if (best_idx2 == 1) { // others will be clusters[0] = 0; clusters[1] =
                                      // 1;
                    clusters[1] = 2;
                    for (int n = 0; n < max_loop; ++n) {
#pragma HLS PIPELINE II = 1
                        hls_clustgrams_out[1][n] = hls_clustgrams_out[2][n];
                    }
                }

                // because of the init is 3
                pair_is_empty = true;
                pairs_size = 0;
                // clusters.resize;
                // because init is 3 ,then if Remove pairs, left is 2
                num_clusters = num_clusters - 1; // init is 3

                for (int i = 0; i < num_clusters; ++i) {
                    hls_CompareAndPushToQueue(hls_clustgrams_out, total_count, max_loop, max_nz_symbol, cluster_size,
                                              bit_cost, best_idx1, clusters[i], pair_is_empty, pairs_size, pairs);
                }
            } else {
                // to end while
                const int kClustersLimit = 64;
                min_cluster_size = kClustersLimit;
            }
        } // end while

    } // endif
}

void CountDChisto_todo(const uint32_t histogram[MAX_ALPHABET_SIZE], // output
                       uint8_t& max_nz_symbol,
                       const uint16_t max_loop,

                       uint16_t& total,
                       int hls_counts[MAX_ALPHABET_SIZE],
                       int hls_counts2[MAX_ALPHABET_SIZE],
                       int hls_counts3[MAX_ALPHABET_SIZE],
                       int hls_countFlat[MAX_ALPHABET_SIZE],
                       int& num_symbols,                    // output
                       int scode_symbols[MAX_ALPHABET_SIZE] // output
                       ) {
#pragma HLS INLINE
    // const int table_size = 1 << ANS_LOG_TAB_SIZE;  // target sum / table size
    // uint16_t total = 0;
    total = 0; // change from 64 to 32 to 16 because there is max 2^16 tockens
    int max_symbol = 0;
    int symbol_count = 0;
    const int flat_cnt = hls_ANS_TAB_SIZE / max_loop;

    // 1. test if symbol_count > precision_table_size
    // cnt = sym_cnt + 0_cnt
    // total of the all the tockens
    for (int n = 0; n < max_loop; ++n) { // 0~255
#pragma HLS PIPELINE II = 1
        total += histogram[n];
        hls_counts[n] = histogram[n];
        hls_counts2[n] = histogram[n];
        hls_counts3[n] = histogram[n];
        hls_countFlat[n] = flat_cnt;

        if (histogram[n] > 0) { // the front 4 non-zero cnt is record
            if (symbol_count < hls_kMaxNumSymbolsForSmallCode) {
                scode_symbols[symbol_count] = n;
            }
            ++symbol_count; // sym_cnt is non-z cnt
            max_symbol = n + 1;
            _XF_IMAGE_PRINT("--historgrams[%d] = %d\n", n, (int)histogram[n]);
        }
    }

    max_nz_symbol = max_symbol;
    // count the symbol to num_symbols
    num_symbols = symbol_count;
}

// dc and ctrl flied have the same 3 in_size for
//{x,y,b} and {quenter, acstrategy, arparameter}
void hls_ClusterHistograms_top(

    const hist_t hls_clustgrams[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
    int max_histograms,
    int& num_clusters,
    uint8_t max_nz_symbol[MAX_NUM_COLOR],

    hist_t hls_clustgrams_out[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
    uint8_t dc_context_map[MAX_NUM_COLOR] // histogram_symbols
    ) {
#pragma HLS INLINE OFF

    // block_group_offsets is 0 forever in the origin codes;

    const int in_size = MAX_NUM_COLOR; // num_contexts * num_blocks;// dc and ctrl
                                       // flied have the same 3 in_size for

    // init
    int num_symbols[MAX_NUM_COLOR];
    uint8_t max_loop = 0;
    uint16_t total_count[MAX_NUM_COLOR];
    float bit_cost[MAX_NUM_COLOR];
    int cluster_size[MAX_NUM_COLOR] = {1, 1, 1}; // debug syn

    int max_symbol = 0;
    int symbol_count = 0;

    uint16_t total_tmp = 0;
    for (int c = 0; c < MAX_NUM_COLOR; ++c) {
#pragma HLS UNROLL
        total_count[c] = 0;
    }

    for (int c = 0; c < MAX_NUM_COLOR; ++c) {
        for (int n = 0; n < MAX_ALPHABET_SIZE; ++n) { // 0~255
#pragma HLS PIPELINE II = 1
            total_count[c] += hls_clustgrams[c][n];

            if (hls_clustgrams[c][n] > 0) { // the front 4 non-zero cnt is record

                ++symbol_count; // sym_cnt is non-z cnt
                max_symbol = n + 1;
            }
        }
        max_nz_symbol[c] = max_symbol;
        num_symbols[c] = symbol_count;
        max_loop = (max_loop < max_symbol) ? max_symbol : max_loop;
        // clear for next loop
        max_symbol = 0;
        symbol_count = 0;
    }
    _XF_IMAGE_PRINT("--- total = %d \n", total_count[0]);

    for (int i = 0; i < MAX_NUM_COLOR; ++i) {
        _XF_IMAGE_PRINT(" counts.size()= %d, total = %d\n", (int)(max_nz_symbol[i]), total_count[i]);
        bit_cost[i] = hls_ANSPopulationCost(hls_clustgrams[i], (int)max_nz_symbol[i], (int)total_count[i]);
        dc_context_map[i] = i;
    }

    // Collapse similar histograms within a block type.

    static const int kMinClustersForHistogramRemap = 24;

    num_clusters = 3;

    // If we did not have block groups , we have to do one final round of
    // clustering.

    hls_HistogramCombine(hls_clustgrams, total_count, max_loop, max_nz_symbol,

                         cluster_size, // 1,1,1
                         bit_cost, dc_context_map,
                         MAX_NUM_COLOR,  // 3
                         max_histograms, // 64
                         hls_clustgrams_out,

                         num_clusters); // 64

    if (dc_context_map[0] == 0 && (dc_context_map[1] == 0) && (dc_context_map[2] == 2)) {
        dc_context_map[2] = 1;
        max_nz_symbol[1] = max_nz_symbol[2];
    }
#ifndef __SYNTHESIS__
    _XF_IMAGE_PRINT("num_clusters= %d\n", (int)(num_clusters));
    _XF_IMAGE_PRINT("---cluster historgram after remap:\n");
    for (int c = 0; c < num_clusters; c++) {
        for (int i = 0; i < (max_nz_symbol[c]); i++) {
            uint32_t tmp = hls_clustgrams_out[c][i];
            _XF_IMAGE_PRINT("%d,", (int)(tmp));
        }
        _XF_IMAGE_PRINT("\n");
    }

#endif
}
