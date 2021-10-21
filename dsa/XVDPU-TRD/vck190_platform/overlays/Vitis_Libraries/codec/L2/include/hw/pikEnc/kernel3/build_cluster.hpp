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
 * @file build_cluster.hpp
 */

#ifndef _XF_CODEC_BUILD_CLUSTER_HPP
#define _XF_CODEC_BUILD_CLUSTER_HPP

#include "kernel3/build_table_encode_histo.hpp"
#include "kernel3/kernel3_common.hpp"

void hls_ClusterHistograms_top(const hist_t hls_clustgrams[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                               int max_histograms,
                               int& num_clusters,
                               uint8_t max_nz_symbol[MAX_NUM_COLOR],
                               hist_t hls_clustgrams_out[hls_kNumStaticContexts][MAX_ALPHABET_SIZE],
                               uint8_t dc_context_map[MAX_NUM_COLOR]);

float hls_ANSPopulationCost(const hist_t* data, int alphabet_size, int total_count);

#endif
