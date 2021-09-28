// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// Functions for clustering similar histograms together.

#ifndef PIK_CLUSTER_H_
#define PIK_CLUSTER_H_

#include <stdint.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "pik/fast_log.h"

namespace pik {

struct HistogramPair {
  int idx1;
  int idx2;
  float cost_combo;
  float cost_diff;
};

inline bool operator<(const HistogramPair& p1, const HistogramPair& p2) {
  if (p1.cost_diff != p2.cost_diff) {
    return p1.cost_diff > p2.cost_diff;
  }
  return std::abs(p1.idx1 - p1.idx2) > std::abs(p2.idx1 - p2.idx2);
}

// Returns entropy reduction of the context map when we combine two clusters.
inline float ClusterCostDiff(int size_a, int size_b) {
  int size_c = size_a + size_b;
  return size_a * FastLog2(size_a) + size_b * FastLog2(size_b) -
         size_c * FastLog2(size_c);
}

// Computes the bit cost reduction by combining out[idx1] and out[idx2] and if
// it is below a threshold, stores the pair (idx1, idx2) in the *pairs queue.
template <typename HistogramType>
void CompareAndPushToQueue(const HistogramType* out, const int* cluster_size,
                           const float* bit_cost, int idx1, int idx2,
                           std::vector<HistogramPair>* pairs) {
  if (idx1 == idx2) {
    return;
  }
  if (idx2 < idx1) {
    int t = idx2;
    idx2 = idx1;
    idx1 = t;
  }
  bool store_pair = false;
  HistogramPair p;
  p.idx1 = idx1;
  p.idx2 = idx2;
  p.cost_diff = 0.5f * ClusterCostDiff(cluster_size[idx1], cluster_size[idx2]);
  p.cost_diff -= bit_cost[idx1];
  p.cost_diff -= bit_cost[idx2];

  if (out[idx1].total_count_ == 0) {
    p.cost_combo = bit_cost[idx2];
    store_pair = true;
  } else if (out[idx2].total_count_ == 0) {
    p.cost_combo = bit_cost[idx1];
    store_pair = true;
  } else {
    const float threshold = pairs->empty()
                                ? std::numeric_limits<float>::max()
                                : std::max(0.0f, (*pairs)[0].cost_diff);
    HistogramType combo = out[idx1];
    combo.AddHistogram(out[idx2]);
    float cost_combo = combo.PopulationCost();
    if (cost_combo + p.cost_diff < threshold) {
      p.cost_combo = cost_combo;
      store_pair = true;
    }
  }
  if (store_pair) {
    p.cost_diff += p.cost_combo;
    if (!pairs->empty() && (pairs->front() < p)) {
      // Replace the top of the queue if needed.
      pairs->push_back(pairs->front());
      pairs->front() = p;
    } else {
      pairs->push_back(p);
    }
  }
}

template <typename HistogramType>
int HistogramCombine(HistogramType* out, int* cluster_size, float* bit_cost,
                     uint32_t* symbols, int symbols_size, int max_clusters) {
  float cost_diff_threshold = 0.0f;
  int min_cluster_size = 1;

  // Uniquify the list of symbols after merging empty clusters.
  std::vector<int> clusters;
  clusters.reserve(symbols_size);
  int sum_of_totals = 0;
  int first_zero_pop_count_symbol = -1;
  for (int i = 0; i < symbols_size; ++i) {
    if (out[symbols[i]].total_count_ == 0) {
      // Merge the zero pop count histograms into one.
      if (first_zero_pop_count_symbol == -1) {
        first_zero_pop_count_symbol = symbols[i];
        clusters.push_back(symbols[i]);
      } else {
        symbols[i] = first_zero_pop_count_symbol;
      }
    } else {
      // Insert all histograms with non-zero pop counts.
      clusters.push_back(symbols[i]);
      sum_of_totals += out[symbols[i]].total_count_;
    }
  }
  if (sum_of_totals < 160) {
    // Use a single histogram if there are only a few samples.
    // This helps with small images (like 64x64 size) where the
    // context map is more expensive than the related savings.
    // TODO: Estimate the the actual difference in bitcost to
    // make the final decision of this strategy and clustering.
    *cluster_size = 1;
    HistogramType combo = out[symbols[0]];
    for (int i = 1; i < symbols_size; ++i) {
      combo.AddHistogram(out[symbols[i]]);
    }
    out[symbols[0]] = combo;
    for (int i = 1; i < symbols_size; ++i) {
      symbols[i] = symbols[0];
    }
    return 1;
  }
  std::sort(clusters.begin(), clusters.end());
  clusters.resize(std::unique(clusters.begin(), clusters.end()) -
                  clusters.begin());

  // We maintain a priority queue of histogram pairs, ordered by the bit cost
  // reduction. For efficiency, only the front of the queue matters, the rest of
  // it is unordered.
  std::vector<HistogramPair> pairs;
  for (int idx1 = 0; idx1 < clusters.size(); ++idx1) {
    for (int idx2 = idx1 + 1; idx2 < clusters.size(); ++idx2) {
      CompareAndPushToQueue(out, cluster_size, bit_cost, clusters[idx1],
                            clusters[idx2], &pairs);
    }
  }

  while (clusters.size() > min_cluster_size) {
    if (pairs[0].cost_diff >= cost_diff_threshold) {
      cost_diff_threshold = std::numeric_limits<float>::max();
      min_cluster_size = max_clusters;
      continue;
    }

    // Take the best pair from the top of queue.
    int best_idx1 = pairs[0].idx1;
    int best_idx2 = pairs[0].idx2;
    out[best_idx1].AddHistogram(out[best_idx2]);
    bit_cost[best_idx1] = pairs[0].cost_combo;
    cluster_size[best_idx1] += cluster_size[best_idx2];
    for (int i = 0; i < symbols_size; ++i) {
      if (symbols[i] == best_idx2) {
        symbols[i] = best_idx1;
      }
    }
    for (auto cluster = clusters.begin(); cluster != clusters.end();
         ++cluster) {
      if (*cluster >= best_idx2) {
        clusters.erase(cluster);
        break;
      }
    }

    // Remove pairs intersecting the just combined best pair.
    auto copy_to = pairs.begin();
    for (int i = 0; i < pairs.size(); ++i) {
      HistogramPair& p = pairs[i];
      if (p.idx1 == best_idx1 || p.idx2 == best_idx1 || p.idx1 == best_idx2 ||
          p.idx2 == best_idx2) {
        // Remove invalid pair from the queue.
        continue;
      }
      if (pairs.front() < p) {
        // Replace the top of the queue if needed.
        auto front = pairs.front();
        pairs.front() = p;
        *copy_to = front;
      } else {
        *copy_to = p;
      }
      ++copy_to;
    }
    pairs.resize(copy_to - pairs.begin());

    // Push new pairs formed with the combined histogram to the queue.
    for (int i = 0; i < clusters.size(); ++i) {
      CompareAndPushToQueue(out, cluster_size, bit_cost, best_idx1, clusters[i],
                            &pairs);
    }
  }
  return clusters.size();
}

// -----------------------------------------------------------------------------
// Histogram refinement

// What is the bit cost of moving histogram from cur_symbol to candidate.
template <typename HistogramType>
float HistogramBitCostDistance(const HistogramType& histogram,
                               const HistogramType& candidate,
                               const float candidate_bit_cost) {
  if (histogram.total_count_ == 0) {
    return 0.0;
  }
  HistogramType tmp = histogram;
  tmp.AddHistogram(candidate);
  return tmp.PopulationCost() - candidate_bit_cost;
}

// Find the best 'out' histogram for each of the 'in' histograms.
// Note: we assume that out[]->bit_cost_ is already up-to-date.
template <typename HistogramType>
void HistogramRemap(const HistogramType* in, int in_size, HistogramType* out,
                    float* bit_cost, uint32_t* symbols) {
  // Uniquify the list of symbols.
  std::vector<int> all_symbols(symbols, symbols + in_size);
  std::sort(all_symbols.begin(), all_symbols.end());
  all_symbols.resize(std::unique(all_symbols.begin(), all_symbols.end()) -
                     all_symbols.begin());

  for (int i = 0; i < in_size; ++i) {
    int best_out = i == 0 ? symbols[0] : symbols[i - 1];
    float best_bits =
        HistogramBitCostDistance(in[i], out[best_out], bit_cost[best_out]);
    for (auto k : all_symbols) {
      const float cur_bits =
          HistogramBitCostDistance(in[i], out[k], bit_cost[k]);
      if (cur_bits < best_bits) {
        best_bits = cur_bits;
        best_out = k;
      }
    }
    symbols[i] = best_out;
  }

  // Recompute each out based on raw and symbols.
  for (auto k : all_symbols) {
    out[k].Clear();
  }
  for (int i = 0; i < in_size; ++i) {
    out[symbols[i]].AddHistogram(in[i]);
  }
}

// Reorder histograms in *out so that the new symbols in *symbols come in
// increasing order.
template <typename HistogramType>
void HistogramReindex(std::vector<HistogramType>* out,
                      std::vector<uint32_t>* symbols) {
  std::vector<HistogramType> tmp(*out);
  std::map<int, int> new_index;
  int next_index = 0;
  for (int i = 0; i < symbols->size(); ++i) {
    if (new_index.find((*symbols)[i]) == new_index.end()) {
      new_index[(*symbols)[i]] = next_index;
      (*out)[next_index] = tmp[(*symbols)[i]];
      ++next_index;
    }
  }
  out->resize(next_index);
  for (int i = 0; i < symbols->size(); ++i) {
    (*symbols)[i] = new_index[(*symbols)[i]];
  }
}

// Clusters similar histograms in 'in' together, the selected histograms are
// placed in 'out', and for each index in 'in', *histogram_symbols will
// indicate which of the 'out' histograms is the best approximation.
// The template parameter HistogramType needs to have Clear(), AddHistogram(),
// and PopulationCost() methods.
template <typename HistogramType>
void ClusterHistograms(const std::vector<HistogramType>& in, int num_contexts,
                       int num_blocks,
                       const std::vector<int> block_group_offsets,
                       int max_histograms, std::vector<HistogramType>* out,
                       std::vector<uint32_t>* histogram_symbols) {
  const int in_size = num_contexts * num_blocks;
  std::vector<int> cluster_size(in_size, 1);
  std::vector<float> bit_cost(in_size);
  out->resize(in_size);
  histogram_symbols->resize(in_size);
  for (int i = 0; i < in_size; ++i) {
    (*out)[i] = in[i];
    bit_cost[i] = in[i].PopulationCost();
    (*histogram_symbols)[i] = i;
  }

  // Collapse similar histograms within a block type.
  if (num_contexts > 1) {
    for (int i = 0; i < num_blocks; ++i) {
      HistogramCombine(&(*out)[0], &cluster_size[0], &bit_cost[0],
                       &(*histogram_symbols)[i * num_contexts], num_contexts,
                       max_histograms);
    }
  }

  static const int kMinClustersForHistogramRemap = 24;

  int num_clusters = 0;
  if (block_group_offsets.size() > 1) {
    // Collapse similar histograms within block groups.
    for (int i = 0; i < block_group_offsets.size(); ++i) {
      int offset = block_group_offsets[i] * num_contexts;
      int length = ((i + 1 < block_group_offsets.size()
                         ? block_group_offsets[i + 1] * num_contexts
                         : in_size) -
                    offset);
      int nclusters = HistogramCombine(
          &(*out)[0], &cluster_size[0], &bit_cost[0],
          &(*histogram_symbols)[offset], length, max_histograms);
      // Find the optimal map from original histograms to the final ones.
      if (nclusters >= 2 && nclusters < kMinClustersForHistogramRemap) {
        HistogramRemap(&in[offset], length, &(*out)[0], &bit_cost[0],
                       &(*histogram_symbols)[offset]);
      }
      num_clusters += nclusters;
    }
  }

  if (block_group_offsets.size() <= 1 || num_clusters > max_histograms) {
    // If we did not have block groups or the per-block-group clustering ended
    // with too many histograms, we have to do one final round of clustering.
    num_clusters =
        HistogramCombine(&(*out)[0], &cluster_size[0], &bit_cost[0],
                         &(*histogram_symbols)[0], in_size, max_histograms);
    // Find the optimal map from original histograms to the final ones.
    if (num_clusters >= 2 && num_clusters < kMinClustersForHistogramRemap) {
      HistogramRemap(&in[0], in_size, &(*out)[0], &bit_cost[0],
                     &(*histogram_symbols)[0]);
    }
  }

  // Convert the context map to a canonical form.
  HistogramReindex(out, histogram_symbols);
}

}  // namespace pik

#endif  // PIK_CLUSTER_H_
