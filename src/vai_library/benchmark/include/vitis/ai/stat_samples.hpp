/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>
namespace vitis {
namespace ai {
class StatSamples {
 public:
  explicit StatSamples(size_t capacity);
  StatSamples(StatSamples &&other) = default;
  StatSamples(const StatSamples &) = delete;
  StatSamples &operator=(const StatSamples &other) = delete;
  virtual ~StatSamples();

 public:
  void reserve(size_t capacity) { store_.reserve(capacity); }
  void addSample(int value) { store_.push_back(value); }

 public:
  double getMean();
  double getStdVar(const double mean);
  void merge(StatSamples &statSamples);

 private:
  std::vector<int> store_;
};
inline StatSamples::StatSamples(size_t capacity) { store_.reserve(capacity); }

inline StatSamples::~StatSamples() {}

inline double StatSamples::getMean() {
  double sum = std::accumulate(store_.begin(), store_.end(), 0.0);
  return sum / store_.size();
}

inline double StatSamples::getStdVar(const double mean) {
  double accum = 0.0;
  std::for_each(store_.begin(), store_.end(),
                [&](const int d) { accum += (d - mean) * (d - mean); });
  return std::sqrt(accum / store_.size());
}

inline void StatSamples::merge(StatSamples &statSamples) {
  store_.insert(store_.end(), statSamples.store_.begin(),
                statSamples.store_.end());
}
}  // namespace ai
}  // namespace vitis
