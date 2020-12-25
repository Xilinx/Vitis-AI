/*
 * Copyright 2019 Xilinx Inc.
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
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace vart {
namespace dpu {
constexpr uint64_t _1M = 1024ull * 1024ull;
constexpr uint64_t _256M = 256ull * _1M;
constexpr uint64_t _4K = 4ull * 1024ull;
struct hbm_channel_def_t {
  uint64_t offset;
  uint64_t capacity = _256M;
  uint64_t alignment = _4K;
};

using chunk_def_t = std::vector<hbm_channel_def_t>;
struct HbmChannelProperty {
  const std::string name = "N/A";
  const unsigned int core_id = 0;
  chunk_def_t channels_;
};
// for testing
const std::vector<HbmChannelProperty>& HBM_CHANNELS();
std::map<std::string, chunk_def_t> get_hbm(size_t core_id);
std::vector<chunk_def_t> get_engine_hbm(size_t core_id);
// size_t get_batch_num(size_t core_id);
}  // namespace dpu
}  // namespace vart
std::ostream& operator<<(std::ostream& out,
                         const vart::dpu::HbmChannelProperty& hbm);
std::ostream& operator<<(std::ostream& out, const vart::dpu::chunk_def_t& def);
