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
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "./hbm_manager.hpp"
namespace vart {
namespace dpu {

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
