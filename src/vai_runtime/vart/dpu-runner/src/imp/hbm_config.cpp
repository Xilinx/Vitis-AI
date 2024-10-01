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
#include "./hbm_config.hpp"

#include <glog/logging.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/parse_value.hpp>

#include "vitis/ai/simple_config.hpp"

DEF_ENV_PARAM_2(XLNX_MAT_CONFIG, "/usr/lib/hbm_address_assignment.txt",
                std::string)
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0")

namespace vart {
namespace dpu {
const std::vector<HbmChannelProperty>& HBM_CHANNELS();
const char* HARDWARE_DEF =
    // "#CU Identifier Offset Size           \n"
    // 201920_2 bing
    "0          D0              0x000000000     0x010000000        \n"
    "1          D0              0x010000000     0x010000000        \n"
    "1          D3              0x020000000     0x010000000        \n"
    "0          W0              0x030000000     0x010000000        \n"
    "0          W1              0x040000000     0x010000000        \n"
    "1          W0              0x050000000     0x010000000        \n"
    "1          W1              0x060000000     0x010000000        \n"
    "0          I               0x070000000     0x010000000        \n"
    "1          I               0x080000000     0x010000000        \n"
    "0          D1              0x100000000     0x010000000        \n"
    "0          D2              0x110000000     0x010000000        \n"
    "1          D1              0x120000000     0x010000000        \n"
    "1          D2              0x130000000     0x010000000        \n";
// 201920_1 bin
/*
"0          D0              0x000000000     0x010000000    \n"
"0          D1              0x100000000     0x010000000    \n"
"0          D2              0x120000000     0x010000000    \n"
"0          I               0x040000000     0x010000000    \n"
"0          W0              0x060000000     0x010000000    \n"
"0          W1              0x070000000     0x010000000    \n"
"1          D0              0x010000000     0x010000000    \n"
"1          D1              0x110000000     0x010000000    \n"
"1          D2              0x130000000     0x010000000    \n"
"1          D3              0x030000000     0x010000000    \n"
"1          I               0x050000000     0x010000000    \n"
"1          W0              0x080000000     0x010000000    \n"
"1          W1              0x090000000     0x010000000    \n";
*/
/*
static std::vector<std::string> fine_mat_search_path() {
  auto ret = std::vector<std::string>{};
  ret.push_back("./");
  // ret.push_back("/usr/share/config/");
  return ret;
}

static size_t filesize(const std::string& filename) {
  size_t ret = 0;
  struct stat statbuf;
  const auto r_stat = stat(filename.c_str(), &statbuf);
  if (r_stat == 0) {
    ret = statbuf.st_size;
  }
  return ret;
}
*/
static std::string get_mat_file_name() {
  std::string mat_name = ENV_PARAM(XLNX_MAT_CONFIG);
  /*for (const auto& p : fine_mat_search_path()) {
    const auto fullname = p + mat_name;
    if (filesize(fullname) > 0) {
      return fullname;
    }
  }
  std::stringstream str;
  str << "cannot find mat_file <" << mat_name
      << "> after checking following files:";
  for (const auto& p : fine_mat_search_path()) {
    const auto fullname = p + mat_name;
    str << "\n\t" << fullname;
  }
  LOG(WARNING) << str.str();
  abort();
  */
  return mat_name;
  // return std::string{""};
}
/*
static std::unique_ptr<std::ifstream> get_stream_for_mat() {
  // return std::make_unique<std::istringstrea>(std::string(HARDWARE_DEF));
  return std::make_unique<std::ifstream>(get_mat_file_name());
}

*/

std::vector<HbmChannelProperty> get_hbm_config_from_hbm_txt() {
  auto ret = std::vector<HbmChannelProperty>();
  if (ret.empty()) {
    auto mat_file_name = get_mat_file_name();
    std::unique_ptr<std::istream> stream =
        std::make_unique<std::ifstream>(mat_file_name);
    if (!stream->good()) {
      LOG(WARNING) << "cannot read memory assignment table file: "
                   << mat_file_name;
      exit(0);
      //                   << ", use default values which might correct.";
      // stream = std::make_unique<std::stringstream>(HARDWARE_DEF);
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
          << "Read MAT CONFIG success:" << mat_file_name;  //
    }
    unsigned int core_id = 0;
    std::string name = "";
    std::string offset;
    std::string capacity;

    while ((*stream >> core_id >> name >> offset >> capacity).good()) {
      uint64_t offset2;
      uint64_t capacity2;
      vitis::ai::parse_value(offset, offset2);
      vitis::ai::parse_value(capacity, capacity2);
      ret.emplace_back(HbmChannelProperty{name,
                                          core_id,
                                          {{
                                              offset2,
                                              capacity2,
                                              _4K,
                                          }}});
    }
  }
  return ret;
}

const std::string get_dpu_xclbin() {
  auto ret = std::string("/usr/lib/dpu.xclbin");
  if (!ENV_PARAM(XLNX_VART_FIRMWARE).empty()) {
    ret = ENV_PARAM(XLNX_VART_FIRMWARE);
    return ret;
  }
  auto config =
      vitis::ai::SimpleConfig::getOrCreateSimpleConfig("/etc/vart.conf");
  if (!config) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
        << "/etc/vart.conf does not exits. use default value "
           "/usr/lib/dpu.xclbin";
    return ret;
  }
  auto has_firmware = (*config).has("firmware");
  if (!has_firmware) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
        << "/etc/vart.conf does not contains firmware: xxx. use default value "
           "/usr/lib/dpu.xclbin";
    return ret;
  }
  ret = (*config)("firmware").as<std::string>();
  return ret;
}

std::vector<HbmChannelProperty> get_hbm_channels() {
  auto ret = std::vector<HbmChannelProperty>();
  if (ret.empty()) {
    ret = get_hbm_config_from_hbm_txt();
  }

  if (ENV_PARAM(DEBUG_DPU_RUNNER)) {
    LOG(INFO) << "HBM config is: ";
    for (auto& hbm : ret) {
      LOG(INFO) << hbm;
    }
  }
  return ret;
}

const std::vector<HbmChannelProperty>& HBM_CHANNELS() {
  static auto ret = get_hbm_channels();
  return ret;
}

std::map<std::string, chunk_def_t> get_hbm(size_t core_id) {
  auto ret = std::map<std::string, chunk_def_t>();
  const auto& all = vart::dpu::HBM_CHANNELS();
  for (const auto& hbm : all) {
    auto used_by_core = core_id == hbm.core_id;
    if (used_by_core) {
      ret.insert(std::make_pair(
          hbm.name, std::vector<vart::dpu::hbm_channel_def_t>(hbm.channels_)));
    }
  }
  return ret;
}

// ret[name]
std::vector<chunk_def_t> get_engine_hbm(size_t core_id) {
  auto ret = std::vector<chunk_def_t>();
  auto chunks_used_by_core = std::vector<std::pair<std::string, chunk_def_t>>();
  const auto& all = vart::dpu::HBM_CHANNELS();
  for (const auto& hbm : all) {
    auto used_by_core = core_id == hbm.core_id;
    if (used_by_core && hbm.name[0] == 'D') {
      chunks_used_by_core.push_back(std::make_pair(hbm.name, hbm.channels_));
    }
  }

  std::sort(chunks_used_by_core.begin(), chunks_used_by_core.end(),
            [](std::pair<std::string, chunk_def_t> a,
               std::pair<std::string, chunk_def_t> b) {
              return a.first.compare(b.first) < 0;
            });

  for (const auto& chunk : chunks_used_by_core) {
    ret.push_back(chunk.second);
  }

  if (ENV_PARAM(DEBUG_DPU_RUNNER)) {
    LOG(INFO) << "sort used HBM by name: ";
    for (auto& chunk : ret) {
      LOG(INFO) << chunk;
    }
  }

  return ret;
}
/*
size_t get_core_num() {
  const auto& all = vart::dpu::HBM_CHANNELS();
  auto core_num = 0;
  for (const auto& hbm : all) {
    if (hbm.name[0] == "I") {
      core_num++;
    }
  }
  return core_num;
 }
size_t get_batch_num(size_t core_id) { return get_engine_hbm(core_id).size(); }
*/
}  // namespace dpu
}  // namespace vart
std::ostream& operator<<(std::ostream& out,
                         const vart::dpu::HbmChannelProperty& hbm) {
  out << "HBM{"
      << "name=" << hbm.name << ","
      << "core_id=" << hbm.core_id << " ";
  for (const auto& chunk : hbm.channels_) {
    out << ", (" << std::hex << "0x" << chunk.offset << ","
        << "0x" << chunk.capacity << ")" << std::dec;
  }
  out << "}";
  return out;
}

std::ostream& operator<<(std::ostream& out, const vart::dpu::chunk_def_t& def) {
  out << std::hex;
  out << "HBM_DEF{";
  for (const auto& ch : def) {
    out << "("
        << "0x" << ch.offset << ","
        << "0x" << ch.capacity << ","
        << "0x" << ch.alignment << ")";
  }
  out << "}";
  out << std::dec;
  return out;
}
