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

#include <glog/logging.h>

#include <iomanip>
#include <sstream>
#include <vitis/ai/env_config.hpp>

#include "./hbm_manager_imp.hpp"

DEF_ENV_PARAM(DEBUG_HBM_MANAGER, "0");
namespace vart {
namespace dpu {
std::string HbmChunk::to_string() const {
  std::stringstream stream;
  stream << "{" << std::hex << std::setfill('0')  //
         << "0x" << get_offset()                  //
         << ","                                   //
         << std::dec << get_size()                //
         << "}";
  return stream.str();
}
void HbmChunk::upload(xir::DeviceMemory* dm, const void* data, size_t offset,
                      size_t size) const {
  auto abs_addr = get_offset() + offset;
  LOG_IF(INFO, ENV_PARAM(DEBUG_HBM_MANAGER))
      << "upload " << to_string() << " from " << data << " "
      << "offset " << offset << " "  //
      << "size " << size << " "      //;
      << "abs_addr "
      << "0x" << std::hex << abs_addr << std::dec << " "  //
      ;
  auto ok = dm->upload(data, abs_addr, size);
  PCHECK(ok) << "ok = " << ok;
  return;
}
bool HbmChunk::download(xir::DeviceMemory* dm, void* data, size_t offset,
                        size_t size, bool ignore_error) const {
  auto abs_addr = get_offset() + offset;
  LOG_IF(INFO, ENV_PARAM(DEBUG_HBM_MANAGER))
      << "download " << to_string() << " to " << data << " "
      << "offset " << offset << " "      //
      << "size " << size << " "          //
      << "abs_addr " << abs_addr << " "  //
      ;
  auto ok = dm->download(data, abs_addr, size);
  if (!ignore_error) {
    PCHECK(ok) << "ok = " << ok << " ignore error=" << ignore_error;
  }
  return ok;
}

std::unique_ptr<HbmManager> HbmManager::create(uint64_t from, uint64_t size,
                                               uint64_t alignment) {
  return HbmManager::create0(from, size, alignment);
}
std::unique_ptr<HbmManager> HbmManager::create(
    const vart::dpu::chunk_def_t& def) {
  return HbmManager::create0(def);
}
}  // namespace dpu
}  // namespace vart
