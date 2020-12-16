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
#include <glog/logging.h>

#include <map>
#include <memory>
#if CROSSCOMPILING
#include "./xclbin_info.hpp"
#else
#include "./xclbin_info_imp.hpp"
#endif

namespace vart {
namespace dpu {

std::unique_ptr<XclbinInfo> XclbinInfo::create(const std::string& xclbin_file) {
#if CROSSCOMPILING
  LOG(FATAL) << "NOT IMPLEMENTATED";
  return nullptr;
#else
  return std::unique_ptr<XclbinInfo>(new XclbinInfoImp(xclbin_file));
#endif
}

}  // namespace dpu
}  // namespace vart
