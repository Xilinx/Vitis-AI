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
#include <algorithm>
#include <memory>

#include "./hbm_config.hpp"
namespace vart {
namespace dpu {

class XclbinInfo {
 public:
  static std::unique_ptr<XclbinInfo> create(const std::string& xclbin_file);

 public:
  explicit XclbinInfo() = default;

 public:
  XclbinInfo(const XclbinInfo&) = delete;
  XclbinInfo& operator=(const XclbinInfo& other) = delete;

  virtual ~XclbinInfo() = default;

 public:
  virtual void show_hbm() = 0;
  virtual void show_instances() = 0;
  virtual std::vector<HbmChannelProperty> HBM_CHANNELS() = 0;
};

}  // namespace dpu
}  // namespace vart
