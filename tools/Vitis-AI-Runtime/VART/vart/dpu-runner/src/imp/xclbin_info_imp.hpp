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
#include <string>
#include <utility>

#include "../../../xrt-device-handle/xclbinutil/Section.h"
#include "../../../xrt-device-handle/xclbinutil/XclBinClass.h"
#include "./hbm_config.hpp"
#include "./xclbin_info.hpp"

namespace vart {
namespace dpu {

class XclbinInfoImp : public XclbinInfo {
 public:
  XclbinInfoImp(const std::string& xclbin_file);
  XclbinInfoImp(const XclbinInfoImp&) = delete;
  XclbinInfoImp& operator=(const XclbinInfoImp& other) = delete;

  virtual ~XclbinInfoImp();

 private:
  virtual void show_hbm() override;
  virtual void show_instances() override;
  virtual std::vector<HbmChannelProperty> HBM_CHANNELS() override;

 private:
  XclBin xclbin_;
  std::vector<Section*> sections_;
  std::unordered_map<std::string, std::pair<std::string, std::uint64_t>>
      hbm_address_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      instances_;
};

}  // namespace dpu
}  // namespace vart
