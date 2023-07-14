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
#include "xir/xrt_device_handle.hpp"

#include <map>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE, "0");

namespace xir {

static std::map<std::string, std::function<std::shared_ptr<XrtDeviceHandle>()>>& get_factory_methods()
{
    static std::map<std::string, std::function<std::shared_ptr<XrtDeviceHandle>()>>
        the_factory_methods;
    return the_factory_methods;
}

void XrtDeviceHandle::registar(
    const std::string& name,
    std::function<std::shared_ptr<XrtDeviceHandle>()> m) {
  auto it = get_factory_methods().begin();
  auto ok = false;
  std::tie(it, ok) = get_factory_methods().emplace(std::make_pair(name, m));
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << "add factory method " << name;
  CHECK(ok);
}

std::shared_ptr<XrtDeviceHandle> XrtDeviceHandle::get_instance() {
  CHECK(!get_factory_methods().empty());
  auto ret = get_factory_methods().begin()->second();
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << "return the xrt handle instance via "
      << get_factory_methods().begin()->first << " "
      << " ret=" << (void*)ret.get();
  return ret;
}

}  // namespace xir

/*std::unique_ptr<xir::XrtDeviceHandle> my_xir_device_handle_create();
std::unique_ptr<xir::XrtDeviceHandle> xir::XrtDeviceHandle::create() {
  return my_xir_device_handle_create();
}
*/
