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
#include "xir/buffer_object.hpp"

#include <glog/logging.h>

#include <map>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_BUFFER_OBJECT, "0");

// DECLARE_INJECTION_NULLPTR(xir::BufferObject, size_t&);
DECLARE_INJECTION_NULLPTR(xir::BufferObject, size_t&, size_t&,
                          const std::string&);
namespace xir {
std::unique_ptr<BufferObject> BufferObject::create(size_t size,
                                                   size_t device_id,
                                                   const std::string& cu_name) {
  return BufferObject::create0(size, device_id, cu_name);
}

XclBo BufferObject::get_xcl_bo() const { return XclBo{nullptr, 0}; }
}  // namespace xir
