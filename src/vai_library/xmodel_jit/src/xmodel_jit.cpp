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
#include "vitis/ai/xmodel_jit.hpp"

#include <dlfcn.h>

namespace vitis {
namespace ai {
std::unique_ptr<XmodelJit> XmodelJit::create(xir::Graph* graph) {
  auto jit = std::string("libvitis_ai_library-xmodel_jit_python.so.3");
  if (graph->has_attr("xmodel_image:jit")) {
    jit = graph->get_attr<std::string>("xmodel_image:jit");
  }
  auto so_name = jit;
  auto handle = dlopen(so_name.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!handle) {
    LOG(FATAL) << "cannot open plugin: name=" << so_name
               << " error: " << dlerror();
  };
  typedef std::unique_ptr<XmodelJit> (*fm_type)(xir::Graph * graph);
  auto factory_method_p = (fm_type)dlsym(handle, "create_xmodel_jit");
  if (factory_method_p == nullptr) {
    LOG(FATAL) << "not a valid plugin, cannot find symbol "
                  "\"create_xmodel_jit\": name="
               << so_name;
  }
  auto ret = (*factory_method_p)(graph);
  CHECK(ret != nullptr) << "plugin return a nullptr."
                           ";name="
                        << so_name;
  return ret;
}
}  // namespace ai
}  // namespace vitis
