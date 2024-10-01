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
#include <dlfcn.h>
#include <stdlib.h>

#include "tools_extra_ops.hpp"
std::vector<std::string> xilinx_version(std::vector<std::string> so_names) {
  typedef char* (*CAC_FUNC)();
  CAC_FUNC cac_func = NULL;
  std::vector<std::string> version_list;
  for (auto & so : so_names) {
    auto handle = dlopen(so.c_str(), RTLD_LAZY);
    if (!handle) {
      version_list.push_back(dlerror());
      continue;
    }

    dlerror();

    cac_func = (CAC_FUNC)dlsym(handle, "xilinx_version");
    if (!cac_func) {
      version_list.push_back(dlerror());
      dlclose(handle);
      continue;
    }

    version_list.push_back(cac_func());
    dlclose(handle);
  }
  return version_list;
}
std::vector<std::string> xilinx_version2(std::vector<std::string> so_names) {
  typedef char* (*CAC_FUNC)();
  CAC_FUNC cac_func = NULL;
  std::vector<std::string> version_list;
  for (auto & so : so_names) {
    auto handle = dlopen(so.c_str(), RTLD_LAZY);
    if (!handle) {
      version_list.push_back("");
      continue;
    }

    dlerror();

    cac_func = (CAC_FUNC)dlsym(handle, "xilinx_version");
    if (!cac_func) {
      version_list.push_back("");
      dlclose(handle);
      continue;
    }

    version_list.push_back(cac_func());
    dlclose(handle);
  }
  return version_list;
}
