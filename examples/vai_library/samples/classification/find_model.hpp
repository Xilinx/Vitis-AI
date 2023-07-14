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

#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <vitis/ai/env_config.hpp>

using namespace std;
DEF_ENV_PARAM_2(VAI_LIBRARY_MODELS_DIR, ".", std::string)

static vector<string> find_model_search_path() {
  auto ret = vector<string>{};
  ret.push_back(".");
  ret.push_back(ENV_PARAM(VAI_LIBRARY_MODELS_DIR));
  ret.push_back("/usr/share/vitis_ai_library/models");
  ret.push_back("/usr/share/vitis_ai_library/.models");
  return ret;
}

static size_t filesize(const string& filename) {
  size_t ret = 0;
  struct stat statbuf;
  const auto r_stat = stat(filename.c_str(), &statbuf);
  if (r_stat == 0) {
    ret = statbuf.st_size;
  }
  return ret;
}

static string find_model(const string& name) {
  if (filesize(name) > 4096u) {
    return name;
  }

  auto ret = std::string();
  for (const auto& p : find_model_search_path()) {
    ret = p + "/" + name + "/" + name;
    const auto xmodel_name = ret + ".xmodel";
    if (filesize(xmodel_name) > 0u) {
      return xmodel_name;
    }
    const auto elf_name = ret + ".elf";
    if (filesize(elf_name) > 0u) {
      return elf_name;
    }
  }
  stringstream str;
  str << "cannot find model <" << name << "> after checking following dir:";
  for (const auto& p : find_model_search_path()) {
    str << "\n\t" << p;
  }
  LOG(FATAL) << str.str();
  return string{""};
}
