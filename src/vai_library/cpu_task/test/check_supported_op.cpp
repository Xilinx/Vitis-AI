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
#include <glog/logging.h>
#include <UniLog/UniLog.hpp>

#include <iostream>
#include <xir/graph/graph.hpp>
using namespace std;
static std::string find_dl_lib_for_op(const xir::Op* op) {
  auto ret = std::string("") + "libvart_op_imp_" + op->get_type() + ".so";
  return ret;
}

static bool check_op(const xir::Op* op) {
  auto so_name = find_dl_lib_for_op(op);

  auto handle = dlopen(so_name.c_str(), RTLD_LAZY);
  auto ret = true;
  if (!handle) {
    // LOG(WARNING) << "unsupported op type: " << op->get_type();
    UNI_LOG_WARNING << "unsupported op type: " << op->get_type();
    ret = false;
  } else {
    dlclose(handle);
  }
  return ret;
}

int main(int argc, char* argv[]) {
  auto xmodel_file_name = std::string(argv[1]);
  auto graph = xir::Graph::deserialize(xmodel_file_name);
  auto children = graph->get_root_subgraph()->children_topological_sort();
  auto ok = true;
  for (auto& s : children) {
    if (s->get_attr<std::string>("device") == "CPU") {
      for (auto op : s->get_ops()) {
        ok = check_op(op) && ok;
      }
    }
  }
  return ok ? 0 : 1;
}
