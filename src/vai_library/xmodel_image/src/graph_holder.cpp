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
#include "./graph_holder.hpp"

#include <glog/logging.h>
#include <libgen.h>
#include <limits.h>
#include <stdlib.h>

#include <fstream>

#include "vitis/ai/path_util.hpp"
#include "vitis/ai/xmodel_jit.hpp"

namespace vitis {
namespace ai {

GraphHolder::GraphHolder(const std::string& filename) {
  auto realpath = filename;  // vitis::ai::file_name_realpath(filename);
  graph = xir::Graph::deserialize(realpath);
  auto dirname = vitis::ai::file_name_directory(realpath);
  auto basename = vitis::ai::file_name_basename(realpath);
  graph->set_attr<std::string>("__file__", filename);
  graph->set_attr<std::string>("__dir__", dirname);
  graph->set_attr<std::string>("__basename__", basename);
  auto jit = XmodelJit::create(graph.get());
  auto ok = jit->jit();
  CHECK(ok == 0);
}

}  // namespace ai
}  // namespace vitis
