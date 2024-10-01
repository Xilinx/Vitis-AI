/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <pybind11/pybind11.h>

#include "vitis/ai/graph_runner.hpp"

#ifndef MODULE_NAME
#define MODULE_NAME vitis_ai_library
#endif

namespace py = pybind11;
PYBIND11_MODULE(MODULE_NAME, m) {
  m.doc() = "vitis_ai_library::GraphRunner inferace";
  py::module::import("vart");
  py::class_<vitis::ai::GraphRunner>(m, "GraphRunner")
      .def_static(
          "create_graph_runner",
          [](const xir::Graph* graph) {
            return vitis::ai::GraphRunner::create_graph_runner(graph, nullptr);
          },
          py::arg("graph"));
}
