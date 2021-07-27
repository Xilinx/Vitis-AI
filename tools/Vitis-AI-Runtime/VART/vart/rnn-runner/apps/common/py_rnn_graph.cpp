/*
 * Copyright 2021 Xilinx Inc.
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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

#include "rnn_graph.hpp"

namespace py = pybind11;

PYBIND11_MODULE(py_rnn_graph, m) {
  m.doc() = "Create XIR Graph for XRNN";
  m.def("create_rnn_graph", &create_rnn_graph,
        "Generate a rnn graph for rnn-runner");
}
