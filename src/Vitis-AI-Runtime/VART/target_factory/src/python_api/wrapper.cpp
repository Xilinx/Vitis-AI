/**
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

// python 3.7 deprecate PyCreateThread, but pybind11 2.2.3 still uses
// this function.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <pybind11/pybind11.h>
#include "vitis/ai/target_factory.hpp"

namespace py = pybind11;

namespace vitis {
namespace ai {

PYBIND11_MODULE(target_factory, m) {
  py::class_<Target>(m, "Target");
  m.def("get_target_by_name",
        [](const std::string& name) { return target_factory()->create(name); });
  m.def("get_target_by_fingerprint", [](const std::uint64_t fingerprint) {
    return target_factory()->create(fingerprint);
  });
  m.def("dump_proto_txt", [](const Target& target, const std::string& file) {
    return target_factory()->dump(target, file);
  });
  m.def("is_registered_target", [](const std::string& name) {
    return target_factory()->is_registered_target(name);
  });
}
}  // namespace ai
}  // namespace vitis
