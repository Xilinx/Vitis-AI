/*
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
#pragma once
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>

#include "vitis/ai/xmodel_jit.hpp"
namespace py = pybind11;
namespace vitis {
namespace ai {
class
    // without this, we get strange warning, declared with greater
    // visibility than the type of its field
    // see
    // https://stackoverflow.com/questions/49252686/exported-class-with-non-exported-data-member
    // https://github.com/vgc/vgc/issues/11
    __attribute__((visibility("hidden"))) XmodelJitPython : public XmodelJit {
 public:
  explicit XmodelJitPython(xir::Graph* graph);
  virtual ~XmodelJitPython() = default;
  XmodelJitPython(const XmodelJitPython& other) = delete;
  XmodelJitPython& operator=(const XmodelJitPython& rhs) = delete;

 private:
  virtual int jit() override;

 private:
  xir::Graph* graph_;
  std::shared_ptr<::pybind11::scoped_interpreter> interpreter_;
};
}  // namespace ai
}  // namespace vitis
