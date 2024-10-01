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
#include "./xmodel_jit_python.hpp"

#include <pyport.h>

#include <memory>
#include <mutex>
#include <vitis/ai/weak.hpp>
#include <xir/attrs/attr_def.hpp>
#include <xir/op/op_def.hpp>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/extra_ops.hpp"

using namespace py::literals;

DEF_ENV_PARAM(DEBUG_XMODEL_JIT, "0")

namespace vitis {
namespace ai {

static std::shared_ptr<::pybind11::scoped_interpreter> init_interpreter() {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  auto ret = std::shared_ptr<::pybind11::scoped_interpreter>();
  if (!Py_IsInitialized()) {
    ret = vitis::ai::WeakSingleton<py::scoped_interpreter>::create();
  }
  return ret;
}

XmodelJitPython::XmodelJitPython(xir::Graph* graph)
    : XmodelJit{}, graph_{graph}, interpreter_{init_interpreter()} {
  maybe_add_op_defs();
}

static std::string remove_ext(const std::string& basename) {
  auto pos = basename.rfind('.');
  if (pos != std::string::npos) {
    return basename.substr(0, pos);
  }
  return basename;
}

static std::string get_script_name(const std::string& dirname,
                                   const std::string& basename) {
  auto b1 = remove_ext(basename);
  return dirname + "/" + b1 + ".py";
}

int XmodelJitPython::jit() {
  py::module sys = py::module::import("sys");
  py::list sys_path = sys.attr("path");
  auto dirname = graph_->get_attr<std::string>("__dir__");
  auto filename = graph_->get_attr<std::string>("__file__");
  auto basename = graph_->get_attr<std::string>("__basename__");
  sys_path.insert(0, py::str(dirname));
  LOG_IF(INFO, ENV_PARAM(DEBUG_XMODEL_JIT))              //
      << "graph.file = " << filename                     //
      << ";graph.dirname = " << dirname                  //
      << ";sys.path=" << std::string(py::str(sys_path))  //
      ;
  py::module jit_module;
  try {
    // jit_module = py::module::import("xmodel_jit");
    py::module::import("xir");
    auto script_name = get_script_name(dirname, basename);
    py::eval_file(script_name, py::globals());
  } catch (py::error_already_set& e) {
    LOG(WARNING) << "cannot find py:" << e.what();
    return 1;
  } catch (std::runtime_error& e) {
    LOG(WARNING) << "eval python code error:" << e.what();
    return 1;
  }
  try {
    py::object jit_fun = py::globals()["jit"];
    jit_fun(graph_);
  } catch (py::error_already_set& e) {
    LOG_IF(WARNING, true || ENV_PARAM(DEBUG_XMODEL_JIT))
        << "execute jit exception: " << std::string(py::str(e.value()));
    return 2;
  }
  return 0;
}
}  // namespace ai
}  // namespace vitis

extern "C" std::unique_ptr<vitis::ai::XmodelJit> create_xmodel_jit(
    xir::Graph* graph) {
  return std::make_unique<vitis::ai::XmodelJitPython>(graph);
}
