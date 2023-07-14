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

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <mutex>
#include <vitis/ai/weak.hpp>

#include "vart/op_imp.h"
namespace py = pybind11;

namespace {
static std::shared_ptr<::pybind11::scoped_interpreter> init_interpreter() {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  auto ret = std::shared_ptr<::pybind11::scoped_interpreter>();
  if (!Py_IsInitialized()) {
    ret = vitis::ai::WeakSingleton<py::scoped_interpreter>::create();
    CHECK(ret != nullptr) << "cannot create python interpreter";
  }
  py::module::import("xir");
  // we have to import vart, otherwise, vart::TensorBuffer is not accessible
  // from Python.
  py::module::import("vart");
  return ret;
}

class PythonCpuOp : public vart::OpImp {
 public:
  explicit PythonCpuOp(const xir::Op* op, xir::Attrs* attrs);
  virtual ~PythonCpuOp();
  PythonCpuOp(const PythonCpuOp& other) = delete;
  PythonCpuOp& operator=(const PythonCpuOp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<vart::OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;

 private:
  std::shared_ptr<::pybind11::scoped_interpreter> interpreter_;
  py::object the_op_imp_;
  py::object the_op_imp_method_;
};

PythonCpuOp::PythonCpuOp(const xir::Op* op, xir::Attrs* attrs)
    : vart::OpImp(op), interpreter_{init_interpreter()} {
  auto op_type = op->get_type();
  // we add a namespace here. I think it is OK to import a module many times.
  auto m = py::module::import((std::string("vart_op_imp.") + op_type).c_str());
  // find the proper python class and create a new python object.
  auto op_type_cls = m.attr(op_type.c_str());
  the_op_imp_ = op_type_cls(py::cast(op));
  // find the member function.
  the_op_imp_method_ = the_op_imp_.attr("calculate");
};
PythonCpuOp::~PythonCpuOp() {}

int PythonCpuOp::calculate(const std::vector<vart::OpImpArg>& inputs,
                           vart::TensorBuffer* output_tensor_buffer) {
  // it is important to get gil again.
  py::gil_scoped_acquire acquire;
  py::dict kwargs;
  for (auto& input : inputs) {
    kwargs[input.arg_name.c_str()] =
        py::cast(input.args, py::return_value_policy::reference);
  }
  kwargs["output"] =
      py::cast(output_tensor_buffer, py::return_value_policy::reference);
  py::tuple args = py::make_tuple();
  the_op_imp_method_(*args, **kwargs);
  return 0;
}

}  // namespace

extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::make_vart_opt_imp<PythonCpuOp>();
}
