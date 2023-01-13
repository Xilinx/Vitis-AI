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
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
using namespace std;

#include <sstream>
#ifndef MODULE_NAME
#define MODULE_NAME vitis_ai_runner
#endif
#include <xir/graph/subgraph.hpp>
#include <xir/tensor/tensor.hpp>

#include "vart/runner.hpp"
namespace py = pybind11;
template <typename T>
static std::string type_to_string(T type) {
  switch (type) {
    case xir::DataType::INT:
      return "INT";
    case xir::DataType::UINT:
      return "UINT";
    case xir::DataType::XINT:
      return "XINT";
    case xir::DataType::XUINT:
      return "XUINT";
    case xir::DataType::FLOAT:
      return "FLOAT";
    case xir::DataType::UNKNOWN:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

class CpuFlatTensorBuffer : public vart::TensorBuffer {
 public:
  explicit CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor)
      : TensorBuffer{tensor}, data_{data} {}
  virtual ~CpuFlatTensorBuffer() = default;

 public:
  virtual std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override {
    uint32_t size = std::ceil(tensor_->get_data_type().bit_width / 8.f);
    if (idx.size() == 0) {
      return {reinterpret_cast<uint64_t>(data_),
              tensor_->get_element_num() * size};
    }
    auto dims = tensor_->get_dims();
    auto offset = 0;
    for (auto k = 0; k < tensor_->get_dim_num(); k++) {
      auto stride = 1;
      for (auto m = k + 1; m < tensor_->get_dim_num(); m++) {
        stride *= dims[m];
      }
      offset += idx[k] * stride;
    }
    auto elem_num = tensor_->get_element_num();
    return {reinterpret_cast<uint64_t>(data_) + offset * size,
            (elem_num - offset) * size};
  }

 private:
  void* data_;
};

static vart::TensorBuffer* array_to_tensor_buffer(py::array_t<float>& a,
                                                  const xir::Tensor* tensor) {
  // TODO check size
  // TODO check dimension
  //
  auto dims = std::vector<std::int32_t>();
  dims.reserve(a.ndim());
  for (auto i = 0; i < a.ndim(); ++i) {
    dims.push_back(a.shape(i));
  }
  void* p = a.mutable_data();
  auto name = tensor->get_name();
  return new CpuFlatTensorBuffer(
      p, xir::Tensor::create(name, dims,
                             {xir::DataType::FLOAT, sizeof(float) * 8u})
             .release());
}

static void destroy(vart::TensorBuffer* tb) {
  delete tb->get_tensor();
  delete tb;
}

static vector<vart::TensorBuffer*> array_to_tensor_buffer(
    const std::vector<py::array_t<float>>& a,
    const std::vector<const xir::Tensor*> tensors) {
  auto ret = vector<vart::TensorBuffer*>{};
  ret.reserve(a.size());
  auto c = 0u;
  for (auto x : a) {
    ret.emplace_back(array_to_tensor_buffer(x, tensors[c++]));
  }
  return ret;
}

static void destroy(const std::vector<vart::TensorBuffer*>& tb) {
  for (auto x : tb) {
    destroy(x);
  }
}
PYBIND11_MODULE(MODULE_NAME, m) {
  m.doc() = "dpu runner";  // optional module docstring
  py::object xir_tensor = py::module::import("xir.tensor").attr("Tensor");
  m.def("Runner",
        [](py::object subgraph,
           std::string mode) -> std::unique_ptr<vart::Runner> {
          auto real_subgraph = subgraph.attr("metadata");
          auto subg = py::cast<xir::Subgraph*>(real_subgraph);
          return vart::Runner::create_runner(subg, mode);
        },
        "create dpu runner");
  auto cls_dpu_runner =
      py::class_<vart::Runner>(m, "DpuRunner")
          .def("get_input_tensors",
               [xir_tensor](vart::Runner* self) -> std::vector<py::object> {
                 auto cxx_ret = self->get_input_tensors();
                 auto py_ret = std::vector<py::object>();
                 py_ret.reserve(cxx_ret.size());
                 for (auto t : cxx_ret) {
                   py::object py_tensor = xir_tensor();
                   py_tensor.attr("metadata") = py::cast(t);
                   py_ret.push_back(py_tensor);
                 }
                 return py_ret;
               })
          .def("get_output_tensors",
               [xir_tensor](vart::Runner* self) -> std::vector<py::object> {
                 // TODO: duplicated code, only difference is get_output vs
                 // get_input
                 auto cxx_ret = self->get_output_tensors();
                 auto py_ret = std::vector<py::object>();
                 py_ret.reserve(cxx_ret.size());
                 for (auto t : cxx_ret) {
                   py::object py_tensor = xir_tensor();
                   py_tensor.attr("metadata") = py::cast(t);
                   py_ret.push_back(py_tensor);
                 }
                 return py_ret;
               })
          .def("execute_async",
               [](vart::Runner* self, std::vector<py::array_t<float>> inputs,
                  std::vector<py::array_t<float>> outputs) {
                 auto cpu_inputs =
                     array_to_tensor_buffer(inputs, self->get_input_tensors());
                 auto cpu_outputs = array_to_tensor_buffer(
                     outputs, self->get_output_tensors());
                 auto ret = make_pair(uint32_t(0), int32_t(0));
                 if (1) {
                   py::gil_scoped_release release;
                   ret = self->execute_async(cpu_inputs, cpu_outputs);
                 }
                 destroy(cpu_inputs);
                 destroy(cpu_outputs);
                 return ret;
               })
          .def("wait",
               [](vart::Runner* self, std::pair<uint32_t, int32_t> job_id) {
                 return self->wait(job_id.first, -1);
               })
          .def("__repr__", [](const vart::Runner* self) {
            std::ostringstream str;
            str << "vart::Runner@" << (void*)self;
            return str.str();
          });
}
