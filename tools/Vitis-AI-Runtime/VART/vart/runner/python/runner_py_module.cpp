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
#define MODULE_NAME vart
#endif
#include <xir/graph/subgraph.hpp>
#include <xir/tensor/tensor.hpp>

#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
#include "vart/tensor_buffer.hpp"
namespace py = pybind11;
namespace {
class CpuFlatTensorBuffer : public vart::TensorBuffer {
 public:
  explicit CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor);
  virtual ~CpuFlatTensorBuffer();

 public:
  virtual std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override {
    uint32_t size = std::ceil(tensor_->get_data_type().bit_width / 8.f);
    if (idx.size() == 0) {
      return {reinterpret_cast<uint64_t>(data_),
              tensor_->get_element_num() * size};
    }
    auto dims = tensor_->get_shape();
    auto offset = 0;
    for (auto k = 0U; k < tensor_->get_shape().size(); k++) {
      auto stride = 1;
      for (auto m = k + 1; m < tensor_->get_shape().size(); m++) {
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
  std::unique_ptr<xir::Tensor> my_tensor_;
};

CpuFlatTensorBuffer::CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor)
    : TensorBuffer{xir::Tensor::clone(tensor).release()},
      data_{data},
      my_tensor_{const_cast<xir::Tensor*>(get_tensor())} {
  LOG_IF(INFO, false) << "create CpuFlatTensorBuffer @" << (void*)this
                      << " data= " << data_ << " DEUBG "
                      << (int)((char*)data_)[0];
}

CpuFlatTensorBuffer::~CpuFlatTensorBuffer() {
  LOG_IF(INFO, false) << "destroy CpuFlatTensorBuffer @" << (void*)this
                      << " data= " << data_;
}

static std::vector<int> calculate_strides(const std::vector<int>& shape,
                                          size_t size_of_elt) {
  auto ret = std::vector<int>(shape.size(), 1);
  for (int i = ((int)shape.size()) - 2; i >= 0; --i) {
    ret[i] = ret[i + 1] * shape[i + 1];
  }
  for (auto& x : ret) {
    x = x * size_of_elt;
  }
  return ret;
}

static std::string to_py_buf_format(const xir::DataType& dtype) {
  auto ret = std::string("");
  if (dtype.type == xir::DataType::XINT && dtype.bit_width == 8) {
    ret = py::format_descriptor<int8_t>::format();
  } else if (dtype.type == xir::DataType::FLOAT && dtype.bit_width == 32) {
    ret = py::format_descriptor<float>::format();
  }
  CHECK(!ret.empty()) << "unsupported data type";
  return ret;
}

static xir::DataType from_py_buf_format(const std::string& format,
                                        size_t itemsize) {
  auto ret = xir::DataType();
  ret.type = xir::DataType::UNKNOWN;
  ret.bit_width = itemsize * 8;
  if (format == py::format_descriptor<int8_t>::format()) {
    ret.type = xir::DataType::XINT;
  } else if (format == py::format_descriptor<float>::format()) {
    ret.type = xir::DataType::FLOAT;
  }
  CHECK(ret.type != xir::DataType::UNKNOWN) << "unsupported data type";
  return ret;
}

static vart::TensorBuffer* array_to_tensor_buffer(py::buffer& a,
                                                  const xir::Tensor* tensor) {
  auto info = a.request(true);
  // TODO TENSOR
  void* data = info.ptr;
  LOG_IF(INFO, false) << "info = " << info.format;
  /*  auto dims = std::vector<std::int32_t>();

    dims.reserve(a.ndim());
    for (auto i = 0; i < a.ndim(); ++i) {
      dims.push_back(a.shape(i));
    }
    void* p = a.mutable_data();
    auto name = tensor->get_name();
    return new CpuFlatTensorBuffer(
        p,
        xir::Tensor::create(name, dims, xir::DataType::FLOAT, sizeof(float) *
    8u) .release());*/
  auto dtype = from_py_buf_format(info.format, info.itemsize);
  auto new_tensor =
      xir::Tensor::create(tensor->get_name(), tensor->get_shape(), dtype);
  new_tensor->set_attrs(tensor->get_attrs());
  return new CpuFlatTensorBuffer(data, new_tensor.get());
}

static vector<vart::TensorBuffer*> array_to_tensor_buffer(
    const std::vector<py::buffer>& a,
    const std::vector<const xir::Tensor*> tensors) {
  auto ret = vector<vart::TensorBuffer*>{};
  ret.reserve(a.size());
  auto c = 0u;
  for (auto x : a) {
    ret.emplace_back(array_to_tensor_buffer(x, tensors[c++]));
  }
  return ret;
}

static void destroy(vart::TensorBuffer* tb) { delete tb; }

static void destroy(const std::vector<vart::TensorBuffer*>& tb) {
  for (auto x : tb) {
    destroy(x);
  }
}

PYBIND11_MODULE(MODULE_NAME, m) {
  m.doc() = "vart::Runner inferace";  // optional module docstring
  py::module::import("xir");
  py::class_<vart::TensorBuffer>(m, "TensorBuffer", py::buffer_protocol())
      .def_buffer([](vart::TensorBuffer& tb) -> py::buffer_info {
        uint64_t data = 0u;
        size_t size = 0u;
        auto tensor = tb.get_tensor();
        auto shape = tensor->get_shape();
        auto idx = std::vector<int>(shape.size(), 0);
        std::tie(data, size) = tb.data(idx);
        auto dtype = tensor->get_data_type();
        auto format = to_py_buf_format(dtype);
        CHECK_EQ(size, tensor->get_data_size())
            << "only support continuous tensor buffer yet";
        return py::buffer_info((void*)data,         /* Pointer to buffer */
                               dtype.bit_width / 8, /* Size of one scalar */
                               format,              /* Python struct-style
                                                       format descriptor */
                               shape.size(),        /* Number of dimensions */
                               shape,               /* Buffer dimensions */
                               calculate_strides(shape, dtype.bit_width / 8));
      })
      // TODO: export other method
      .def("get_tensor", &vart::TensorBuffer::get_tensor,
           py::return_value_policy::reference)
      .def("__str__",
           [](vart::TensorBuffer* self) { return self->to_string(); })
      .def("__repr__",
           [](vart::TensorBuffer* self) { return self->to_string(); });

  py::class_<vart::Runner>(m, "Runner")
      .def_static("create_runner",
                  py::overload_cast<const xir::Subgraph*, const std::string&>(
                      &vart::Runner::create_runner),
                  py::arg("subgraph"), py::arg("mode") = "")
      .def("get_input_tensors", &vart::Runner::get_input_tensors,
           py::return_value_policy::reference)
      .def("get_output_tensors", &vart::Runner::get_output_tensors,
           py::return_value_policy::reference)
      .def("execute_async",
           [](vart::Runner* self, std::vector<py::buffer> inputs,
              std::vector<py::buffer> outputs) {
             auto cpu_inputs =
                 array_to_tensor_buffer(inputs, self->get_input_tensors());
             auto cpu_outputs =
                 array_to_tensor_buffer(outputs, self->get_output_tensors());
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
           [](vart::Runner* self, std::pair<uint32_t, int> job_id) {
             return self->wait(job_id.first, -1);
           })
      .def("__repr__", [](const vart::Runner* self) {
        std::ostringstream str;
        str << "vart::Runner@" << (void*)self;
        return str.str();
      });
  py::class_<vart::RunnerExt, vart::Runner>(m, "RunnerExt")
      .def_static(
          "create_runner",
          [](const xir::Subgraph* s,
             const std::string& mode) -> std::unique_ptr<vart::RunnerExt> {
            auto runner = vart::Runner::create_runner(s, mode);
            auto runner_ext = dynamic_cast<vart::RunnerExt*>(runner.get());
            if (runner_ext == nullptr) {
              return nullptr;
            }
            runner.release();
            return std::unique_ptr<vart::RunnerExt>(runner_ext);
          })
      .def("get_inputs", &vart::RunnerExt::get_inputs,
           // TODO: on edge, tensor buffers are not in a continuous
           // region, copy
           py::return_value_policy::reference)
      .def("get_outputs", &vart::RunnerExt::get_outputs,
           py::return_value_policy::reference);
}
}  // namespace
