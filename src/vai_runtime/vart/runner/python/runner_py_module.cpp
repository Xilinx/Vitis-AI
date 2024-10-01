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
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <iostream>
#include <thread>
using namespace std;

#include <sstream>
#ifndef MODULE_NAME
#define MODULE_NAME vart
#endif
#include <unordered_map>
#include <xir/graph/subgraph.hpp>
#include <xir/tensor/tensor.hpp>

#include "../src/runner_helper.hpp"
#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
#include "vart/tensor_buffer.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/weak.hpp"

DEF_ENV_PARAM(DEBUG_RUNNER, "0");

namespace py = pybind11;
namespace {
using tensor_buffers_t = std::vector<vart::TensorBuffer*>;
using map_from_job_id_to_tensor_buffers_t =
    std::unordered_map<int, tensor_buffers_t>;
using the_map_t =
    std::unordered_map<vart::Runner*, map_from_job_id_to_tensor_buffers_t>;

static std::shared_ptr<the_map_t> get_store() {
  return vitis::ai::WeakSingleton<the_map_t>::create();
}

class CpuFlatTensorBuffer : public vart::TensorBuffer {
 public:
  explicit CpuFlatTensorBuffer(py::buffer_info&& info,
                               std::unique_ptr<xir::Tensor>&& tensor);

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
  void save_to_map(vart::Runner* runner, int job_id);

 private:
  py::buffer_info info_;
  void* data_;
  std::unique_ptr<xir::Tensor> my_tensor_;
  std::shared_ptr<the_map_t> the_shared_map_;
  vart::Runner* runner_;
  int job_id_;
};

CpuFlatTensorBuffer::CpuFlatTensorBuffer(py::buffer_info&& info,
                                         std::unique_ptr<xir::Tensor>&& tensor)
    : TensorBuffer{tensor.get()},
      info_{std::move(info)},
      data_{info.ptr},
      my_tensor_{std::move(tensor)},
      the_shared_map_{nullptr},
      runner_{nullptr},
      job_id_{0} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
      << "create CpuFlatTensorBuffer @" << (void*)this << " data= " << data_
      << " DEUBG " << (int)((char*)data_)[0];
}

CpuFlatTensorBuffer::~CpuFlatTensorBuffer() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
      << "destroy CpuFlatTensorBuffer @" << (void*)this << " data= " << data_;
  if (the_shared_map_) {
    CHECK(runner_ != nullptr);
    auto& the_map = *the_shared_map_.get();
    auto& v = the_map[runner_][job_id_];
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
    if (v.empty()) {
      the_map[runner_].erase(job_id_);
      if (the_map[runner_].empty()) {
        the_map.erase(runner_);
      }
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
        << "size of map:" << the_map.size() << " "             //
        << "use_count:" << the_shared_map_.use_count() << " "  //
        << endl;
  }
}

void CpuFlatTensorBuffer::save_to_map(vart::Runner* runner, int job_id) {
  the_shared_map_ = get_store();
  CHECK(runner != nullptr);
  CHECK_GE(job_id, 0);
  runner_ = runner;
  job_id_ = job_id;
  (*the_shared_map_)[runner][job_id].push_back(this);
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
  } else if (dtype.type == xir::DataType::XINT && dtype.bit_width == 16) {
    ret = py::format_descriptor<int16_t>::format();
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
  } else if (format == py::format_descriptor<int16_t>::format()) {
    ret.type = xir::DataType::XINT;
  }
  CHECK(ret.type != xir::DataType::UNKNOWN) << "unsupported data type";
  return ret;
}

static vart::TensorBuffer* array_to_tensor_buffer(py::buffer& a,
                                                  const xir::Tensor* tensor) {
  auto info = a.request(true);
  LOG_IF(INFO, false) << "info = " << info.format;
  auto dtype = from_py_buf_format(info.format, info.itemsize);
  // here we have to clone a tensor buffer, because the input tensor
  // buffer might be in different data type.
  auto new_tensor =
      xir::Tensor::create(tensor->get_name(), tensor->get_shape(), dtype);
  // do not copy attrs, we should check vart::TensorBuffer::copy, if
  // "fix_point" is get from the attr or not
  //
  // new_tensor->set_attrs(tensor->get_attrs());
  return new CpuFlatTensorBuffer(std::move(info), std::move(new_tensor));
}

// Convert py::buffer to TensorBuffer with real shape of py::buffer
// instead of tensor shape from Runner as in `array_to_tensor_buffer`
static vart::TensorBuffer* dynamic_array_to_tensor_buffer(
    py::buffer& a, const xir::Tensor* tensor) {
  auto info = a.request(true);
  LOG_IF(INFO, false) << "info = " << info.format;
  auto dtype = from_py_buf_format(info.format, info.itemsize);
  std::vector<int> shape;
  shape.reserve(info.shape.size());
  for (auto i : info.shape) shape.push_back(i);
  auto new_tensor = xir::Tensor::create(tensor->get_name(), shape, dtype);
  return new CpuFlatTensorBuffer(std::move(info), std::move(new_tensor));
}

static vector<vart::TensorBuffer*> array_to_tensor_buffer(
    const std::vector<py::buffer>& a,
    const std::vector<const xir::Tensor*> tensors, bool enable_dynamic_array) {
  auto ret = vector<vart::TensorBuffer*>{};
  ret.reserve(a.size());
  auto c = 0u;
  if (enable_dynamic_array) {
    for (auto x : a) {
      ret.emplace_back(dynamic_array_to_tensor_buffer(x, tensors[c++]));
    }
  } else {
    for (auto x : a) {
      ret.emplace_back(array_to_tensor_buffer(x, tensors[c++]));
    }
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
      .def(
          "execute_async",
          [](vart::Runner* self, std::vector<py::buffer> inputs,
             std::vector<py::buffer> outputs, bool enable_dynamic_array) {
            // NOTE: it is important to initialize cpu_inputs and
            // cpu_outputs with GIL protection. the_map is the global
            // variable alike.
            auto cpu_inputs = array_to_tensor_buffer(
                inputs, self->get_input_tensors(), enable_dynamic_array);
            auto cpu_outputs = array_to_tensor_buffer(
                outputs, self->get_output_tensors(), enable_dynamic_array);
            auto ret = make_pair(uint32_t(0), int32_t(0));
            if (1) {
              py::gil_scoped_release release;
              ret = self->execute_async(cpu_inputs, cpu_outputs);
            }
            // obtain the GIL again.
            if (ret.first >= 0) {
              for (auto t : cpu_inputs) {
                static_cast<CpuFlatTensorBuffer*>(t)->save_to_map(self,
                                                                  ret.first);
              }
              for (auto t : cpu_outputs) {
                static_cast<CpuFlatTensorBuffer*>(t)->save_to_map(self,
                                                                  ret.first);
              }
            } else {
              destroy(cpu_inputs);
              destroy(cpu_outputs);
            }
            return ret;
          },
          py::arg("inputs"), py::arg("outputs"),
          py::arg("enable_dynamic_array") = false)
      .def("wait",
           [](vart::Runner* self, std::pair<uint32_t, int> job_id) {
             auto ret = self->wait(job_id.first, -1);
             auto the_map = get_store();
             // copy instead of reference, it is important, do not use
             // reference here, the decontructor will clean up the mess.
             auto v = (*the_map)[self][(int)job_id.first];
             for (auto t : v) {
               delete t;
             }
             return ret;
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
            runner.release();
            if (runner_ext == nullptr) {
              return nullptr;
            }
            return std::unique_ptr<vart::RunnerExt>(runner_ext);
          })
      .def("get_inputs",
           [](vart::RunnerExt* self) {
             return vart::alloc_cpu_flat_tensor_buffers(
                 self->get_input_tensors());
           })
      .def("get_outputs", [](vart::RunnerExt* self) {
        return vart::alloc_cpu_flat_tensor_buffers(self->get_output_tensors());
      });
}
}  // namespace
