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
 * */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xir/op/op.hpp"
#include "xir/op/op_def.hpp"
#include "vitis/ai/extra_ops.hpp"
#include "xir/graph/graph.hpp"
#include "xir/util/tool_function.hpp"
namespace py = pybind11;

static void glog(const std::string& v) { LOG(INFO) << v; }

static std::vector<char> get_vec_char(py::buffer buffer) {
  auto info = buffer.request();
  auto ptr = info.ptr;
  auto size = info.size;
  auto ret = std::vector<char>((size_t)size);
  // TODO: handle stride
  std::memcpy(&ret[0], ptr, size);
  return ret;
}

static std::vector<std::vector<char>> get_list_of_vec_char(py::list lst) {
  auto ret = std::vector<std::vector<char>>(lst.size());
  auto i = 0u;
  for (auto& elt : lst) {
    ret[i] = get_vec_char(py::cast<py::buffer>(elt));
    i = i + 1;
  }
  return ret;
}

template <typename ElementType>
bool is_list_type(py::list list) {
  UNI_LOG_CHECK(list.size() > 0, PYXIR_INVALID_DATA_TYPE)
      << "Cannot set empty List";
  return std::all_of(list.begin(), list.end(), [](auto element) {
    return py::isinstance<ElementType>(element);
  });
}

static void set_op_py_buffer_type_attr(xir::Op* op, const std::string& key, py::list value) {
  if (is_list_type<py::buffer>(value)) {
    auto attrs = op->get_opdef()->attrs();
    auto iter = std::find_if(attrs.begin(), attrs.end(), [key](const auto& attr) {
      return attr.name == key;
    });
    if (iter == attrs.end()) {
      op->set_attr(key, get_list_of_vec_char(value));
    } else {
      if (iter->data_type == xir::TYPE_INDEX_BYTES_VEC) {
        op->set_attr(key, get_list_of_vec_char(value));
      }
      else {
        LOG(FATAL) << "attr " << key << "'s data_type mismatch, please double check, "
	        << "or create a new attr key name";
      }
    }
  }
  else {
    LOG(FATAL) << "attr " << key << "'s data_type don't support now. "
	       << "key: " << key << ", data_type is " << (is_list_type<py::bytes>(value))
               << " " << (is_list_type<py::str>(value)) << " "
               << (is_list_type<py::buffer>(value));
  }
}


PYBIND11_MODULE(xir_extra_ops, m) {
  m.doc() = "extra ops bindings";
  py::module::import("xir");
  vitis::ai::maybe_add_op_defs();
  m.def("remove_xfix", &xir::remove_xfix);
  m.def("log", &glog);
  m.def("set_postprocessor",
        [](xir::Graph* graph, const std::string& postprocessor,
           const std::map<std::string, std::vector<std::string>>& value) {
          graph->set_attr("xmodel_postprocessor", postprocessor);
          graph->set_attr("xmodel_outputs", value);
        });
  m.def("set_op_py_buffer_type_attr", &set_op_py_buffer_type_attr);
}
