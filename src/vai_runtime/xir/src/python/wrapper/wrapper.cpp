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
#if _WIN32
#else
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <UniLog/UniLog.hpp>
#include <algorithm>
#include <locale>
#include <string>

#include "xir/attrs/attr_def.hpp"
#include "xir/attrs/attrs.hpp"
#include "xir/graph/graph.hpp"
#include "xir/graph/subgraph.hpp"
#include "xir/op/op.hpp"
#include "xir/op/op_def.hpp"
#include "xir/op/op_def_factory_imp.hpp"
#include "xir/tensor/tensor.hpp"
#include "xir/util/data_type.hpp"
#include "xir/util/tool_function.hpp"

namespace py = pybind11;

namespace helper {

template <typename ElementType>
bool is_list_type(py::handle value) {
  if (!py::isinstance<py::list>(value)) return false;
  auto list = py::cast<py::list>(value);
  UNI_LOG_CHECK(list.size() > 0, PYXIR_INVALID_DATA_TYPE)
      << "Cannot set empty List";
  return std::all_of(list.begin(), list.end(), [](auto element) {
    return py::isinstance<ElementType>(element);
  });
}

template <typename ElementType>
bool is_dict_type(py::handle value) {
  if (!py::isinstance<py::dict>(value)) return false;
  auto dict = py::cast<py::dict>(value);
  UNI_LOG_CHECK(dict.size() > 0, PYXIR_INVALID_DATA_TYPE)
      << "Cannot set empty Dict";
  return std::all_of(dict.begin(), dict.end(), [](auto pair) {
    return py::isinstance<py::str>(pair.first) &&
           py::isinstance<ElementType>(pair.second);
  });
}

std::vector<char> get_vec_char(py::bytes bytes) {
  std::string str = py::cast<std::string>(bytes);
  std::vector<char> chars;
  for (auto i = 0u; i < str.size(); i++) chars.push_back(str[i]);
  return chars;
}

std::vector<char> get_vec_char(py::array array) {
  auto length = array.nbytes();
  auto chars = std::vector<char>(length);
  std::memcpy(chars.data(), array.data(), length);
  return chars;
}

template <typename Object>
struct AttrHelper {
  static void set(Object* obj, const std::string& key, py::handle value) {
    if (py::isinstance<py::bool_>(value)) {
      obj->set_attr(key, py::cast<bool>(value));
    } else if (py::isinstance<py::int_>(value)) {
      obj->set_attr(key, py::cast<std::int32_t>(value));
    } else if (py::isinstance<py::float_>(value)) {
      obj->set_attr(key, py::cast<double>(value));
    } else if (py::isinstance<py::str>(value)) {
      obj->set_attr(key, py::cast<std::string>(value));
    } else if (is_list_type<py::int_>(value)) {
      obj->set_attr(key, py::cast<std::vector<std::int32_t>>(value));
    } else if (is_list_type<py::bool_>(value)) {
      obj->set_attr(key, py::cast<std::vector<bool>>(value));
    } else if (is_list_type<py::float_>(value)) {
      obj->set_attr(key, py::cast<std::vector<double>>(value));
    } else if (is_list_type<py::str>(value)) {
      obj->set_attr(key, py::cast<std::vector<std::string>>(value));
    } else if (is_dict_type<py::int_>(value)) {
      obj->set_attr(key, py::cast<std::map<std::string, std::int32_t>>(value));
    } else if (is_dict_type<py::str>(value)) {
      obj->set_attr(key, py::cast<std::map<std::string, std::string>>(value));
    } else if (is_dict_type<py::float_>(value)) {
      obj->set_attr(key, py::cast<std::map<std::string, double>>(value));
    } else if (py::isinstance<py::bytes>(value)) {
      obj->set_attr(key, get_vec_char(py::cast<py::bytes>(value)));
    } else if (py::isinstance<py::array>(value)) {
      obj->set_attr(key, get_vec_char(py::cast<py::array>(value)));
    } else {
      UNI_LOG_FATAL(PYXIR_INVALID_DATA_TYPE) << "Unsupported data type!";
    }
  }

  static void set_hint(Object* obj, const std::string& key, py::handle value,
                       std::type_index hint) {
    if (hint == xir::TYPE_INDEX_BOOL) {
      obj->set_attr(key, py::cast<bool>(value));
    } else if (hint == xir::TYPE_INDEX_INT8) {
      obj->set_attr(key, py::cast<std::int8_t>(value));
    } else if (hint == xir::TYPE_INDEX_INT16) {
      obj->set_attr(key, py::cast<std::int16_t>(value));
    } else if (hint == xir::TYPE_INDEX_INT32) {
      obj->set_attr(key, py::cast<std::int32_t>(value));
    } else if (hint == xir::TYPE_INDEX_INT64) {
      obj->set_attr(key, py::cast<std::int64_t>(value));
    } else if (hint == xir::TYPE_INDEX_UINT8) {
      obj->set_attr(key, py::cast<std::uint8_t>(value));
    } else if (hint == xir::TYPE_INDEX_UINT16) {
      obj->set_attr(key, py::cast<std::uint16_t>(value));
    } else if (hint == xir::TYPE_INDEX_UINT32) {
      obj->set_attr(key, py::cast<std::uint32_t>(value));
    } else if (hint == xir::TYPE_INDEX_UINT64) {
      obj->set_attr(key, py::cast<std::uint64_t>(value));
    } else if (hint == xir::TYPE_INDEX_FLOAT) {
      obj->set_attr(key, py::cast<float>(value));
    } else if (hint == xir::TYPE_INDEX_DOUBLE) {
      obj->set_attr(key, py::cast<double>(value));
    } else if (hint == xir::TYPE_INDEX_STRING) {
      obj->set_attr(key, py::cast<std::string>(value));
    } else if (hint == xir::TYPE_INDEX_BOOL_VEC) {
      obj->set_attr(key, py::cast<std::vector<bool>>(value));
    } else if (hint == xir::TYPE_INDEX_INT8_VEC) {
      obj->set_attr(key, py::cast<std::vector<std::int8_t>>(value));
    } else if (hint == xir::TYPE_INDEX_INT16_VEC) {
      obj->set_attr(key, py::cast<std::vector<std::int16_t>>(value));
    } else if (hint == xir::TYPE_INDEX_INT32_VEC) {
      obj->set_attr(key, py::cast<std::vector<std::int32_t>>(value));
    } else if (hint == xir::TYPE_INDEX_INT64_VEC) {
      obj->set_attr(key, py::cast<std::vector<std::int64_t>>(value));
    } else if (hint == xir::TYPE_INDEX_UINT8_VEC) {
      obj->set_attr(key, py::cast<std::vector<std::uint8_t>>(value));
    } else if (hint == xir::TYPE_INDEX_UINT16_VEC) {
      obj->set_attr(key, py::cast<std::vector<std::uint16_t>>(value));
    } else if (hint == xir::TYPE_INDEX_UINT32_VEC) {
      obj->set_attr(key, py::cast<std::vector<std::uint32_t>>(value));
    } else if (hint == xir::TYPE_INDEX_UINT64_VEC) {
      obj->set_attr(key, py::cast<std::vector<std::uint64_t>>(value));
    } else if (hint == xir::TYPE_INDEX_FLOAT_VEC) {
      obj->set_attr(key, py::cast<std::vector<float>>(value));
    } else if (hint == xir::TYPE_INDEX_DOUBLE_VEC) {
      obj->set_attr(key, py::cast<std::vector<double>>(value));
    } else if (hint == xir::TYPE_INDEX_STRING_VEC) {
      obj->set_attr(key, py::cast<std::vector<std::string>>(value));
    } else if (hint == xir::TYPE_INDEX_MAP_STR_2_INT32) {
      obj->set_attr(key, py::cast<std::map<std::string, std::int32_t>>(value));
    } else if (hint == xir::TYPE_INDEX_MAP_STR_2_STR) {
      obj->set_attr(key, py::cast<std::map<std::string, std::string>>(value));
    } else if (hint == xir::TYPE_INDEX_BYTES) {
      if (py::isinstance<py::bytes>(value)) {
        obj->set_attr(key, get_vec_char(py::cast<py::bytes>(value)));
      } else if (py::isinstance<py::array>(value)) {
        obj->set_attr(key, get_vec_char(py::cast<py::array>(value)));
      }
    } else {
      AttrHelper::set(obj, key, value);
    }
  }

  static py::object get(const Object* obj, const std::string& key) {
    auto type = std::type_index(obj->get_attr(key).type());
    if (type == xir::TYPE_INDEX_BOOL) {
      return py::cast(obj->template get_attr<bool>(key));
    } else if (type == xir::TYPE_INDEX_INT8) {
      return py::cast(obj->template get_attr<std::int8_t>(key));
    } else if (type == xir::TYPE_INDEX_INT16) {
      return py::cast(obj->template get_attr<std::int16_t>(key));
    } else if (type == xir::TYPE_INDEX_INT32) {
      return py::cast(obj->template get_attr<std::int32_t>(key));
    } else if (type == xir::TYPE_INDEX_INT64) {
      return py::cast(obj->template get_attr<std::int64_t>(key));
    } else if (type == xir::TYPE_INDEX_UINT8) {
      return py::cast(obj->template get_attr<std::uint8_t>(key));
    } else if (type == xir::TYPE_INDEX_UINT16) {
      return py::cast(obj->template get_attr<std::uint16_t>(key));
    } else if (type == xir::TYPE_INDEX_UINT32) {
      return py::cast(obj->template get_attr<std::uint32_t>(key));
    } else if (type == xir::TYPE_INDEX_UINT64) {
      return py::cast(obj->template get_attr<std::uint64_t>(key));
    } else if (type == xir::TYPE_INDEX_FLOAT) {
      return py::cast(obj->template get_attr<float>(key));
    } else if (type == xir::TYPE_INDEX_DOUBLE) {
      return py::cast(obj->template get_attr<double>(key));
    } else if (type == xir::TYPE_INDEX_STRING) {
      return py::cast(obj->template get_attr<std::string>(key));
    } else if (type == xir::TYPE_INDEX_INT8_VEC) {
      return py::cast(obj->template get_attr<std::vector<std::int8_t>>(key));
    } else if (type == xir::TYPE_INDEX_INT16_VEC) {
      return py::cast(obj->template get_attr<std::vector<std::int16_t>>(key));
    } else if (type == xir::TYPE_INDEX_INT32_VEC) {
      return py::cast(obj->template get_attr<std::vector<std::int32_t>>(key));
    } else if (type == xir::TYPE_INDEX_INT64_VEC) {
      return py::cast(obj->template get_attr<std::vector<std::int64_t>>(key));
    } else if (type == xir::TYPE_INDEX_UINT8_VEC) {
      return py::cast(obj->template get_attr<std::vector<std::uint8_t>>(key));
    } else if (type == xir::TYPE_INDEX_UINT16_VEC) {
      return py::cast(obj->template get_attr<std::vector<std::uint16_t>>(key));
    } else if (type == xir::TYPE_INDEX_UINT32_VEC) {
      return py::cast(obj->template get_attr<std::vector<std::uint32_t>>(key));
    } else if (type == xir::TYPE_INDEX_UINT64_VEC) {
      return py::cast(obj->template get_attr<std::vector<std::uint64_t>>(key));
    } else if (type == xir::TYPE_INDEX_BOOL_VEC) {
      return py::cast(obj->template get_attr<std::vector<bool>>(key));
    } else if (type == xir::TYPE_INDEX_FLOAT_VEC) {
      return py::cast(obj->template get_attr<std::vector<float>>(key));
    } else if (type == xir::TYPE_INDEX_DOUBLE_VEC) {
      return py::cast(obj->template get_attr<std::vector<double>>(key));
    } else if (type == xir::TYPE_INDEX_STRING_VEC) {
      return py::cast(obj->template get_attr<std::vector<std::string>>(key));
    } else if (type == xir::TYPE_INDEX_MAP_STR_2_INT32) {
      return py::cast(
          obj->template get_attr<std::map<std::string, std::int32_t>>(key));
    } else if (type == xir::TYPE_INDEX_MAP_STR_2_STR) {
      return py::cast(
          obj->template get_attr<std::map<std::string, std::string>>(key));
    } else if (type == xir::TYPE_INDEX_BYTES) {
      auto data = obj->template get_attr<std::vector<char>>(key);
      return py::bytes(data.data(), data.size());
    }
    return py::none();
  }
};

template <typename Object>
void set_attr(Object* obj, const std::string& key, py::handle value) {
  AttrHelper<Object>::set(obj, key, value);
}

template <typename Object>
py::object get_attr(const Object* obj, const std::string& key) {
  return AttrHelper<Object>::get(obj, key);
}

template <>
void set_attr<xir::Op>(xir::Op* op, const std::string& key, py::handle value) {
  auto attrs = op->get_opdef()->attrs();
  auto iter = std::find_if(attrs.begin(), attrs.end(), [key](const auto& attr) {
    return attr.name == key;
  });
  if (iter == attrs.end()) {
    AttrHelper<xir::Op>::set(op, key, value);
  } else {
    AttrHelper<xir::Op>::set_hint(op, key, value, iter->data_type);
  }
}

template <typename Object>
void set_attrs(Object* obj, py::dict dict) {
  auto attrs = xir::Attrs::create();
  UNI_LOG_CHECK(py::isinstance<py::dict>(dict), PYXIR_INVALID_DATA_TYPE)
      << "attrs should be a dict";
  for (auto attr : py::cast<py::dict>(dict)) {
    set_attr<xir::Attrs>(attrs.get(), py::cast<std::string>(attr.first),
                         attr.second);
  }
  if (std::is_same<Object, xir::Op>::value) {
    auto defs = ((xir::Op*)(obj))->get_opdef()->attrs();
    for (auto key : attrs->get_keys()) {
      auto iter =
          std::find_if(defs.begin(), defs.end(),
                       [key](const auto& def) { return def.name == key; });
      if (iter != defs.end()) {
        AttrHelper<xir::Attrs>::set_hint(attrs.get(), key,
                                         get_attr<xir::Attrs>(attrs.get(), key),
                                         iter->data_type);
      }
    }
  }
  obj->set_attrs(std::move(attrs));
}

template <typename Object>
py::dict get_attrs(const Object* obj) {
  py::dict dict;
  auto attrs = obj->get_attrs();
  for (auto key : attrs->get_keys()) {
    dict[py::str(key)] = get_attr<xir::Attrs>(attrs.get(), key);
  }
  return dict;
}

// useless function, to be removed
xir::Op* add_const_op(xir::Graph* graph, const std::string& name,
                      const py::handle& data) {
  UNI_LOG_CHECK(py::isinstance<py::array>(data), PYXIR_INVALID_DATA_TYPE)
      << " attr.data should be in numpy format.";
  auto info = py::cast<py::array>(data).request();
  std::string type;
  int data_bytes = 1;
  if (info.format == "b") {
    type = "int8";
    data_bytes = 1;
  } else if (info.format == "B") {
    type = "uint8";
    data_bytes = 1;
  } else if (info.format == "h") {
    type = "int16";
    data_bytes = 2;
  } else if (info.format == "H") {
    type = "uint16";
    data_bytes = 2;
  } else if (info.format == "i") {
    type = "int32";
    data_bytes = 4;
  } else if (info.format == "I") {
    type = "uint32";
    data_bytes = 4;
  } else if (info.format == "l") {
    type = "int64";
    data_bytes = 8;
  } else if (info.format == "L") {
    type = "uint64";
    data_bytes = 8;
  } else if (info.format == "f") {
    type = "float32";
    data_bytes = 4;
  } else if (info.format == "d") {
    type = "float64";
    data_bytes = 8;
  }
  auto xir_data = get_vec_char(py::cast<py::array>(data));
  std::vector<std::int32_t> shape;
  for (auto s : info.shape) shape.push_back(static_cast<std::int32_t>(s));
  std::vector<std::int32_t> strides;
  for (auto s : info.strides) strides.push_back(static_cast<std::int32_t>(s));
  auto make_contiguous =
      [](const std::vector<int32_t>& shape, const uint32_t& num_elements,
         const std::vector<int32_t>& strides, const uint32_t& data_bytes,
         const std::vector<char>& data) -> std::vector<char> {
    std::vector<char> data_contiguous(num_elements * data_bytes);
    std::vector<int> dim_sizes(shape.size());
    dim_sizes.back() = data_bytes;
    for (int32_t i = int(shape.size()) - 2; i >= 0; i--) {
      dim_sizes[i] = dim_sizes[i + 1] * shape[i + 1];
    }
    if (dim_sizes == strides) {
      data_contiguous = data;
      return data_contiguous;
    }
    for (uint32_t i = 0; i < num_elements; i++) {
      int src_idx = 0;
      int dst_idx = 0;
      for (uint32_t j = 0; j < shape.size(); j++) {
        int dim_idx = (i / (dim_sizes[j] / data_bytes)) % shape[j];
        src_idx += dim_idx * strides[j];
        dst_idx += dim_idx * dim_sizes[j];
      }
      memcpy(data_contiguous.data() + dst_idx, data.data() + src_idx,
             data_bytes);
    }
    return data_contiguous;
  };
  xir_data = make_contiguous(shape, info.size, strides, data_bytes, xir_data);
  auto attrs = xir::Attrs::create();
  attrs->set_attr("shape", shape);
  attrs->set_attr("data_type", type);
  attrs->set_attr("data", xir_data);
  return graph->add_op(name, "const", std::move(attrs), {});
}

xir::Op* add_op(xir::Graph* graph, const std::string& name,
                const std::string& type, const py::handle& py_attrs,
                const py::handle& py_input_ops_map, xir::Subgraph* subgraph) {
  std::map<std::string, std::vector<xir::Op*>> input_ops_map;
  if (!py::isinstance<py::none>(py_input_ops_map)) {
    UNI_LOG_CHECK(py::isinstance<py::dict>(py_input_ops_map),
                  PYXIR_INVALID_DATA_TYPE)
        << "input_ops should be a dict";
    for (auto map : py::cast<py::dict>(py_input_ops_map)) {
      auto key = py::cast<std::string>(map.first);
      auto vec = py::cast<std::vector<xir::Op*>>(map.second);
      input_ops_map[key] = vec;
    }
  }
  auto attrs = xir::Attrs::create();
  if (!py::isinstance<py::none>(py_attrs)) {
    UNI_LOG_CHECK(py::isinstance<py::dict>(py_attrs), PYXIR_INVALID_DATA_TYPE)
        << "attrs should be a dict";
    auto build_in_ops = xir::op_def_factory()->get_registered_ops();
    if (std::find(build_in_ops.begin(), build_in_ops.end(), type) ==
        build_in_ops.end()) {
      xir::register_customized_operator_definition(name, type);
    }
    auto defs = xir::op_def_factory()->create(type)->attrs();
    for (auto attr : py::cast<py::dict>(py_attrs)) {
      auto key = py::cast<std::string>(attr.first);
      auto iter =
          std::find_if(defs.begin(), defs.end(),
                       [key](const auto& attr) { return attr.name == key; });
      if (iter == defs.end()) {
        AttrHelper<xir::Attrs>::set(attrs.get(), key, attr.second);
      } else {
        AttrHelper<xir::Attrs>::set_hint(attrs.get(), key, attr.second,
                                         iter->data_type);
      }
    }
  }
  auto op =
      graph->add_op(name, type, std::move(attrs), input_ops_map, subgraph);
  return op;
}

static std::string to_lower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str;
}

}  // namespace helper

namespace xir {

PYBIND11_MODULE(xir, m) {
  m.doc() = "pyxir module for python bindings";

  py::class_<DataType> data_type(m, "DataType");                    //
  data_type                                                         //
      .def(py::init<>())                                            //
      .def(py::init<const std::string&>())                          //
      .def(py::init<const DataType::Type&, const std::int32_t&>())  //
      .def_readwrite("type",                                        //
                     &DataType::type)                               //
      .def_readwrite("bit_width",                                   //
                     &DataType::bit_width)                          //
      .def("__repr__",                                              //
           [](const DataType& d) {                                  //
             return "<xir.DataType '" +                             //
                    helper::to_lower(d.to_string()) + "'>";         //
           })                                                       //
      .def("__str__",                                               //
           [](const DataType& d) {                                  //
             return "'" + helper::to_lower(d.to_string()) + "'";    //
           });
  py::enum_<DataType::Type>(data_type, "Type", py::arithmetic())  //
      .value("INT", DataType::Type::INT)                          //
      .value("UINT", DataType::Type::UINT)                        //
      .value("XINT", DataType::Type::XINT)                        //
      .value("XUINT", DataType::Type::XUINT)                      //
      .value("FLOAT", DataType::Type::FLOAT)                      //
      .value("UNKNOWN", DataType::Type::UNKNOWN)                  //
      .export_values();                                           //

  py::class_<Graph>(m, "Graph")                                            //
      .def(py::init<>(&Graph::create),                                     //
           py::arg("name"))                                                //
      .def_static("deserialize",                                           //
                  &Graph::deserialize,                                     //
                  py::arg("fname"),                                        //
                  py::return_value_policy::move)                           //
      .def("get_name",                                                     //
           &Graph::get_name)                                               //
      .def("create_op",                                                    //
           &helper::add_op,                                                //
           py::arg("name"),                                                //
           py::arg("kind"),                                                //
           py::arg("attrs") = py::none(),                                  //
           py::arg("input_ops") = py::none(),                              //
           py::arg("subgraph") = py::none(),                               //
           py::return_value_policy::reference_internal)                    //
      .def("create_const_op",                                              //
           &helper::add_const_op,                                          //
           py::return_value_policy::reference_internal)                    //
      .def("remove_op",                                                    //
           &Graph::remove_op,                                              //
           py::arg("op"))                                                  //
      .def("get_op",                                                       //
           py::overload_cast<const std::string&>(&Graph::get_op),          //
           py::arg("name"),                                                //
           py::return_value_policy::reference_internal)                    //
      .def("get_ops",                                                      //
           py::overload_cast<>(&Graph::get_ops),                           //
           py::return_value_policy::reference_internal)                    //
      .def("get_op_num",                                                   //
           &Graph::get_op_num)                                             //
      .def("get_tensors",                                                  //
           py::overload_cast<>(&Graph::get_tensors),                       //
           py::return_value_policy::reference_internal)                    //
      .def("get_head_ops",                                                 //
           py::overload_cast<>(&Graph::get_head_ops),                      //
           py::return_value_policy::reference_internal)                    //
      .def("get_tail_ops",                                                 //
           py::overload_cast<>(&Graph::get_tail_ops),                      //
           py::return_value_policy::reference_internal)                    //
      .def("get_tensor",                                                   //
           py::overload_cast<const std::string&>(&Graph::get_tensor),      //
           py::arg("name"),                                                //
           py::return_value_policy::reference_internal)                    //
      .def("get_tensor_producer",                                          //
           py::overload_cast<const Tensor*>(&Graph::get_tensor_producer),  //
           py::arg("tensor"),                                              //
           py::return_value_policy::reference_internal)                    //
      .def("toposort",                                                     //
           py::overload_cast<>(&Graph::topological_sort),                  //
           py::return_value_policy::reference_internal)                    //
      .def("infer_shape",                                                  //
           &Graph::infer_shape)                                            //
      .def("save_as_dot",                                                  //
           &Graph::save_to_dot,                                            //
           py::arg("fname"))                                               //
      .def("save_as_image",                                                //
           &Graph::visualize,                                              //
           py::arg("fname"),                                               //
           py::arg("format"))                                              //
      .def("serialize",                                                    //
           &Graph::serialize,                                              //
           py::arg("fname"))                                               //
      .def("get_root_subgraph",                                            //
           py::overload_cast<>(&Graph::get_root_subgraph),                 //
           py::return_value_policy::reference_internal)                    //
      .def("get_leaf_subgraph",                                            //
           py::overload_cast<const Op*>(&Graph::get_leaf_subgraph),        //
           py::return_value_policy::reference_internal)                    //
      .def("has_attr",                                                     //
           &Graph::has_attr,                                               //
           py::arg("name"))                                                //
      .def("set_attr",                                                     //
           &helper::set_attr<Graph>,                                       //
           py::arg("name"),                                                //
           py::arg("value"),                                               //
           py::return_value_policy::reference)                             //
      .def("set_attrs",                                                    //
           &helper::set_attrs<Graph>,                                      //
           py::arg("attrs"),                                               //
           py::return_value_policy::reference)                             //
      .def("get_attr",                                                     //
           &helper::get_attr<Graph>,                                       //
           py::arg("name"))                                                //
      .def("get_attrs",                                                    //
           &helper::get_attrs<Graph>,                                      //
           py::return_value_policy::move)                                  //
      .def("__repr__",                                                     //
           [](const Graph* g) {                                            //
             return "<xir.Graph named '" + g->get_name() + "'>";           //
           })                                                              //
      .def("__str__",                                                      //
           [](const Graph* g) {                                            //
             return "{name: '" + g->get_name() + "', op_num: " +           //
                    std::to_string(g->get_op_num()) + ", attrs: " +        //
                    py::cast<std::string>(                                 //
                        helper::get_attrs<Graph>(g).attr("__str__")()) +   //
                    "}";                                                   //
           });                                                             //

  py::class_<Op, std::unique_ptr<Op, py::nodelete>>(m, "Op")
      .def(py::init<>([](xir::Op* op) { return op; }))                      //
      .def("get_name",                                                      //
           &Op::get_name)                                                   //
      .def("get_type",                                                      //
           &Op::get_type)                                                   //
      .def("get_input_num",                                                 //
           py::overload_cast<>(&Op::get_input_num, py::const_))             //
      .def("get_input_num",                                                 //
           py::overload_cast<std::string>(&Op::get_input_num, py::const_),  //
           py::arg("name"))                                                 //
      .def("get_input_ops",                                                 //
           py::overload_cast<>(&Op::get_input_ops),                         //
           py::return_value_policy::reference_internal)                     //
      .def("get_input_ops_by_name",                                         //
           py::overload_cast<std::string>(&Op::get_input_ops),              //
           py::arg("name"),                                                 //
           py::return_value_policy::reference_internal)                     //
      .def("get_input_op",                                                  //
           py::overload_cast<std::string, int>(&Op::get_input_op),          //
           py::arg("name"),                                                 //
           py::arg("idx") = int(0),                                         //
           py::return_value_policy::reference_internal)                     //
      .def("set_input_ops",                                                 //
           &Op::set_input_ops,                                              //
           py::arg("name"),                                                 //
           py::arg("ops"))                                                  //
      .def("replace_input_ops",                                             //
           &Op::replace_input_op,                                           //
           py::arg("op_old"),                                               //
           py::arg("op_new"))                                               //
      .def("get_fanout_num",                                                //
           &Op::get_fanout_num)                                             //
      .def("get_fanout_ops",                                                //
           py::overload_cast<>(&Op::get_fanout_ops),                        //
           py::return_value_policy::reference_internal)                     //
      .def("get_input_tensors",                                             //
           py::overload_cast<>(&Op::get_input_tensors),                     //
           py::return_value_policy::reference_internal)                     //
      .def("get_input_tensors_by_name",                                     //
           py::overload_cast<std::string>(&Op::get_input_tensors),          //
           py::arg("name"),                                                 //
           py::return_value_policy::reference_internal)                     //
      .def("get_input_tensor",                                              //
           py::overload_cast<std::string, int>(&Op::get_input_tensor),      //
           py::arg("name"),                                                 //
           py::arg("idx") = int(0),                                         //
           py::return_value_policy::reference_internal)                     //
      .def("get_output_tensor",                                             //
           py::overload_cast<>(&Op::get_output_tensor),                     //
           py::return_value_policy::reference_internal)                     //
      .def("get_graph",                                                     //
           py::overload_cast<>(&Op::get_graph),                             //
           py::return_value_policy::reference)                              //
      .def("infer_shape",                                                   //
           &Op::shape_infer)                                                //
      .def("has_attr",                                                      //
           &Op::has_attr,                                                   //
           py::arg("name"))                                                 //
      .def("set_attr",                                                      //
           &helper::set_attr<Op>,                                           //
           py::arg("name"),                                                 //
           py::arg("value"),                                                //
           py::return_value_policy::reference)                              //
      .def("set_attrs",                                                     //
           &helper::set_attrs<Op>,                                          //
           py::arg("attrs"),                                                //
           py::return_value_policy::reference)                              //
      .def("get_attr",                                                      //
           &helper::get_attr<Op>,                                           //
           py::arg("name"))                                                 //
      .def("get_attrs",                                                     //
           &helper::get_attrs<Op>,                                          //
           py::return_value_policy::move)                                   //
      .def("__repr__",                                                      //
           [](const Op* op) {                                               //
             return "<xir.Op named '" + op->get_name() + "'>";              //
           })                                                               //
      .def("__str__",                                                       //
           [](const Op* op) {                                               //
             return "{name: '" + op->get_name() + "', type: '" +            //
                    op->get_type() + "', attrs: " +                         //
                    py::cast<std::string>(                                  //
                        helper::get_attrs<Op>(op).attr("__str__")()) +      //
                    "}";                                                    //
           });                                                              //

  py::class_<Subgraph, std::unique_ptr<Subgraph, py::nodelete>>(m, "Subgraph")
      .def(py::init<>([](xir::Subgraph* subgraph) { return subgraph; }))      //
      .def("get_name",                                                        //
           &Subgraph::get_name)                                               //
      .def("set_name",                                                        //
           &Subgraph::set_name,                                               //
           py::arg("name"),                                                   //
           py::return_value_policy::reference)                                //
      .def("get_op_num",                                                      //
           &Subgraph::get_op_num)                                             //
      .def("get_ops",                                                         //
           py::overload_cast<>(&Subgraph::get_ops),                           //
           py::return_value_policy::reference_internal)                       //
      .def("get_tensor_producer",                                             //
           py::overload_cast<const Tensor*>(&Subgraph::get_tensor_producer),  //
           py::arg("tensor"),                                                 //
           py::return_value_policy::reference_internal)                       //
      .def("get_input_tensors",                                               //
           py::overload_cast<>(&Subgraph::get_input_tensors),                 //
           py::return_value_policy::reference_internal)                       //
      .def("get_output_tensors",                                              //
           py::overload_cast<>(&Subgraph::get_output_tensors),                //
           py::return_value_policy::reference_internal)                       //
      .def("has_op",                                                          //
           py::overload_cast<const std::string&>(&Subgraph::has_op,           //
                                                 py::const_),                 //
           py::arg("name"))                                                   //
      .def("has_op",                                                          //
           py::overload_cast<const Op*>(&Subgraph::has_op, py::const_),       //
           py::arg("op"))                                                     //
      .def("find_child_subgraph_by_op",                                       //
           py::overload_cast<const Op*>(&Subgraph::find_op),                  //
           py::arg("op"),                                                     //
           py::return_value_policy::reference_internal)                       //
      .def("find_child_subgraph_by_op_name",                                  //
           py::overload_cast<const std::string&>(&Subgraph::find_op),         //
           py::arg("name"),                                                   //
           py::return_value_policy::reference_internal)                       //
      .def_property_readonly("is_root",                                       //
                             &Subgraph::is_root)                              //
      .def_property_readonly("is_leaf",                                       //
                             &Subgraph::is_leaf)                              //
      .def_property_readonly("root",                                          //
                             py::overload_cast<>(&Subgraph::get_root),        //
                             py::return_value_policy::reference)              //
      .def_property_readonly("depth",                                         //
                             &Subgraph::get_depth)                            //
      .def("get_parent",                                                      //
           py::overload_cast<>(&Subgraph::get_parent),                        //
           py::return_value_policy::reference)                                //
      .def("create_child_subgraph",                                           //
           &Subgraph::create_children)                                        //
      .def("get_children_num",                                                //
           &Subgraph::get_children_num)                                       //
      .def("get_children",                                                    //
           py::overload_cast<>(&Subgraph::get_children),                      //
           py::return_value_policy::reference_internal)                       //
      .def("is_child",                                                        //
           &Subgraph::is_child,                                               //
           py::arg("op"))                                                     //
      .def("merge_children",                                                  //
           &Subgraph::merge_children,                                         //
           py::arg("child_subgraph_list"),                                    //
           py::return_value_policy::reference_internal)                       //
      .def("get_graph",                                                       //
           py::overload_cast<>(&Subgraph::get_graph),                         //
           py::return_value_policy::reference)                                //
      .def("has_attr",                                                        //
           &Subgraph::has_attr,                                               //
           py::arg("name"))                                                   //
      .def("set_attr",                                                        //
           &helper::set_attr<Subgraph>,                                       //
           py::arg("name"),                                                   //
           py::arg("value"),                                                  //
           py::return_value_policy::reference)                                //
      .def("set_attrs",                                                       //
           &helper::set_attrs<Subgraph>,                                      //
           py::arg("attrs"),                                                  //
           py::return_value_policy::reference)                                //
      .def("get_attr",                                                        //
           &helper::get_attr<Subgraph>,                                       //
           py::arg("name"))                                                   //
      .def("get_attrs",                                                       //
           &helper::get_attrs<Subgraph>,                                      //
           py::return_value_policy::move)                                     //
      .def("toposort",                                                        //
           py::overload_cast<>(&Subgraph::topological_sort),                  //
           py::return_value_policy::reference_internal)                       //
      .def("toposort_child_subgraph",                                         //
           py::overload_cast<>(&Subgraph::children_topological_sort),         //
           py::return_value_policy::reference_internal)                       //
      .def("save_as_dot",                                                     //
           &Subgraph::save_to_dot,                                            //
           py::arg("fname"))                                                  //
      .def("__repr__",                                                        //
           [](const Subgraph* s) {                                            //
             return "<xir.Subgraph named '" + s->get_name() + "'>";           //
           })                                                                 //
      .def("__str__",                                                         //
           [](const Subgraph* s) {                                            //
             return "{name: '" + s->get_name() + "', op_num: " +              //
                    std::to_string(s->get_op_num()) + ", attrs: " +           //
                    py::cast<std::string>(                                    //
                        helper::get_attrs<Subgraph>(s).attr("__str__")()) +   //
                    "}";                                                      //
           });                                                                //

  py::class_<Tensor, std::unique_ptr<Tensor, py::nodelete>>(m, "Tensor")
      .def(py::init<>([](xir::Tensor* tensor) { return tensor; }))         //
      .def_static("clone",                                                 //
                  py::overload_cast<const Tensor*>(&Tensor::clone))        //
      .def_property("name",                                                //
                    &Tensor::get_name,                                     //
                    &Tensor::rename)                                       //
      .def_property_readonly("ndim",                                       //
                             &Tensor::get_dim_num)                         //
      .def_property_readonly("dims",                                       //
                             &Tensor::get_shape)                           //
      .def_property_readonly(                                              //
          "dtype",                                                         //
          [](const Tensor* tensor) {                                       //
            return helper::to_lower(tensor->get_data_type().to_string());  //
          })                                                               //
      .def_property_readonly("producer",                                   //
                             py::overload_cast<>(&Tensor::get_producer),   //
                             py::return_value_policy::reference_internal)  //
      .def("get_element_num",                                              //
           &Tensor::get_element_num)                                       //
      .def("get_data_size",                                                //
           &Tensor::get_data_size)                                         //
      .def("has_attr",                                                     //
           &Tensor::has_attr,                                              //
           py::arg("name"))                                                //
      .def("set_attr",                                                     //
           &helper::set_attr<Tensor>,                                      //
           py::arg("name"),                                                //
           py::arg("value"),                                               //
           py::return_value_policy::reference)                             //
      .def("set_attrs",                                                    //
           &helper::set_attrs<Tensor>,                                     //
           py::arg("attrs"),                                               //
           py::return_value_policy::reference)                             //
      .def("get_attr",                                                     //
           &helper::get_attr<Tensor>,                                      //
           py::arg("name"))                                                //
      .def("get_attrs",                                                    //
           &helper::get_attrs<Tensor>,                                     //
           py::return_value_policy::move)                                  //
      .def("__repr__",                                                     //
           [](const Tensor* t) {                                           //
             return "<xir.Tensor named '" + t->get_name() + "'>";          //
           })                                                              //
      .def("__str__",                                                      //
           [](const Tensor* t) {                                           //
             return "{name: '" + t->get_name() + "', shape: " +            //
                    py::cast<std::string>(                                 //
                        py::cast(t->get_shape()).attr("__str__")()) +      //
                    ", type: '" +                                          //
                    helper::to_lower(t->get_data_type().to_string()) +     //
                    "', attrs: " +                                         //
                    py::cast<std::string>(                                 //
                        helper::get_attrs<Tensor>(t).attr("__str__")()) +  //
                    "}";                                                   //
           });                                                             //

  py::class_<OpTemplate, std::unique_ptr<OpTemplate, py::nodelete>>(
      m, "OpTemplate")
      .def(py::init<>(
          [](xir::OpTemplate* OpTemplate) { return OpTemplate; }))  //
      .def("get_name",                                              //
           &OpTemplate::get_name)                                   //
      .def("get_types",                                             //
           &OpTemplate::get_types)                                  //
      .def("get_input_ops",                                         //
           (&OpTemplate::get_input_ops),                            //
           py::return_value_policy::reference_internal)             //
      .def("get_fanout_ops",                                        //
           (&OpTemplate::get_fanout_ops),                           //
           py::return_value_policy::reference_internal);            //

  py::class_<GraphTemplate, std::unique_ptr<GraphTemplate, py::nodelete>>(
      m, "GraphTemplate")
      .def(py::init<>(
          [](xir::GraphTemplate* GraphTemplate) { return GraphTemplate; }))  //
      .def("get_name",                                                       //
           &GraphTemplate::get_name)                                         //
      .def("get_op",                                                         //
           (&GraphTemplate::get_op),                                         //
           py::arg("name"),                                                  //
           py::return_value_policy::reference_internal)                      //
      .def("get_op_num",                                                     //
           &GraphTemplate::get_op_num)                                       //
      .def("toposort",                                                       //
           (&GraphTemplate::topological_sort),                               //
           py::return_value_policy::reference_internal);                     //

}  // PYBIND11_MODULE

}  // namespace xir
