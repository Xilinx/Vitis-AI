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

#include "xir/graph/serialize_v2.hpp"
// must include this first to define XIR_DLLSPEC;
#include "xir/XirExport.hpp"

#include <fstream>
#include <type_traits>
#include <typeindex>
#include <unordered_map>

#include "config.hpp"
#include "graph_proto_v2.pb.h"

#include "xir/attrs/attr_def.hpp"
#include "xir/attrs/attrs.hpp"
#include "xir/graph/graph.hpp"
#include "xir/op/op.hpp"
#include "xir/op/op_def.hpp"
#include "xir/op/op_def_factory_imp.hpp"
#include "xir/tensor/tensor.hpp"
#if __cplusplus > 201700
#define HAS_VALUE(x) (x.has_value())  // # for std::any
#else
#define HAS_VALUE(x) (!x.empty())  // # for std::experimental::any
#endif

namespace xir {
namespace v2 {

using std::ifstream;
using std::map;
using std::ofstream;
using std::string;
using std::unique_ptr;
using std::vector;
using bytes_t = vector<char>;
template <typename t>
using map_t = std::map<std::string, t>;
using pb_map_attr_value_t =
    ::google::protobuf::Map<::std::string, ::serial_v2::AttrValue>;

template <typename PBType>
struct pb_type_to_cpp_type_t;
template <typename CppType>
struct cpp_type_to_pb_type_t;

template <typename From, typename To>
struct convert_t {
  static To fun(const From& x) { return x; }
};

template <typename CppType>
static typename cpp_type_to_pb_type_t<CppType>::type
convert_from_cpp_type_to_pb(const CppType& x) {
  using PBType = typename cpp_type_to_pb_type_t<CppType>::type;
  return convert_t<CppType, PBType>::fun(x);
}

template <typename PBType>
static typename pb_type_to_cpp_type_t<PBType>::type
convert_from_pb_type_tp_cpp_type(const PBType& x) {
  using CppType = typename pb_type_to_cpp_type_t<PBType>::type;
  return convert_t<PBType, CppType>::fun(x);
}

#define DECLARE_IDENTICAL_PB_CPP_MAPPING(TYPE)                                 \
  template <>                                                                  \
  struct pb_type_to_cpp_type_t<TYPE> {                                         \
    using type = TYPE;                                                         \
  };                                                                           \
  template <>                                                                  \
  struct cpp_type_to_pb_type_t<TYPE> {                                         \
    using type = TYPE;                                                         \
  };
#define DECLARE_PB_CPP_MAPPING(PB, CPP)                                        \
  template <>                                                                  \
  struct pb_type_to_cpp_type_t<PB> {                                           \
    using type = CPP;                                                          \
  };                                                                           \
  template <>                                                                  \
  struct cpp_type_to_pb_type_t<CPP> {                                          \
    using type = PB;                                                           \
  };                                                                           \
  template <> /* not conversion is needed if same */                           \
  struct convert_t<CPP, PB> {                                                  \
    static PB fun(const CPP& x);                                               \
  };                                                                           \
  template <>                                                                  \
  struct convert_t<PB, CPP> {                                                  \
    static CPP fun(const PB& x);                                               \
  };

#define DECLARE_PB_CPP_MAPPING_VEC(pb_type, cpp_type)                          \
  DECLARE_PB_CPP_MAPPING(pb_type, std::vector<cpp_type>)                       \
  pb_type convert_t<std::vector<cpp_type>, pb_type>::fun(                      \
      const vector<cpp_type>& x) {                                             \
    {                                                                          \
      auto ret = pb_type{};                                                    \
      auto value = ret.mutable_value();                                        \
      value->Reserve(x.size());                                                \
      for (const auto& v : x) {                                                \
        *(value->Add()) = convert_from_cpp_type_to_pb(v);                      \
      }                                                                        \
      return ret;                                                              \
    }                                                                          \
  }                                                                            \
  vector<cpp_type> convert_t<pb_type, std::vector<cpp_type>>::fun(             \
      const pb_type& x) {                                                      \
    auto value = std::vector<cpp_type>{};                                      \
    value.reserve(x.value().size());                                           \
    for (const auto& v : x.value()) {                                          \
      value.emplace_back(convert_from_pb_type_tp_cpp_type(v));                 \
    }                                                                          \
    return value;                                                              \
  }

DECLARE_IDENTICAL_PB_CPP_MAPPING(bool);
DECLARE_IDENTICAL_PB_CPP_MAPPING(int32_t);
DECLARE_IDENTICAL_PB_CPP_MAPPING(uint32_t);
DECLARE_IDENTICAL_PB_CPP_MAPPING(int64_t);
DECLARE_IDENTICAL_PB_CPP_MAPPING(uint64_t);
DECLARE_IDENTICAL_PB_CPP_MAPPING(float);
DECLARE_IDENTICAL_PB_CPP_MAPPING(double);
DECLARE_IDENTICAL_PB_CPP_MAPPING(string);
DECLARE_PB_CPP_MAPPING(serial_v2::Bytes, bytes_t);

DECLARE_PB_CPP_MAPPING_VEC(serial_v2::BoolVec, bool);
DECLARE_PB_CPP_MAPPING_VEC(serial_v2::Int32Vec, int32_t);
DECLARE_PB_CPP_MAPPING_VEC(serial_v2::Uint32Vec, uint32_t);
DECLARE_PB_CPP_MAPPING_VEC(serial_v2::Int64Vec, int64_t);
DECLARE_PB_CPP_MAPPING_VEC(serial_v2::Uint64Vec, uint64_t);
DECLARE_PB_CPP_MAPPING_VEC(serial_v2::FloatVec, float);
DECLARE_PB_CPP_MAPPING_VEC(serial_v2::DoubleVec, double);
DECLARE_PB_CPP_MAPPING_VEC(serial_v2::BytesVec, bytes_t);
DECLARE_PB_CPP_MAPPING_VEC(serial_v2::StringVec, string);

#define DECLARE_PB_CPP_MAPPING_MAP(pb_type, cpp_type)                          \
  DECLARE_PB_CPP_MAPPING(pb_type, map_t<cpp_type>)                             \
  pb_type convert_t<map_t<cpp_type>, pb_type>::fun(const map_t<cpp_type>& x) { \
    auto ret = pb_type{};                                                      \
    auto value = ret.mutable_value();                                          \
    for (const auto& v : x) {                                                  \
      (*value)[v.first] = convert_from_cpp_type_to_pb(v.second);               \
    }                                                                          \
    return ret;                                                                \
  }                                                                            \
  map_t<cpp_type> convert_t<pb_type, map_t<cpp_type>>::fun(const pb_type& x) { \
    auto value = map_t<cpp_type>{};                                            \
    for (const auto& v : x.value()) {                                          \
      value[v.first] = convert_from_pb_type_tp_cpp_type(v.second);             \
    }                                                                          \
    return value;                                                              \
  }

// other types
struct OpArg {
  string arg_name;
  std::vector<xir::Op*> arg_ops;
};
using xir_op_ptr_t = xir::Op*;
using xir_subgraph_ptr_t = xir::Subgraph*;
DECLARE_PB_CPP_MAPPING(serial_v2::Graph, unique_ptr<xir::Graph>);
DECLARE_PB_CPP_MAPPING(serial_v2::OPNode, xir_op_ptr_t);
DECLARE_PB_CPP_MAPPING(serial_v2::OpArg, OpArg);
DECLARE_PB_CPP_MAPPING(serial_v2::Tensor, unique_ptr<xir::Tensor>);
DECLARE_PB_CPP_MAPPING(serial_v2::SubGraph, xir_subgraph_ptr_t);
DECLARE_PB_CPP_MAPPING(serial_v2::OpDef, xir::OpDef);
DECLARE_PB_CPP_MAPPING(serial_v2::OpArgDef, xir::OpArgDef);
DECLARE_PB_CPP_MAPPING(serial_v2::AttrDef, xir::AttrDef);
DECLARE_PB_CPP_MAPPING(serial_v2::AttrValue, xir::any);
DECLARE_PB_CPP_MAPPING(pb_map_attr_value_t, unique_ptr<xir::Attrs>);
DECLARE_PB_CPP_MAPPING_MAP(serial_v2::MapString2Int32, int32_t);
DECLARE_PB_CPP_MAPPING_MAP(serial_v2::MapString2Uint32, uint32_t);
DECLARE_PB_CPP_MAPPING_MAP(serial_v2::MapString2Int64, int64_t);
DECLARE_PB_CPP_MAPPING_MAP(serial_v2::MapString2Uint64, uint64_t);
DECLARE_PB_CPP_MAPPING_MAP(serial_v2::MapString2String, string);
DECLARE_PB_CPP_MAPPING_MAP(serial_v2::MapString2Bytes, bytes_t);
DECLARE_PB_CPP_MAPPING_MAP(serial_v2::MapString2Int32Vec, std::vector<int32_t>);
DECLARE_PB_CPP_MAPPING_MAP(serial_v2::MapString2Uint32Vec,
                           std::vector<uint32_t>);
DECLARE_PB_CPP_MAPPING_MAP(serial_v2::MapString2Int64Vec, std::vector<int64_t>);
DECLARE_PB_CPP_MAPPING_MAP(serial_v2::MapString2Uint64Vec,
                           std::vector<uint64_t>);
DECLARE_PB_CPP_MAPPING_MAP(serial_v2::MapString2StringVec, std::vector<string>);
// partial specialization for vector
template <typename PBType>
struct pb_type_to_cpp_type_t<::google::protobuf::RepeatedPtrField<PBType>> {
  using type = vector<typename pb_type_to_cpp_type_t<PBType>::type>;
};
template <typename CppType>
struct cpp_type_to_pb_type_t<vector<CppType>> {
  using type = ::google::protobuf::RepeatedPtrField<
      typename cpp_type_to_pb_type_t<CppType>::type>;
};
template <typename PB, typename CPP>
struct convert_t<vector<CPP>, ::google::protobuf::RepeatedPtrField<PB>> {
  static ::google::protobuf::RepeatedPtrField<PB> fun(const vector<CPP>& x) {
    auto ret = ::google::protobuf::RepeatedPtrField<PB>{};
    ret.Reserve(x.size());
    for (const auto& v : x) {
      auto p = ret.Add();
      (*p) = convert_from_cpp_type_to_pb(v);
    }
    return ret;
  }
};
template <typename PB, typename CPP>
struct convert_t<::google::protobuf::RepeatedPtrField<PB>, vector<CPP>> {
  static vector<CPP> fun(const ::google::protobuf::RepeatedPtrField<PB>& x) {
    auto ret = vector<CPP>{};
    ret.reserve(x.size());
    for (const auto& v : x) {
      ret.emplace_back(convert_from_pb_type_tp_cpp_type(v));
    }
    return ret;
  }
};

// local helper functions
static vector<OpDef> get_op_defs();
static void collect_ops(const serial_v2::SubGraph& subgraph,
                        std::set<Op*>* ret);
static void create_subgraphs(const serial_v2::SubGraph& subg,
                             xir_subgraph_ptr_t parent);
// conversion between graph
// use a thread local variable to share the graph among converters.
thread_local xir::Graph* the_graph = nullptr;
static xir::Graph* get_graph() { return the_graph; }

serial_v2::Graph convert_t<unique_ptr<xir::Graph>, serial_v2::Graph>::fun(
    const unique_ptr<xir::Graph>& graph) {
  auto pb_graph = serial_v2::Graph();
  pb_graph.set_graph_name(graph->get_name());
  *(pb_graph.mutable_op_node()) =
      convert_from_cpp_type_to_pb(graph->topological_sort());
  *(pb_graph.mutable_subg_root()) =
      convert_from_cpp_type_to_pb(graph->get_root_subgraph());
  *(pb_graph.mutable_op_defs()) = convert_from_cpp_type_to_pb(get_op_defs());
  *(pb_graph.mutable_graph_attr()) =
      convert_from_cpp_type_to_pb(graph->get_attrs());
  return pb_graph;
}

unique_ptr<xir::Graph> convert_t<serial_v2::Graph, unique_ptr<xir::Graph>>::fun(
    const serial_v2::Graph& pb_graph) {
  auto graph = xir::Graph::create(pb_graph.graph_name());
  // the graph is under construction, share it among converters.
  the_graph = graph.get();
  // this function has side effect, which imports all op defs;
  convert_from_pb_type_tp_cpp_type(pb_graph.op_defs());
  // this function has side effect, which add op into the graph.
  convert_from_pb_type_tp_cpp_type(pb_graph.op_node());
  // this function has side effect, which create all subgraphs.
  if (pb_graph.has_subg_root()) {
    auto& subg_root = pb_graph.subg_root();
    auto root = graph->get_root_subgraph();
    root->set_name(subg_root.subgraph_name());
    root->set_attrs(convert_from_pb_type_tp_cpp_type(subg_root.subg_attr()));
    for (const auto& child : subg_root.subg_child()) {
      create_subgraphs(child, root);
    }
  }
  graph->set_attrs(convert_from_pb_type_tp_cpp_type(pb_graph.graph_attr()));
  return graph;
}
// conversion btw xir::Op and serial_v2::Op
serial_v2::OPNode convert_t<xir_op_ptr_t, serial_v2::OPNode>::fun(
    const xir_op_ptr_t& op) {
  auto get_op_args = [](const xir_op_ptr_t& op) {
    auto ret = vector<OpArg>{};
    for (const auto& op_def_input :
         op_def_factory()->create(op->get_type())->input_args()) {
      ret.emplace_back(
          OpArg{op_def_input.name, op->get_input_ops(op_def_input.name)});
    }
    return ret;
  };
  auto pb_node = serial_v2::OPNode();
  pb_node.set_op_name(op->get_name());
  pb_node.set_op_type(op->get_type());
  *(pb_node.mutable_op_attr()) = convert_from_cpp_type_to_pb(op->get_attrs());
  *(pb_node.mutable_args()) = convert_from_cpp_type_to_pb(get_op_args(op));
  // dirty hack, borrow instead owne
  auto tensor_ref = std::unique_ptr<xir::Tensor>(
      const_cast<xir::Tensor*>(op->get_output_tensor()));
  *(pb_node.mutable_output_tensor()) = convert_from_cpp_type_to_pb(tensor_ref);
  tensor_ref.release();
  return pb_node;
}

xir_op_ptr_t convert_t<serial_v2::OPNode, xir_op_ptr_t>::fun(
    const serial_v2::OPNode& pb_node) {
  auto convert_to_iom = [](const vector<OpArg>& args) {
    auto ret = std::map<std::string, std::vector<Op*>>{};
    for (auto& arg : args) {
      ret[arg.arg_name] = arg.arg_ops;
    }
    return ret;
  };
  auto op_attrs = convert_from_pb_type_tp_cpp_type(pb_node.op_attr());
  auto tensor = convert_from_pb_type_tp_cpp_type(pb_node.output_tensor());
  auto ret = get_graph()->add_op(
      pb_node.op_name(), pb_node.op_type(),
      std::move(op_attrs),  //
      convert_to_iom(convert_from_pb_type_tp_cpp_type(pb_node.args())));
  ret->replace_output_tensor(

      std::move(tensor));
  return ret;
}

// conversion between OpArg
serial_v2::OpArg convert_t<OpArg, serial_v2::OpArg>::fun(
    const OpArg& xir_op_arg) {
  auto ret = serial_v2::OpArg();
  ret.set_arg_name(xir_op_arg.arg_name);
  ret.mutable_arg_ops()->Reserve(xir_op_arg.arg_ops.size());
  for (const auto& op : xir_op_arg.arg_ops) {
    *(ret.mutable_arg_ops()->Add()) = op->get_name();
  }
  return ret;
}

OpArg convert_t<serial_v2::OpArg, OpArg>::fun(
    const serial_v2::OpArg& pb_op_arg) {
  auto ret = OpArg();
  ret.arg_name = pb_op_arg.arg_name();
  ret.arg_ops.reserve(pb_op_arg.arg_ops_size());
  for (const auto& op : pb_op_arg.arg_ops()) {
    ret.arg_ops.emplace_back(get_graph()->get_op(op));
  }
  return ret;
}
// conversion of Tensor
serial_v2::Tensor
convert_t<std::unique_ptr<xir::Tensor>, serial_v2::Tensor>::fun(
    const std::unique_ptr<xir::Tensor>& tensor) {
  CHECK(tensor != nullptr);
  serial_v2::Tensor pb_tensor;
  // save tensor name
  pb_tensor.set_tensor_name(tensor->get_name());

  // save tensor dim
  std::int32_t dim_num = tensor->get_shape().size();
  for (auto i = 0; i < dim_num; i++) {
    pb_tensor.add_tensor_dim(tensor->get_shape().at(i));
  }

  auto data_type = tensor->get_data_type();
  int data_type_idx = static_cast<int>(data_type.type);
  pb_tensor.set_data_type(data_type_idx);

  int bit_width = data_type.bit_width;
  pb_tensor.set_tensor_bit_width(bit_width);

  // save tensor attr
  auto attrs = tensor->get_attrs();
  if (attrs) {
    *(pb_tensor.mutable_tensor_attr()) = convert_from_cpp_type_to_pb(attrs);
  }
  return pb_tensor;
}
// conversion of subgraph
// only implement from xir_subgraph_ptr_t to serial_v2::Subgraph
// reverse convesion is done by create_subgraphs(...)
serial_v2::SubGraph convert_t<xir_subgraph_ptr_t, serial_v2::SubGraph>::fun(
    const xir_subgraph_ptr_t& subg) {
  auto ret = serial_v2::SubGraph();
  ret.set_subgraph_name(subg->get_name());
  *(ret.mutable_subg_attr()) = convert_from_cpp_type_to_pb(subg->get_attrs());
  if (subg->is_leaf()) {
    auto ops = ret.mutable_op_name();
    auto children = subg->topological_sort();
    for (const auto& child : children) {
      *(ops->Add()) = child->get_name();
    }
  } else {
    *(ret.mutable_subg_child()) =
        convert_from_cpp_type_to_pb(subg->children_topological_sort());
  }
  return ret;
}
// conversion between tensors
std::unique_ptr<Tensor>
convert_t<serial_v2::Tensor, std::unique_ptr<Tensor>>::fun(
    const serial_v2::Tensor& pb_tensor) {
  // read tensor name and tensor dim
  auto tensor_name = pb_tensor.tensor_name();
  auto tensor_bit_width = pb_tensor.tensor_bit_width();
  vector<int> tensor_dim;
  for (auto i = 0; i < pb_tensor.tensor_dim_size(); i++)
    tensor_dim.push_back(pb_tensor.tensor_dim(i));
  auto tensor = Tensor::create(
      tensor_name, tensor_dim,
      DataType{static_cast<DataType::Type>(pb_tensor.data_type()),
               tensor_bit_width});

  // read tensor attrs
  tensor->set_attrs(convert_from_pb_type_tp_cpp_type(pb_tensor.tensor_attr()));

  return tensor;
}
// conversion of OpDef
serial_v2::OpDef convert_t<xir::OpDef, serial_v2::OpDef>::fun(
    const xir::OpDef& op_def) {
  auto pb_op_def = ::serial_v2::OpDef();
  pb_op_def.set_name(op_def.name());
  *(pb_op_def.mutable_input_args()) =
      convert_from_cpp_type_to_pb(op_def.input_args());
  *(pb_op_def.mutable_attrs()) = convert_from_cpp_type_to_pb(op_def.attrs());
  pb_op_def.set_annotation(op_def.annotation());
  return pb_op_def;
}
xir::OpDef convert_t<serial_v2::OpDef, xir::OpDef>::fun(
    const serial_v2::OpDef& op_def) {
  return xir::OpDef{/// const std::string name;
                    op_def.name(),
                    /// const std::vector<OpArgDef> input_args;
                    convert_from_pb_type_tp_cpp_type(op_def.input_args()),
                    /// const std::vector<AttrDef> attrs;
                    convert_from_pb_type_tp_cpp_type(op_def.attrs()),
                    /// std::function<void(Op * op)> shape_infer;
                    nullptr,
                    /// Some comments
                    op_def.annotation()

  };
}
// conversion of OpArgDef
serial_v2::OpArgDef convert_t<xir::OpArgDef, serial_v2::OpArgDef>::fun(
    const xir::OpArgDef& input_arg) {
  auto pb_input_arg = serial_v2::OpArgDef();
  pb_input_arg.set_name(input_arg.name);
  pb_input_arg.set_occur_type(
      (serial_v2::OpArgDef::OccurType)(int)input_arg.occur_type);
  pb_input_arg.set_data_type((int)input_arg.data_type);
  pb_input_arg.set_annotation(input_arg.annotation);
  return pb_input_arg;
}
xir::OpArgDef convert_t<serial_v2::OpArgDef, xir::OpArgDef>::fun(
    const serial_v2::OpArgDef& arg) {
  return xir::OpArgDef{/// const std::string name;
                       arg.name(),
                       /// const OccurenceType occur_type;
                       (xir::OpArgDef::OccurenceType)(int)arg.occur_type(),
                       /// const DataType Classification for Tensor, data_type;
                       (xir::DataType::Type)(int)arg.data_type(),
                       ///  const std::string annotation;
                       arg.annotation()};
}
// conversion of AttrDef
serial_v2::AttrDef convert_t<xir::AttrDef, serial_v2::AttrDef>::fun(
    const xir::AttrDef& attr_def) {
  serial_v2::AttrDef pb_attr_def;
  pb_attr_def.set_name(attr_def.name);
  pb_attr_def.set_occur_type(
      (serial_v2::AttrDef::OccurType)(int)attr_def.occur_type);
  *(pb_attr_def.mutable_default_value()) =
      convert_from_cpp_type_to_pb(attr_def.default_value);
  pb_attr_def.set_list_length(attr_def.list_length);
  pb_attr_def.set_annotation(attr_def.annotation);
  return pb_attr_def;
}
xir::AttrDef convert_t<serial_v2::AttrDef, xir::AttrDef>::fun(
    const serial_v2::AttrDef& attr_def) {
  auto default_value =
      convert_from_pb_type_tp_cpp_type(attr_def.default_value());
  auto ret =
      xir::AttrDef{///  const std::string name
                   attr_def.name(),
                   ///  const std::type_index data_type;  not used
                   TYPE_INDEX_BOOL,
                   /// const OccurenceType occur_type;
                   (xir::AttrDef::OccurenceType)(int)attr_def.occur_type(),
                   ///  const std::uint32_t list_length;
                   (uint32_t)attr_def.list_length(),
                   /// const std::string annotation;
                   attr_def.annotation(),
                   /// const xir::any default_value;
                   default_value};

  UNI_LOG_CHECK(HAS_VALUE(ret.default_value), XIR_INTERNAL_ERROR)
      << "load attr value error: pb= " << attr_def.DebugString()
      << "HAS_VALUE(default_value) " << HAS_VALUE(default_value) << " "  //
      << "type = " << default_value.type().name()
      << "annotation = " << attr_def.annotation() << " "
      << "attr_def.annotation() " << attr_def.annotation() << " "  //
      << std::endl;
  return ret;
}

// conversion between bytes_t and serial_v2::Bytes
serial_v2::Bytes convert_t<bytes_t, serial_v2::Bytes>::fun(const bytes_t& x) {
  serial_v2::Bytes ret;
  if (x.empty()) {
    *(ret.mutable_value()) = std::string();
  } else {
    *(ret.mutable_value()) = std::string(&x[0], x.size());
  }
  return ret;
}

bytes_t convert_t<serial_v2::Bytes, bytes_t>::fun(const serial_v2::Bytes& x) {
  auto& value = x.value();
  return bytes_t(value.begin(), value.end());
};

#define FROM_ANY_TO_PB_SCALAR(type, method)                                    \
  {                                                                            \
    std::type_index(typeid(std::remove_cv_t<std::remove_reference_t<type>>)),  \
        [](const xir::any& value) {                                            \
          serial_v2::AttrValue ret;                                            \
          ret.method(                                                          \
              convert_from_cpp_type_to_pb(xir::stdx::any_cast<type>(value)));  \
          return ret;                                                          \
        },                                                                     \
  }
#define FROM_ANY_TO_PB_MSG(type, method)                                       \
  {                                                                            \
    std::type_index(typeid(std::remove_cv_t<std::remove_reference_t<type>>)),  \
        [](const xir::any& value) {                                            \
          serial_v2::AttrValue ret;                                            \
          *(ret.method()) =                                                    \
              convert_from_cpp_type_to_pb(xir::stdx::any_cast<type>(value));   \
          return ret;                                                          \
        },                                                                     \
  }

using from_any_to_pb_t =
    std::add_pointer<serial_v2::AttrValue(const xir::any&)>::type;

static std::unordered_map<std::type_index, from_any_to_pb_t>
    dispatcher_from_any_to_pb = {
        // scalars
        FROM_ANY_TO_PB_SCALAR(bool, set_bool_value),
        FROM_ANY_TO_PB_SCALAR(int32_t, set_int32_value),
        FROM_ANY_TO_PB_SCALAR(uint32_t, set_uint32_value),
        FROM_ANY_TO_PB_SCALAR(int64_t, set_int64_value),
        FROM_ANY_TO_PB_SCALAR(uint64_t, set_uint64_value),
        FROM_ANY_TO_PB_SCALAR(float, set_float_value),
        FROM_ANY_TO_PB_SCALAR(double, set_double_value),
        FROM_ANY_TO_PB_SCALAR(const string&, set_string_value),
        FROM_ANY_TO_PB_MSG(const bytes_t&, mutable_bytes_value),
        // vectors
        FROM_ANY_TO_PB_MSG(const vector<bool>&, mutable_bool_vec_value),
        FROM_ANY_TO_PB_MSG(const vector<int32_t>&, mutable_int32_vec_value),
        FROM_ANY_TO_PB_MSG(const vector<uint32_t>&, mutable_uint32_vec_value),
        FROM_ANY_TO_PB_MSG(const vector<int64_t>&, mutable_int64_vec_value),
        FROM_ANY_TO_PB_MSG(const vector<uint64_t>&, mutable_uint64_vec_value),
        FROM_ANY_TO_PB_MSG(const vector<float>&, mutable_float_vec_value),
        FROM_ANY_TO_PB_MSG(const vector<double>&, mutable_double_vec_value),
        FROM_ANY_TO_PB_MSG(const vector<string>&, mutable_string_vec_value),
        FROM_ANY_TO_PB_MSG(const vector<bytes_t>&, mutable_bytes_vec_value),
        FROM_ANY_TO_PB_MSG(const map_t<int32_t>&,
                           mutable_map_string_2_int32_value),
        FROM_ANY_TO_PB_MSG(const map_t<uint32_t>&,
                           mutable_map_string_2_uint32_value),
        FROM_ANY_TO_PB_MSG(const map_t<int64_t>&,
                           mutable_map_string_2_int64_value),
        FROM_ANY_TO_PB_MSG(const map_t<uint64_t>&,
                           mutable_map_string_2_uint64_value),
        FROM_ANY_TO_PB_MSG(const map_t<string>&,
                           mutable_map_string_2_string_value),
        FROM_ANY_TO_PB_MSG(const map_t<bytes_t>&,
                           mutable_map_string_2_bytes_value),
        FROM_ANY_TO_PB_MSG(const map_t<vector<int32_t>>&,
                           mutable_map_string_2_int32_vec_value),
        FROM_ANY_TO_PB_MSG(const map_t<vector<uint32_t>>&,
                           mutable_map_string_2_uint32_vec_value),
        FROM_ANY_TO_PB_MSG(const map_t<vector<int64_t>>&,
                           mutable_map_string_2_int64_vec_value),
        FROM_ANY_TO_PB_MSG(const map_t<vector<uint64_t>>&,
                           mutable_map_string_2_uint64_vec_value),
        FROM_ANY_TO_PB_MSG(const map_t<vector<string>>&,
                           mutable_map_string_2_string_vec_value),
};

#define FROM_PB_TO_ANY(case_value, method)                                     \
  {                                                                            \
    serial_v2::AttrValue::case_value,                                          \
        [](const serial_v2::AttrValue& pb_attr) {                              \
          return xir::any(convert_from_pb_type_tp_cpp_type(pb_attr.method())); \
        },                                                                     \
  }

using from_pb_to_any_t =
    std::add_pointer<xir::any(const serial_v2::AttrValue&)>::type;

#ifdef CMAKE_CXX_COMPILER_VERSION_LESS_THAN_6_0
static std::unordered_map<serial_v2::AttrValue::ValueCase, from_pb_to_any_t,
                          std::hash<int>>
#else
static std::unordered_map<serial_v2::AttrValue::ValueCase, from_pb_to_any_t>
#endif
    dispatcher_from_pb_to_any = {
        // scalars
        FROM_PB_TO_ANY(kBoolValue, bool_value),
        FROM_PB_TO_ANY(kInt32Value, int32_value),
        FROM_PB_TO_ANY(kUint32Value, uint32_value),
        FROM_PB_TO_ANY(kInt64Value, int64_value),
        FROM_PB_TO_ANY(kUint64Value, uint64_value),
        FROM_PB_TO_ANY(kFloatValue, float_value),
        FROM_PB_TO_ANY(kDoubleValue, double_value),
        FROM_PB_TO_ANY(kStringValue, string_value),
        FROM_PB_TO_ANY(kBytesValue, bytes_value),
        // vector
        FROM_PB_TO_ANY(kBoolVecValue, bool_vec_value),
        FROM_PB_TO_ANY(kInt32VecValue, int32_vec_value),
        FROM_PB_TO_ANY(kUint32VecValue, uint32_vec_value),
        FROM_PB_TO_ANY(kInt64VecValue, int64_vec_value),
        FROM_PB_TO_ANY(kUint64VecValue, uint64_vec_value),
        FROM_PB_TO_ANY(kFloatVecValue, float_vec_value),
        FROM_PB_TO_ANY(kDoubleVecValue, double_vec_value),
        FROM_PB_TO_ANY(kStringVecValue, string_vec_value),
        FROM_PB_TO_ANY(kBytesVecValue, bytes_vec_value),
        // map
        FROM_PB_TO_ANY(kMapString2Int32Value, map_string_2_int32_value),
        FROM_PB_TO_ANY(kMapString2Uint32Value, map_string_2_uint32_value),
        FROM_PB_TO_ANY(kMapString2Int64Value, map_string_2_int64_value),
        FROM_PB_TO_ANY(kMapString2Uint64Value, map_string_2_uint64_value),
        FROM_PB_TO_ANY(kMapString2StringValue, map_string_2_string_value),
        FROM_PB_TO_ANY(kMapString2BytesValue, map_string_2_bytes_value),
        // map vec
        FROM_PB_TO_ANY(kMapString2Int32VecValue, map_string_2_int32_vec_value),
        FROM_PB_TO_ANY(kMapString2Uint32VecValue,
                       map_string_2_uint32_vec_value),
        FROM_PB_TO_ANY(kMapString2Int64VecValue, map_string_2_int64_vec_value),
        FROM_PB_TO_ANY(kMapString2Uint64VecValue,
                       map_string_2_uint64_vec_value),
        FROM_PB_TO_ANY(kMapString2StringVecValue,
                       map_string_2_string_vec_value),
};

serial_v2::AttrValue convert_t<xir::any, serial_v2::AttrValue>::fun(
    const xir::any& x) {
  auto ret = serial_v2::AttrValue();
  auto type_index = std::type_index(x.type());
  auto it = dispatcher_from_any_to_pb.find(type_index);
  if (it == dispatcher_from_any_to_pb.end()) {
    UNI_LOG_ERROR(XIR_UNSUPPORTED_TYPE) << "not supported type for attr."
                                        << "type_id: " << x.type().name();
  } else {
    ret = it->second(x);
  }
  return ret;
}
xir::any convert_t<serial_v2::AttrValue, xir::any>::fun(
    const serial_v2::AttrValue& x) {
  auto ret = xir::any();
  auto it = dispatcher_from_pb_to_any.find(x.value_case());
  if (it == dispatcher_from_pb_to_any.end()) {
    UNI_LOG_ERROR(XIR_UNSUPPORTED_TYPE) << " AttrValue=" << x.DebugString();
  } else {
    ret = it->second(x);
    UNI_LOG_CHECK(HAS_VALUE(ret), XIR_INTERNAL_ERROR)
        << " no value!" << x.DebugString();
  }
  return ret;
}

pb_map_attr_value_t convert_t<unique_ptr<xir::Attrs>, pb_map_attr_value_t>::fun(
    const unique_ptr<xir::Attrs>& x) {
  auto ret = pb_map_attr_value_t();
  auto keys = x->get_keys();
  for (auto key : keys) {
    if (key.size() >= 2 && key[0] == '_' && key[1] == '_') {
      UNI_LOG_DEBUG_INFO << "ignore private attribute: " << key;
      continue;
    }
    ret[key] = convert_from_cpp_type_to_pb(x->get_attr(key));
  }
  return ret;
}

unique_ptr<xir::Attrs>
convert_t<pb_map_attr_value_t, unique_ptr<xir::Attrs>>::fun(
    const pb_map_attr_value_t& x) {
  auto ret = xir::Attrs::create();
  for (const auto& value : x) {
    ret->set_attr(value.first, convert_from_pb_type_tp_cpp_type(value.second));
  }
  return ret;
}

unique_ptr<Graph> Serialize::read(const string& pb_fname) {
  std::ifstream ifs(pb_fname, std::ios::binary);
  serial_v2::Graph pb_graph;

  if (!pb_graph.ParseFromIstream(&ifs)) {
    UNI_LOG_FATAL(XIR_READ_PB_FAILURE) << "file = " << pb_fname;
  }
  return convert_from_pb_type_tp_cpp_type(pb_graph);
}

unique_ptr<Graph> Serialize::read_from_string(const std::string& str) {
  serial_v2::Graph pb_graph;
  if (!pb_graph.ParseFromString(str)) {
    UNI_LOG_FATAL(XIR_READ_PB_FAILURE)
        << "fail to generate pb struct from string.";
  }
  return convert_from_pb_type_tp_cpp_type(
      *(static_cast<serial_v2::Graph*>(&pb_graph)));
}

void Serialize::write(const Graph* graph, const string& pb_fname) {
  std::ofstream ofs(pb_fname, std::ios::binary | std::ios::trunc);
  // dirty hack, g only borrows graph, not owned.
  auto g = std::unique_ptr<Graph>(const_cast<Graph*>(graph));
  serial_v2::Graph pb_graph = convert_from_cpp_type_to_pb(g);
  g.release();
  if (!pb_graph.SerializeToOstream(&ofs)) {
    UNI_LOG_FATAL(XIR_WRITE_PB_FAILURE) << "file = " << pb_fname;
  }
}

void Serialize::write_to_string(const Graph* graph, std::string* str) {
  auto g = std::unique_ptr<Graph>(const_cast<Graph*>(graph));
  serial_v2::Graph pb_graph = convert_from_cpp_type_to_pb(g);
  g.release();
  if (!pb_graph.SerializeToString(str)) {
    UNI_LOG_FATAL(XIR_WRITE_PB_FAILURE) << "fail to parse pb struct to string.";
  }
}

// local helper functions implementations
static vector<OpDef> get_op_defs() {
  auto ret = vector<OpDef>{};
  auto factory = op_def_factory();
  auto ops = factory->get_registered_ops();
  for (const auto& op_name : ops) {
    ret.emplace_back(*factory->create(op_name));
  }
  return ret;
}
static void collect_ops(const serial_v2::SubGraph& subgraph,
                        std::set<Op*>* ret) {
  for (auto& child : subgraph.subg_child()) {
    collect_ops(child, ret);
  }
  for (auto& op : subgraph.op_name()) {
    ret->insert(get_graph()->get_op(op));
  }
}

static void create_subgraphs(const serial_v2::SubGraph& subg,
                             xir_subgraph_ptr_t parent) {
  auto is_leaf = true;
  auto has_op = !subg.op_name().empty();
  auto has_subg = !subg.subg_child().empty();
  if (has_subg && !has_op) {
    is_leaf = false;
  } else if (!has_subg && has_op) {
    is_leaf = true;
  } else {
    UNI_LOG_CHECK(false, XIR_INTERNAL_ERROR)
        << " a subg cannot have both op and subg. or neither of them:"
        << "has_subg " << has_subg << " "  //
        << "has_op " << has_op << " "      //
        ;
  }
  auto ops = std::set<Op*>();
  collect_ops(subg, &ops);
  auto ops_to_subgs = [parent](const std::set<Op*>& ops) {
    auto subgs = std::set<xir_subgraph_ptr_t>{};
    for (auto op : ops) {
      auto s = parent->find_op(op);
      UNI_LOG_CHECK(s != nullptr, XIR_INTERNAL_ERROR)
          << "cannot find subgraph!"
          << "op->name " << op->get_name() << " "                //
          << "parent->get_name() " << parent->get_name() << " "  //
          << "parent->get_children_num() " << parent->get_children_num()
          << " "  //
          ;
      subgs.insert(s);
    }
    return subgs;
  };
  if (parent->is_leaf()) {
    parent->create_children();
  }
  auto subgs = ops_to_subgs(ops);
  if (!is_leaf) {
    auto new_branch = parent->merge_children(subgs);
    new_branch->set_name(subg.subgraph_name());
    new_branch->set_attrs(convert_from_pb_type_tp_cpp_type(subg.subg_attr()));
    for (const auto& child : subg.subg_child()) {
      create_subgraphs(child, new_branch);
    }
  } else {
    auto new_leaf = parent->merge_children(subgs);
    new_leaf->set_name(subg.subgraph_name());
    new_leaf->set_attrs(convert_from_pb_type_tp_cpp_type(subg.subg_attr()));
  }
}
}  // namespace v2
}  // namespace xir
