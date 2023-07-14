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
#include "./util.hpp"
#include "xir/cxir.h"
#include "xir/graph/graph_imp.hpp"

namespace xir {
class c_api {
 public:
  static const char* get_name(xir_graph_t graph) {
    auto self = static_cast<xir::GraphImp*>(graph);
    return boost::get_property(*self->graph_, boost::graph_name).c_str();
  }

  static void xir_graph_get_ops(xir_graph_t graph, xir_op_t ret[]) {
    auto self = static_cast<xir::GraphImp*>(graph);
    auto i = 0;
    for (auto vd : boost::make_iterator_range(boost::vertices(*self->graph_))) {
      auto r = (op_up_cast((*self->graph_)[vd].get()));
      ret[i] = r;
      i = i + 1;
    }
  }

  static void xir_graph_get_tensors(xir_graph_t graph, xir_tensor_t ret[]) {
    auto self = static_cast<xir::GraphImp*>(graph);
    auto i = 0;
    for (auto vd : boost::make_iterator_range(boost::vertices(*self->graph_))) {
      ret[i] = (op_up_cast((*self->graph_)[vd].get()))->get_output_tensor();
      i = i + 1;
    }
  }
  static xir_attrs_t graph_get_attrs(xir_graph_t graph) {
    auto self = static_cast<xir::GraphImp*>(graph);
    return static_cast<xir_graph_t>(self->attrs_.get());
  }
};
}  // namespace xir
extern "C" const char* xir_graph_get_name(xir_graph_t graph) {
  return xir::c_api::get_name(graph);
}

extern "C" xir_graph_t xir_graph_create(const char* name) {
  auto g = xir::Graph::create(std::string(name));
  return static_cast<xir_graph_t>(g.release());
}
extern "C" xir_graph_t xir_graph_deserialize(const char* name) {
  auto g = xir::Graph::deserialize(std::string(name));
  return static_cast<xir_graph_t>(g.release());
}

extern "C" void xir_graph_serialize(const xir_graph_t graph,
                                    const char* file_path) {
  auto g = static_cast<xir::Graph*>(graph);
  g->serialize(std::string(file_path));
}

extern "C" int xir_graph_destroy(xir_graph_t graph) {
  auto g = static_cast<xir::Graph*>(graph);
  delete g;
  return 0;
}
static std::map<std::string, std::vector<xir::Op*>> build_input_ops(
    struct xir_graph_input_ops_t* input_ops, size_t num_of_ops) {
  std::map<std::string, std::vector<xir::Op*>> ret;
  for (auto i = 0u; i < num_of_ops; ++i) {
    auto begin = (xir::Op**)(input_ops[i].ops);
    auto size = input_ops[i].num_of_ops;
    ret.emplace(conv_to_std_string(input_ops[i].name),
                std::vector<xir::Op*>(begin, begin + size));
  }
  return ret;
}

extern "C" xir_op_t xir_graph_add_op(
    xir_graph_t graph,                        //
    xir_string_t name,                        //
    xir_string_t type,                        //
    xir_attrs_t attrs,                        //
    struct xir_graph_input_ops_t* input_ops,  //
    size_t num_of_ops,                        //
    xir_subgraph_t subgraph                   //
) {
  auto self = static_cast<xir::Graph*>(graph);
  auto ret =
      self->add_op(conv_to_std_string(name), conv_to_std_string(type),
                   std::unique_ptr<xir::Attrs>(static_cast<xir::Attrs*>(attrs)),
                   build_input_ops(input_ops, num_of_ops),
                   static_cast<xir::Subgraph*>(subgraph));
  return static_cast<xir_op_t>(ret);
}

extern "C" void xir_graph_remove_op(xir_graph_t graph, xir_op_t op) {
  auto self = static_cast<xir::Graph*>(graph);
  self->remove_op(static_cast<xir::Op*>(op));
}

extern "C" int xir_graph_get_op_num(xir_graph_t graph) {
  return static_cast<xir::Graph*>(graph)->get_op_num();
}

extern "C" xir_op_t xir_graph_get_op(xir_graph_t graph, xir_string_t op) {
  return static_cast<xir_op_t>(
      static_cast<xir::Graph*>(graph)->get_op(conv_to_std_string(op)));
}

extern "C" void xir_graph_get_ops(xir_graph_t graph, xir_op_t ret[]) {
  return xir::c_api::xir_graph_get_ops(graph, ret);
}

extern "C" void xir_graph_get_tensors(xir_graph_t graph, xir_tensor_t ret[]) {
  return xir::c_api::xir_graph_get_tensors(graph, ret);
}

extern "C" void xir_graph_get_head_ops(xir_graph_t graph, void* data,
                                       xir_get_op_callback_t cb) {
  auto self = static_cast<xir::Graph*>(graph);
  auto ops = self->get_head_ops();
  for (auto& op : ops) {
    cb(data, op);
  }
}

extern "C" void xir_graph_get_tail_ops(xir_graph_t graph, void* data,
                                       xir_get_op_callback_t cb) {
  auto self = static_cast<xir::Graph*>(graph);
  auto ops = self->get_tail_ops();
  for (auto& op : ops) {
    cb(data, op);
  }
}

extern "C" xir_op_t xir_graph_get_tensor_producer(xir_graph_t graph,
                                                  xir_tensor_t tensor) {
  auto self = static_cast<xir::Graph*>(graph);
  return static_cast<xir_op_t>(
      self->get_tensor_producer(static_cast<xir::Tensor*>(tensor)));
}

extern "C" void xir_graph_topological_sort(xir_tensor_t graph, void* data,
                                           xir_get_op_callback_t cb) {
  auto self = static_cast<xir::Graph*>(graph);
  auto ops = self->topological_sort();
  for (auto& op : ops) {
    cb(data, op);
  }
}

extern "C" xir_tensor_t xir_graph_get_tensor(xir_graph_t graph,
                                             xir_string_t tensor_name) {
  auto self = static_cast<xir::Graph*>(graph);
  return static_cast<xir_tensor_t>(
      self->get_tensor(conv_to_std_string(tensor_name)));
}

extern "C" xir_subgraph_t xir_graph_get_root_subgraph(xir_graph_t graph) {
  return static_cast<xir_subgraph_t>(
      static_cast<xir::Graph*>(graph)->get_root_subgraph());
}

extern "C" xir_subgraph_t xir_graph_get_leaf_subgraph(xir_graph_t graph,
                                                      xir_op_t op1) {
  auto self = static_cast<xir::Graph*>(graph);
  auto op = static_cast<xir::Op*>(op1);
  return static_cast<xir_subgraph_t>(self->get_leaf_subgraph(op));
}

extern "C" xir_subgraph_t xir_graph_get_subgraph(xir_graph_t graph,
                                                 xir_string_t name) {
  auto self = static_cast<xir::Graph*>(graph);
  return static_cast<xir_subgraph_t>(
      self->get_subgraph(conv_to_std_string(name)));
}

extern "C" xir_attrs_t xir_graph_get_attrs(xir_graph_t graph) {
  return xir::c_api::graph_get_attrs(graph);
}

extern "C" void xir_graph_set_attrs(xir_graph_t graph, xir_attrs_t attrs) {
  auto self = static_cast<xir::Graph*>(graph);
  self->set_attrs(std::unique_ptr<xir::Attrs>(static_cast<xir::Attrs*>(attrs)));
}
