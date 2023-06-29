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
#include "xir/graph/subgraph_imp.hpp"

/* C API implementations */
namespace xir {
class c_api {
 public:
  static xir_string_t subgraph_get_name(xir_subgraph_t subgraph) {
    auto self = static_cast<xir::SubgraphImp*>(subgraph);
    return conv_to_xir_string(self->name_);
  }

  static xir_attrs_t subgraph_get_attrs(xir_subgraph_t subgraph) {
    auto self = static_cast<xir::SubgraphImp*>(subgraph);
    return static_cast<xir_subgraph_t>(self->attrs_.get());
  }

  // static xir_subgraph_t subgraph_get_child(xir_subgraph_t subgraph,
  //                                        int32_t idx) {
  // auto self = static_cast<xir::SubgraphImp*>(subgraph);
  // auto child = self->children_.begin();
  // for (int i = 0; i < idx; i++) {
  //  child++;
  // }
  // return static_cast<xir_subgraph_t>((*child).get());
  // }
  static void subgraph_get_children(xir_subgraph_t subgraph,
                                    xir_subgraph_t children[]) {
    auto self = static_cast<xir::SubgraphImp*>(subgraph);

    auto ret = self->get_children();
    int i = 0;
    for (auto& child : ret) {
      children[i++] = static_cast<xir_subgraph_t>(child);
    }
  }

  static void subgraph_children_topological_sort(xir_subgraph_t subgraph,
                                                 xir_subgraph_t children[]) {
    auto self = static_cast<xir::SubgraphImp*>(subgraph);
    auto ret = self->children_topological_sort();
    int i = 0;
    for (auto& child : ret) {
      children[i++] = static_cast<xir_subgraph_t>(child);
    }
  }
};
};  // namespace xir
extern "C" xir_string_t xir_subgraph_get_name(xir_subgraph_t subgraph) {
  return xir::c_api::subgraph_get_name(subgraph);
}
extern "C" void xir_subgraph_set_name(xir_subgraph_t subgraph,
                                      xir_string_t subgraph_name) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  self->set_name(conv_to_std_string(subgraph_name));
}

extern "C" int32_t xir_subgraph_get_op_num(xir_subgraph_t subgraph) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return self->get_op_num();
}

extern "C" void xir_subgraph_get_ops(xir_subgraph_t subgraph, void* data,
                                     xir_get_op_callback_t cb) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  for (auto& op : self->get_ops()) {
    cb(data, op);
  }
}

extern "C" xir_op_t xir_subgraph_get_tensor_producer(xir_subgraph_t subgraph,
                                                     xir_tensor_t tensor) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return static_cast<xir_op_t>(
      self->get_tensor_producer(static_cast<xir::Tensor*>(tensor)));
}

extern "C" void xir_subgraph_get_input_tensors(xir_subgraph_t subgraph,
                                               void* data,
                                               xir_get_tensor_callback_t cb) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  for (auto& t : self->get_input_tensors()) {
    cb(data, t);
  }
}
extern "C" void xir_subgraph_get_output_tensors(xir_subgraph_t subgraph,
                                                void* data,
                                                xir_get_tensor_callback_t cb) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  for (auto& t : self->get_output_tensors()) {
    cb(data, t);
  }
}

extern "C" int xir_subgraph_has_op_by_name(xir_subgraph_t subgraph,
                                           xir_string_t op_name) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return self->has_op(conv_to_std_string(op_name));
}

extern "C" int xir_subgraph_has_op(xir_subgraph_t subgraph, xir_op_t op) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return self->has_op(static_cast<xir::Op*>(op));
}

extern "C" xir_subgraph_t xir_subgraph_find_op_by_name(xir_subgraph_t subgraph,
                                                       xir_string_t op_name) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return static_cast<xir_subgraph_t>(
      self->find_op(conv_to_std_string(op_name)));
}

extern "C" xir_subgraph_t xir_subgraph_find_op(xir_subgraph_t subgraph,
                                               xir_op_t op) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return static_cast<xir_subgraph_t>(self->find_op(static_cast<xir::Op*>(op)));
}

extern "C" int32_t xir_subgraph_get_children_num(xir_subgraph_t subgraph) {
  return static_cast<xir::Subgraph*>(subgraph)->get_children_num();
}
// extern "C" xir_subgraph_t xir_subgraph_get_child(xir_subgraph_t
// subgraph,
//                                                   int32_t idx) {
//  return xir::c_api::subgraph_get_child(subgraph, idx);
//}
extern "C" int xir_subgraph_is_root(xir_subgraph_t subgraph) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return self->is_root();
}

extern "C" int xir_subgraph_is_leaf(xir_subgraph_t subgraph) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return self->is_leaf();
}

extern "C" xir_subgraph_t xir_subgraph_get_root(xir_subgraph_t subgraph) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return static_cast<xir_subgraph_t>(self->get_root());
}

extern "C" int32_t xir_subgraph_get_depth(xir_subgraph_t subgraph) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return self->get_depth();
}

extern "C" xir_subgraph_t xir_subgraph_get_parent(xir_subgraph_t subgraph) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return static_cast<xir_subgraph_t>(self->get_parent());
}

extern "C" void xir_subgraph_create_children(xir_subgraph_t subgraph) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return self->create_children();
}

extern "C" void xir_subgraph_get_children(xir_subgraph_t subgraph,
                                          xir_subgraph_t children[]) {
  return xir::c_api::subgraph_get_children(subgraph, children);
}

extern "C" int xir_subgraph_is_child(xir_subgraph_t subgraph,
                                     xir_subgraph_t child) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return self->is_child(static_cast<xir::Subgraph*>(child));
}

extern "C" xir_graph_t xir_subgraph_get_graph(xir_subgraph_t subgraph) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return static_cast<xir_graph_t>(self->get_graph());
}

extern "C" int xir_subgraph_has_attrs(xir_subgraph_t subgraph) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return self->has_attrs();
}

extern "C" xir_attrs_t xir_subgraph_get_attrs(xir_subgraph_t subgraph) {
  return xir::c_api::subgraph_get_attrs(subgraph);
}

extern "C" void xir_subgraph_set_attrs(xir_subgraph_t subgraph,
                                       xir_attrs_t attrs) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  self->set_attrs(std::unique_ptr<xir::Attrs>(static_cast<xir::Attrs*>(attrs)));
}
extern "C" int xir_subgraph_has_attr(xir_subgraph_t subgraph, const char* key) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return self->has_attr(std::string(key));
}

extern "C" void xir_subgraph_children_topological_sort(
    xir_subgraph_t subgraph, xir_subgraph_t children[]) {
  return xir::c_api::subgraph_children_topological_sort(subgraph, children);
}

extern "C" void xir_subgraph_save_to_dot(xir_subgraph_t subgraph,
                                         const char* file_name) {
  auto self = static_cast<xir::Subgraph*>(subgraph);
  return self->save_to_dot(std::string(file_name));
}
