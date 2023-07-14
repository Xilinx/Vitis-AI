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

#pragma once

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/isomorphism.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/vf2_sub_graph_iso.hpp>
#include <boost/property_map/dynamic_property_map.hpp>

#include "xir/attrs/attrs_imp.hpp"
#include "xir/graph/graph.hpp"
#include "xir/graph/graph_template_imp.hpp"
#include "xir/util/data_type.hpp"

/*
 * Filename: graph_imp.hpp
 *
 * Description:
 * This file defines Graph implementation
 */

namespace xir {

class OpImp;
class SubgraphImp;
namespace v1 {
class Serialize;
}
class GraphImp : public Graph {
 public:
  using VertexPropertyType =
      boost::property<boost::vertex_index_t, size_t, std::unique_ptr<OpImp>>;
  using GraphPropertyType = boost::property<boost::graph_name_t, std::string>;
  using GraphType =
      boost::adjacency_list<boost::listS,           // OutEdgeList
                            boost::listS,           // VertexList
                            boost::bidirectionalS,  // Directed
                            VertexPropertyType,     // VertexProperties
                            boost::no_property,     // EdgeProperties
                            GraphPropertyType       // GraphProperties
                            >;
  using VertexD = boost::graph_traits<GraphType>::vertex_descriptor;
  using EdgeD = boost::graph_traits<GraphType>::edge_descriptor;
  using VertexI = boost::graph_traits<GraphType>::vertex_iterator;
  using OutEdgeI = boost::graph_traits<GraphType>::out_edge_iterator;
  using InEdgeI = boost::graph_traits<GraphType>::in_edge_iterator;

 public:
  GraphImp() = delete;
  GraphImp(std::string name);
  virtual ~GraphImp() = default;

  const std::string get_name() const override;

  Op* add_op(const std::string& name, const std::string& type,
             std::unique_ptr<Attrs> attrs,
             const std::map<std::string, std::vector<Op*>>& input_ops_map,
             Subgraph* subgraph = nullptr) override;

  Op* add_op(const std::string& name, const std::string& type,
             std::unique_ptr<Attrs> attrs,
             const std::map<std::string, std::vector<Op*>>& input_ops_map,
             const DataType& output_data_type,
             Subgraph* subgraph = nullptr) override;

  void remove_op(Op* op) override;
  void replace_ops(
      std::map<std::string, Op*> replaced_ops,
      std::function<std::map<Op*, Op*>(Graph*, std::map<std::string, Op*>)>
          creator) override;

  Op* get_op(const std::string& op_name) override;
  const Op* get_op(const std::string& op_name) const override;
  Tensor* get_tensor(const std::string& tensor_name) override;
  const Tensor* get_tensor(const std::string& tensor_name) const override;
  int get_op_num() const override;
  std::set<Op*> get_ops() override;
  const std::set<const Op*> get_ops() const override;
  std::set<Tensor*> get_tensors() override;
  const std::set<const Tensor*> get_tensors() const override;
  std::set<Op*> get_head_ops() override;
  const std::set<const Op*> get_head_ops() const override;
  std::set<Op*> get_tail_ops() override;
  const std::set<const Op*> get_tail_ops() const override;

  Op* get_tensor_producer(const Tensor* tensor) override;
  const Op* get_tensor_producer(const Tensor* tensor) const override;

  // algorithm topologic sort
  std::vector<Op*> topological_sort() override;

  // algorithm topologic sort
  const std::vector<const Op*> topological_sort() const override;

  void infer_shape() override;

  // algorithm isomorphism
  std::vector<std::map<OpTemplate*, Op*>> isomorphism(
      GraphTemplate* graph_template) override;

  // visualization
  void save_to_dot(const std::string& file_name) const override;
  void visualize(const std::string& filename,
                 const std::string& format) const override;

  // serialize and de-serialize
  friend class v1::Serialize;
  void serialize(const std::string& pb_fname = "") const override;
  void serialize_to_string(std::string* str) const override;

  // subgraph
  Subgraph* get_root_subgraph() override;
  const Subgraph* get_root_subgraph() const override;
  Subgraph* get_leaf_subgraph(const Op* op) override;
  const Subgraph* get_leaf_subgraph(const Op* op) const override;
  Subgraph* get_subgraph(const std::string& name) override;
  const Subgraph* get_subgraph(const std::string& name) const override;

  bool has_attrs() const override;

  std::unique_ptr<Attrs> get_attrs() const override;

  Graph* set_attrs(std::unique_ptr<Attrs> attrs) override;

  bool has_attr(const std::string& key) const override;

  const xir::any get_attr(const std::string& key) const override;

  Graph* set_attr(const std::string& key, const xir::any& value) override;

  const std::string to_string(const std::string& delimiter,     //
                              const std::string& left_bracket,  //
                              const std::string& right_bracket) const override;

  GraphType* get_boost_graph();

 private:
  void update_vertex_index();

 private:
  std::unique_ptr<GraphType> graph_;
  std::unique_ptr<SubgraphImp> root_subgraph_;
  std::unique_ptr<Attrs> attrs_;
  friend class c_api;
};

}  // namespace xir

#include "xir/graph/subgraph_imp.hpp"
#include "xir/op/op_imp.hpp"
