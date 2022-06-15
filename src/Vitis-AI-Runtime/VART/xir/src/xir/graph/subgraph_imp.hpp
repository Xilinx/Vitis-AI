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

#pragma once

#include <map>
#include <memory>

#include "xir/graph/graph_imp.hpp"
#include "xir/graph/subgraph.hpp"
#include "xir/op/op.hpp"
#include "xir/util/any.hpp"

namespace xir {
namespace v1 {
class Serialize;
}

using GraphvizAttributes = std::map<std::string, std::string>;

using BOOSTSubGraph = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::directedS,
    boost::property<boost::vertex_attribute_t, GraphvizAttributes>,
    boost::property<
        boost::edge_index_t, int,
        boost::property<boost::edge_attribute_t, GraphvizAttributes>>,
    boost::property<
        boost::graph_name_t, std::string,
        boost::property<
            boost::graph_graph_attribute_t, GraphvizAttributes,
            boost::property<
                boost::graph_vertex_attribute_t, GraphvizAttributes,
                boost::property<boost::graph_edge_attribute_t,
                                GraphvizAttributes>>>>>;  // the infomation in
                                                          // this definition are
                                                          // required.

class Serialize;

struct EdgePredicate {
  EdgePredicate() {}
  EdgePredicate(GraphImp* graph, const std::set<Op*>& ops)
      : graph_{graph}, ops_{ops} {}
  bool operator()(const GraphImp::EdgeD& ed) const;

 private:
  GraphImp* graph_;
  std::set<Op*> ops_;
};

struct VertexPredicate {
  VertexPredicate() {}
  VertexPredicate(const std::set<Op*>& ops) : ops_{ops} {}
  bool operator()(const GraphImp::VertexD& vd) const;

 private:
  std::set<Op*> ops_;
};

class SubgraphImp : public Subgraph {
 public:
  using FilteredGraphType =
      boost::filtered_graph<GraphImp::GraphType, EdgePredicate,
                            VertexPredicate>;

  SubgraphImp(GraphImp* graph, SubgraphImp* parent, const std::set<Op*> ops);
  SubgraphImp() = delete;
  SubgraphImp(SubgraphImp&&) = default;
  virtual ~SubgraphImp() = default;
  friend v1::Serialize;

 public:
  const std::string get_name() const override;

  Subgraph* set_name(const std::string& subgraph_name) override;

  std::int32_t get_op_num() const override;

  std::set<Op*> get_ops() override;
  const std::set<const Op*> get_ops() const override;

  Op* get_tensor_producer(const Tensor* tensor) override;
  const Op* get_tensor_producer(const Tensor* tensor) const override;

  std::set<Tensor*> get_input_tensors() override;
  const std::set<const Tensor*> get_input_tensors() const override;

  std::vector<Tensor*> get_sorted_input_tensors(
      const std::function<bool(Tensor*, Tensor*)>& compare) override;
  const std::vector<const Tensor*> get_sorted_input_tensors(
      const std::function<bool(Tensor*, Tensor*)>& compare) const override;

  std::set<Tensor*> get_output_tensors() override;
  const std::set<const Tensor*> get_output_tensors() const override;

  std::vector<Tensor*> get_sorted_output_tensors(
      const std::function<bool(Tensor*, Tensor*)>& compare) override;
  const std::vector<const Tensor*> get_sorted_output_tensors(
      const std::function<bool(Tensor*, Tensor*)>& compare) const override;

  bool has_op(const std::string& op_name) const override;

  bool has_op(const Op* op) const override;

  Subgraph* find_op(const Op* op) override;
  const Subgraph* find_op(const Op* op) const override;

  Subgraph* find_op(const std::string& op_name) override;
  const Subgraph* find_op(const std::string& op_name) const override;

  bool is_root() const override;

  bool is_leaf() const override;

  Subgraph* get_root() override;
  const Subgraph* get_root() const override;

  std::int32_t get_depth() const override;

  Subgraph* get_parent() override;
  const Subgraph* get_parent() const override;

  void create_children() override;

  std::int32_t get_children_num() const override;

  std::set<Subgraph*> get_children() override;
  const std::set<const Subgraph*> get_children() const override;

  bool is_child(Subgraph* subgraph) const override;

  Subgraph* merge_children(std::set<Subgraph*> subgraph_list) override;

  Graph* get_graph() override;
  const Graph* get_graph() const override;

  Subgraph* get_subgraph(const std::string& name) override;
  const Subgraph* get_subgraph(const std::string& name) const override;

  std::unique_ptr<Attrs> get_attrs() const override;

  bool has_attrs() const override;

  Subgraph* set_attrs(std::unique_ptr<Attrs> attrs) override;

  bool has_attr(const std::string& key) const override;

  const xir::any get_attr(const std::string& key) const override;

  Subgraph* set_attr(const std::string& key, const xir::any& value) override;

  std::unique_ptr<FilteredGraphType> get_filtered_graph() const;

  std::vector<Op*> topological_sort() override;
  const std::vector<const Op*> topological_sort() const override;

  std::vector<Subgraph*> children_topological_sort() override;
  const std::vector<const Subgraph*> children_topological_sort() const override;

  std::vector<std::map<OpTemplate*, Op*>> isomorphism(
      GraphTemplate* graph_template) override;

  void save_to_dot(const std::string& file_path) const override;

  const std::string to_string(const std::string& delimiter,     //
                              const std::string& left_bracket,  //
                              const std::string& right_bracket) const override;

 public:
  const std::uint32_t get_id_() const;

  void update_id_();
  std::uint32_t update_id_helper_(std::uint32_t id);

  std::set<Op*> filter_op_by_name_(const std::string& op_name) const;

  void add_op(Op* op);
  void add_op_helper_(Op* op);

  void remove_op(Op* op);
  void remove_op_helper_(Op* op);

  std::int32_t count_op_(const std::set<std::string>& op_types) const;

 private:
  std::string name_;

  std::uint32_t id_;

  GraphImp* graph_;

  SubgraphImp* parent_;

  std::set<std::unique_ptr<SubgraphImp>> children_;

  std::set<Op*> ops_;

  std::unique_ptr<Attrs> attrs_;
};

}  // namespace xir
