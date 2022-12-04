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

#include "xir/graph/graph_template.hpp"
#include "xir/op/op.hpp"

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

namespace xir {

class OpTemplateImp;

class GraphTemplateImp : public GraphTemplate {
 public:
  using VertexPropertyType = std::unique_ptr<OpTemplate>;
  using EdgePropertyType = std::string;
  using GraphPropertyType = boost::property<boost::graph_name_t, std::string>;
  using GraphType =
      boost::adjacency_list<boost::vecS,            // OutEdgeList
                            boost::vecS,            // VertexList
                            boost::bidirectionalS,  // Directed
                            VertexPropertyType,     // VertexProperties
                            EdgePropertyType,       // EdgeProperties
                            GraphPropertyType       // GraphProperties
                            >;
  using VertexD = boost::graph_traits<GraphType>::vertex_descriptor;
  using EdgeD = boost::graph_traits<GraphType>::edge_descriptor;

 public:
  GraphTemplateImp(std::string name);
  GraphTemplateImp() = delete;
  ~GraphTemplateImp() = default;

  const std::string get_name() const override;

  OpTemplate* add_op(const std::string name,
                     const std::set<std::string> types) override;

  OpTemplate* add_op(
      const std::string name, const std::set<std::string> types,
      const std::map<OpTemplate*, std::string> input_ops) override;

  OpTemplate* get_op(const std::string op_name) override;

  std::set<OpTemplate*> get_ops() override;

  void remove_op(OpTemplate* op) override;

  int get_op_num() const override;

  void set_filter(
      const std::function<bool(std::map<OpTemplate*, Op*>)>&) override;

  const std::function<bool(std::map<OpTemplate*, Op*>)>& get_filter()
      const override;

  const std::vector<OpTemplate*> topological_sort() const override;

  std::function<void(std::ostream& out, const GraphTemplateImp::VertexD& vd)>
  get_vertex_writer();
  std::function<void(std::ostream& out, const GraphTemplateImp::EdgeD& ed)>
  get_edge_writer();
  void save_to_dot(const std::string& file_name) override;
  void visualize(const std::string& file_name,
                 const std::string& format) override;

  GraphType* get_boost_graph();

 private:
  std::unique_ptr<GraphType> graph_;
  std::function<bool(std::map<OpTemplate*, Op*>)> filter_;
};

class OpTemplateImp : public OpTemplate {
 public:
  OpTemplateImp(const GraphTemplateImp::VertexD vd, const std::string name,
                const std::set<std::string> types, GraphTemplateImp* graph);
  OpTemplateImp() = delete;
  ~OpTemplateImp() = default;

  const std::string get_name() const override;

  void set_types(std::set<std::string> types) override;

  const std::set<std::string> get_types() const override;

  const int get_input_num() const override;

  const std::set<OpTemplate*> get_input_ops() const override;

  const int get_fanout_num() const override;

  const std::set<OpTemplate*> get_fanout_ops() const override;

  void set_filter(const std::function<bool(Op*)>&) override;

  const std::function<bool(Op*)>& get_filter() const override;

  void replace_input_op(OpTemplate* op_old, OpTemplate* op_new) override;

 public:
  const GraphTemplateImp::VertexD vd_;

 private:
  const std::string name_;
  std::set<std::string> types_;
  std::function<bool(Op*)> filter_;

  GraphTemplateImp* graph_;
};

}  // namespace xir
