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

#include "xir/graph/graph_imp.hpp"

#include "UniLog/UniLog.hpp"
#include "xir/graph/serialize_v2.hpp"
#include "xir/op/op_def.hpp"
#include "xir/util/internal_util.hpp"
#include "xir/util/tool_function.hpp"

namespace xir {

GraphImp::GraphImp(std::string name)
    : graph_{std::make_unique<GraphType>(0)},
      root_subgraph_{
          std::make_unique<SubgraphImp>(this, nullptr, std::set<Op*>{})} {
  attrs_ = Attrs::create();
  boost::get_property(*graph_, boost::graph_name) = name;
}

const std::string GraphImp::get_name() const {
  return boost::get_property(*graph_, boost::graph_name);
}

Op* GraphImp::add_op(
    const std::string& name, const std::string& type,
    std::unique_ptr<Attrs> attrs,
    const std::map<std::string, std::vector<Op*>>& map_input_ops,
    Subgraph* subgraph) {
  return this->add_op(name, type, std::move(attrs), map_input_ops, DataType{},
                      subgraph);
}

Op* GraphImp::add_op(
    const std::string& name, const std::string& type,
    std::unique_ptr<Attrs> attrs,
    const std::map<std::string, std::vector<Op*>>& map_input_ops,
    const DataType& output_data_type, Subgraph* subgraph) {
  const auto op_range = vertices(*graph_);
  UNI_LOG_CHECK(std::find_if(op_range.first, op_range.second,
                             [this, name](const VertexD& vd) -> bool {
                               return !((*graph_)[vd]->to_be_removed_) &&
                                      (*graph_)[vd]->get_name() == name;
                             }) == op_range.second,
                XIR_MULTI_DEFINED_OP)
      << name;
  auto sub = subgraph == nullptr ? root_subgraph_.get()
                                 : static_cast<SubgraphImp*>(subgraph);

  auto vd = boost::add_vertex(*graph_);
  (*graph_)[vd] =
      std::make_unique<OpImp>(vd, name, type, std::move(attrs), map_input_ops,
                              nullptr, this, output_data_type);
  update_vertex_index();
  auto ret = op_up_cast((*graph_)[vd].get());
  sub->add_op(ret);
  return ret;
}

void GraphImp::remove_op(Op* op) {
  auto fanout_ops = op->get_fanout_ops();
  UNI_LOG_CHECK(
      fanout_ops.size() == 0 ||
          std::all_of(
              fanout_ops.begin(), fanout_ops.end(),
              [](Op* op) { return static_cast<OpImp*>(op)->to_be_removed_; }),
      XIR_REMOVE_OP_FAIL)
      << "Cannot remove " << op->to_string()
      << " from graph. Because there is at least one op who takes its output "
         "as input.";
  root_subgraph_->remove_op(op);
  auto ptr = op_down_cast(op);
  boost::clear_vertex(ptr->vd_, *graph_);
  boost::remove_vertex(ptr->vd_, *graph_);
  update_vertex_index();
}

void GraphImp::replace_ops(
    std::map<std::string, Op*> replaced_ops,
    std::function<std::map<Op*, Op*>(Graph*, std::map<std::string, Op*>)>
        creator) {
  for (auto op : replaced_ops) {
    static_cast<OpImp*>(op.second)->to_be_removed_ = true;
  }
  auto corrs = creator(static_cast<Graph*>(this), replaced_ops);
  for (auto corr : corrs) {
    for (auto fanout_op : corr.first->get_fanout_ops()) {
      fanout_op->replace_input_op(corr.first, corr.second);
    }
  }
  for (auto op : replaced_ops) {
    this->remove_op(op.second);
  }
}

Op* GraphImp::get_op(const std::string& op_name) {
  return const_cast<Op*>(static_cast<const GraphImp&>(*this).get_op(op_name));
}

const Op* GraphImp::get_op(const std::string& op_name) const {
  Op* ret = nullptr;
  for (auto vd : boost::make_iterator_range(boost::vertices(*graph_))) {
    if ((*graph_)[vd]->get_name() == op_name) {
      ret = (*graph_)[vd].get();
      break;
    }
  }
  return ret;
}

Tensor* GraphImp::get_tensor(const std::string& tensor_name) {
  return const_cast<Tensor*>(
      static_cast<const GraphImp&>(*this).get_tensor(tensor_name));
}

const Tensor* GraphImp::get_tensor(const std::string& tensor_name) const {
  Tensor* ret = nullptr;
  for (auto vd : boost::make_iterator_range(boost::vertices(*graph_))) {
    if ((*graph_)[vd]->get_output_tensor()->get_name() == tensor_name) {
      ret = (*graph_)[vd]->get_output_tensor();
      break;
    }
  }
  return ret;
}

std::set<Op*> GraphImp::get_ops() {
  return internal::cast_from_const_set(
      static_cast<const GraphImp&>(*this).get_ops());
}

const std::set<const Op*> GraphImp::get_ops() const {
  auto ret = std::set<const Op*>{};
  for (auto vd : boost::make_iterator_range(boost::vertices(*graph_))) {
    ret.insert(op_up_cast((*graph_)[vd].get()));
  }
  return ret;
}

std::set<Tensor*> GraphImp::get_tensors() {
  return internal::cast_from_const_set(
      static_cast<const GraphImp&>(*this).get_tensors());
}

const std::set<const Tensor*> GraphImp::get_tensors() const {
  auto ret = std::set<const Tensor*>{};
  auto ops = get_ops();
  std::transform(
      ops.begin(), ops.end(), std::inserter(ret, ret.end()),
      [](const Op* op) -> const Tensor* { return op->get_output_tensor(); });
  return ret;
}

std::set<Op*> GraphImp::get_head_ops() {
  return internal::cast_from_const_set(
      static_cast<const GraphImp&>(*this).get_head_ops());
}

const std::set<const Op*> GraphImp::get_head_ops() const {
  std::set<const Op*> ret;
  for (auto op : this->get_ops()) {
    if (op->get_input_num() == 0) {
      ret.insert(op);
    }
  }
  return ret;
}

std::set<Op*> GraphImp::get_tail_ops() {
  return internal::cast_from_const_set(
      static_cast<const GraphImp&>(*this).get_tail_ops());
}

const std::set<const Op*> GraphImp::get_tail_ops() const {
  std::set<const Op*> ret;
  for (auto op : this->get_ops()) {
    if (op->get_fanout_num() == 0) {
      ret.insert(op);
    }
  }
  return ret;
}

Op* GraphImp::get_tensor_producer(const Tensor* tensor) {
  return const_cast<Op*>(
      static_cast<const GraphImp&>(*this).get_tensor_producer(tensor));
}

const Op* GraphImp::get_tensor_producer(const Tensor* tensor) const {
  const Op* ret = nullptr;
  for (auto vd : boost::make_iterator_range(boost::vertices(*graph_))) {
    if ((*graph_)[vd]->get_output_tensor() == tensor) {
      ret = (*graph_)[vd].get();
      break;
    }
  }
  return ret;
}

int GraphImp::get_op_num() const { return boost::num_vertices(*graph_); }

std::vector<Op*> GraphImp::topological_sort() {
  return internal::cast_from_const_vector(
      static_cast<const GraphImp&>(*this).topological_sort());
}

const std::vector<const Op*> GraphImp::topological_sort() const {
  auto ret_vd = std::vector<VertexD>{};
  boost::topological_sort(*graph_, std::back_inserter(ret_vd));
  auto ret = std::vector<const Op*>{ret_vd.size()};
  std::transform(ret_vd.begin(), ret_vd.end(), ret.rbegin(),
                 [this](const VertexD& vd) { return (*graph_)[vd].get(); });
  return ret;
}

void GraphImp::infer_shape() {
  auto ops = this->topological_sort();
  for (auto op : ops) {
    op->shape_infer();
  }
}

std::vector<std::map<OpTemplate*, Op*>> GraphImp::isomorphism(
    GraphTemplate* graph_template) {
  return this->get_root_subgraph()->isomorphism(graph_template);
}

void GraphImp::save_to_dot(const std::string& file_path) const {
  this->get_root_subgraph()->save_to_dot(file_path);
}

void GraphImp::visualize(const std::string& file_path,
                         const std::string& format) const {
  auto file_name_dot = file_path + ".dot";
  save_to_dot(file_name_dot);
  auto create_fig_from_dot =
      "dot -T" + format + " " + file_name_dot + " -o " + file_path;
  auto result = std::system(create_fig_from_dot.c_str());
  UNI_LOG_CHECK(result == 0, XIR_OPERATION_FAILED)
      << "command: \"" << create_fig_from_dot << "\", std::system exit("
      << result << ")";
  auto rm_dot = "rm " + file_name_dot;
  result = std::system(rm_dot.c_str());
  UNI_LOG_CHECK(result == 0, XIR_OPERATION_FAILED)
      << "command: \"" << rm_dot << "\", std::system exit(" << result << ")";
}

void GraphImp::serialize(const std::string& file_path) const {
  v2::Serialize s{};
  s.write(this, file_path);
}

void GraphImp::serialize_to_string(std::string* str) const {
  v2::Serialize s{};
  s.write_to_string(this, str);
}

GraphImp::GraphType* GraphImp::get_boost_graph() { return graph_.get(); }

void GraphImp::update_vertex_index() {
  auto vertex_index_map = boost::get(boost::vertex_index, *graph_);
  auto idx = 0U;
  for (auto vd : boost::make_iterator_range(boost::vertices(*graph_))) {
    boost::put(vertex_index_map, vd, idx++);
  }
}

const Subgraph* GraphImp::get_root_subgraph() const {
  return static_cast<Subgraph*>(root_subgraph_.get());
}

Subgraph* GraphImp::get_root_subgraph() {
  return const_cast<Subgraph*>(
      static_cast<const GraphImp&>(*this).get_root_subgraph());
}

Subgraph* GraphImp::get_leaf_subgraph(const Op* op) {
  return const_cast<Subgraph*>(
      static_cast<const GraphImp&>(*this).get_leaf_subgraph(op));
}

const Subgraph* GraphImp::get_leaf_subgraph(const Op* op) const {
  auto ret = static_cast<Subgraph*>(root_subgraph_.get());
  UNI_LOG_CHECK(ret->has_op(op), XIR_UNDEFINED_OP);
  while (!ret->is_leaf()) {
    ret = ret->find_op(op);
  }
  return ret;
}

Subgraph* GraphImp::get_subgraph(const std::string& name) {
  return const_cast<Subgraph*>(
      static_cast<const GraphImp&>(*this).get_subgraph(name));
}

const Subgraph* GraphImp::get_subgraph(const std::string& name) const {
  if (root_subgraph_->get_name() == name) return root_subgraph_.get();
  return root_subgraph_->get_subgraph(name);
}

bool GraphImp::has_attrs() const { return !(nullptr == attrs_); }

std::unique_ptr<Attrs> GraphImp::get_attrs() const {
  if (nullptr == attrs_) {
    return Attrs::create();
  } else {
    return Attrs::clone(attrs_.get());
  }
}

Graph* GraphImp::set_attrs(std::unique_ptr<Attrs> attrs) {
  attrs_ = std::move(attrs);
  return this;
}

bool GraphImp::has_attr(const std::string& key) const {
  if (nullptr == attrs_) {
    return false;
  }
  return attrs_->has_attr(key);
}

const xir::any GraphImp::get_attr(const std::string& key) const {
  return attrs_->get_attr(key);
}

Graph* GraphImp::set_attr(const std::string& key, const xir::any& value) {
  attrs_->set_attr(key, value);
  return this;
}

const std::string GraphImp::to_string(const std::string& delimiter,     //
                                      const std::string& left_bracket,  //
                                      const std::string& right_bracket) const {
  std::ostringstream out;
  out << "xir::Graph" << left_bracket                //
      << "name = " << this->get_name() << delimiter  //
      << " ops_num = " << this->get_op_num()         //
      << right_bracket;
  return out.str();
}
}  // namespace xir
