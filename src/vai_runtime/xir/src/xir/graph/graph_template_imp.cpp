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

#include "xir/graph/graph_template_imp.hpp"
#include "xir/op/op_def.hpp"

#include "UniLog/UniLog.hpp"

namespace xir {

OpTemplateImp::OpTemplateImp(const GraphTemplateImp::VertexD vd,
                             const std::string name,
                             const std::set<std::string> types,
                             GraphTemplateImp* graph)
    : vd_{vd}, name_{name}, types_{types}, graph_{graph} {
  filter_ = [](Op* op) { return true; };
}

void OpTemplateImp::replace_input_op(OpTemplate* op_old, OpTemplate* op_new) {
  UNI_LOG_CHECK(op_old != nullptr && op_new != nullptr, XIR_UNEXPECTED_VALUE)
      << "Op_old and op_new can not be empty!";
  auto input_ops = get_input_ops();
  for (auto iter = input_ops.begin(); iter != input_ops.end(); ++iter) {
    if (op_old == static_cast<OpTemplateImp*>(*iter)) {
      auto ed = boost::edge(static_cast<OpTemplateImp*>(op_old)->vd_, vd_,
                            *graph_->get_boost_graph())
                    .first;
      boost::remove_edge(ed, *graph_->get_boost_graph());
      boost::add_edge(static_cast<OpTemplateImp*>(op_new)->vd_, vd_,
                      *graph_->get_boost_graph());
    }
  }
}

const std::string OpTemplateImp::get_name() const { return name_; }

void OpTemplateImp::set_types(std::set<std::string> types) {
  types_ = move(types);
}

const std::set<std::string> OpTemplateImp::get_types() const { return types_; }

const int OpTemplateImp::get_input_num() const {
  return boost::in_degree(vd_, *graph_->get_boost_graph());
}

const std::set<OpTemplate*> OpTemplateImp::get_input_ops() const {
  auto ret = std::set<OpTemplate*>{};
  for (auto ed : boost::make_iterator_range(
           boost::in_edges(vd_, *graph_->get_boost_graph()))) {
    ret.insert((*graph_->get_boost_graph())[boost::source(
                                                ed, *graph_->get_boost_graph())]
                   .get());
  }
  return ret;
}

const int OpTemplateImp::get_fanout_num() const {
  return boost::out_degree(vd_, *graph_->get_boost_graph());
}

const std::set<OpTemplate*> OpTemplateImp::get_fanout_ops() const {
  auto ret = std::set<OpTemplate*>{};
  for (auto ed : boost::make_iterator_range(
           boost::out_edges(vd_, *graph_->get_boost_graph()))) {
    ret.insert((*graph_->get_boost_graph())[boost::target(
                                                ed, *graph_->get_boost_graph())]
                   .get());
  }
  return ret;
}

void OpTemplateImp::set_filter(const std::function<bool(Op*)>& filter) {
  filter_ = filter;
}

const std::function<bool(Op*)>& OpTemplateImp::get_filter() const {
  return filter_;
}

GraphTemplateImp::GraphTemplateImp(std::string name)
    : graph_{std::make_unique<GraphType>(0)} {
  boost::get_property(*graph_, boost::graph_name) = name;
  filter_ = [](std::map<OpTemplate*, Op*> map) { return true; };
}

const std::string GraphTemplateImp::get_name() const {
  return boost::get_property(*graph_, boost::graph_name);
}

OpTemplate* GraphTemplateImp::add_op(const std::string name,
                                     const std::set<std::string> types) {
  const auto op_range = vertices(*graph_);
  UNI_LOG_CHECK(std::find_if(op_range.first, op_range.second,
                             [this, name](const VertexD& vd) -> bool {
                               return (*graph_)[vd]->get_name() == name;
                             }) == op_range.second,
                XIR_MULTI_DEFINED_OP)
      << name;

  auto vd = boost::add_vertex(*graph_);
  (*graph_)[vd] = std::unique_ptr<OpTemplate>{
      static_cast<OpTemplate*>(new OpTemplateImp{vd, name, types, this})};
  return (*graph_)[vd].get();
}

OpTemplate* GraphTemplateImp::add_op(
    const std::string name, const std::set<std::string> types,
    const std::map<OpTemplate*, std::string> input_ops) {
  const auto op_range = vertices(*graph_);
  UNI_LOG_CHECK(std::find_if(op_range.first, op_range.second,
                             [this, name](const VertexD& vd) -> bool {
                               return (*graph_)[vd]->get_name() == name;
                             }) == op_range.second,
                XIR_MULTI_DEFINED_OP)
      << name;

  auto vd = boost::add_vertex(*graph_);
  (*graph_)[vd] = std::unique_ptr<OpTemplate>{
      static_cast<OpTemplate*>(new OpTemplateImp{vd, name, types, this})};
  for (auto input_op : input_ops) {
    auto ed = boost::add_edge(static_cast<OpTemplateImp*>(input_op.first)->vd_,
                              vd, *graph_);
    (*graph_)[ed.first] = input_op.second;
  }
  return (*graph_)[vd].get();
}

OpTemplate* GraphTemplateImp::get_op(const std::string op_name) {
  OpTemplate* ret = nullptr;
  for (auto vd : boost::make_iterator_range(boost::vertices(*graph_))) {
    if ((*graph_)[vd]->get_name() == op_name) {
      ret = (*graph_)[vd].get();
      break;
    }
  }
  return ret;
}

std::set<OpTemplate*> GraphTemplateImp::get_ops() {
  auto ret = std::set<OpTemplate*>{};
  for (auto vd : boost::make_iterator_range(boost::vertices(*graph_))) {
    ret.insert((*graph_)[vd].get());
  }
  return ret;
}

void GraphTemplateImp::remove_op(OpTemplate* op) {
  UNI_LOG_CHECK(op != nullptr, XIR_UNEXPECTED_VALUE) << "Null pointer!";
  auto op_name = op->get_name();
  UNI_LOG_CHECK(get_op(op_name) != nullptr, XIR_REMOVE_OP_FAIL)
      << "Can not find " << op_name << " in graph template.";
  auto fanout_ops = op->get_fanout_ops();
  UNI_LOG_CHECK(fanout_ops.size() == 0, XIR_REMOVE_OP_FAIL)
      << "Cannot remove " << op->get_name()
      << " from graph. Because there is at least one op who takes its output "
         "as input.";
  auto ptr = static_cast<OpTemplateImp*>(op);
  boost::clear_vertex(ptr->vd_, *graph_);
  boost::remove_vertex(ptr->vd_, *graph_);
}

void GraphTemplateImp::set_filter(
    const std::function<bool(std::map<OpTemplate*, Op*>)>& filter) {
  filter_ = filter;
}

const std::function<bool(std::map<OpTemplate*, Op*>)>&
GraphTemplateImp::get_filter() const {
  return filter_;
}

int GraphTemplateImp::get_op_num() const {
  return boost::num_vertices(*graph_);
}

const std::vector<OpTemplate*> GraphTemplateImp::topological_sort() const {
  auto ret_vd = std::vector<VertexD>{};
  boost::topological_sort(*graph_, std::back_inserter(ret_vd));
  auto ret = std::vector<OpTemplate*>{ret_vd.size()};
  std::transform(ret_vd.begin(), ret_vd.end(), ret.rbegin(),
                 [this](const VertexD& vd) { return (*graph_)[vd].get(); });
  return ret;
}

static std::string fold_str(std::string in, uint32_t stride) {
  std::string ret;
  for (auto idx = 0U; idx < in.length(); idx++) {
    ret.push_back(in[idx]);
    if (idx != 0 && idx != (in.length() - 1) && idx % stride == 0) {
      ret.push_back('\n');
    }
  }
  return ret;
}

std::function<void(std::ostream& out, const GraphTemplateImp::VertexD& vd)>
GraphTemplateImp::get_vertex_writer() {
  return [this](std::ostream& out, const VertexD& vd) {
    out << "["
        << "label=\""
        << fold_str((*graph_)[vd]->get_name(),
                    30)  //
        << "\n(type=";
    auto types = (*graph_)[vd]->get_types();
    for (auto type : types) out << type << ", ";
    out << ")\""         //
        << ",shape=box"  //
        << "]";
  };
}

std::function<void(std::ostream& out, const GraphTemplateImp::EdgeD& ed)>
GraphTemplateImp::get_edge_writer() {
  return [](std::ostream& out, const EdgeD& ed) {
    out << "["  //
        << "]";
  };
}

void GraphTemplateImp::save_to_dot(const std::string& file_name) {
  std::ofstream f(file_name);
  boost::write_graphviz(
      f, *graph_, get_vertex_writer(), get_edge_writer(),
      [this](std::ostream& os) { os << "name=\"" << get_name() << "\";\n"; });
}

void GraphTemplateImp::visualize(const std::string& filename,
                                 const std::string& format) {
  auto file_name_dot = filename + ".dot";
  save_to_dot(file_name_dot);
  auto create_fig_from_dot = "dot -T" + format + " " + file_name_dot + " -o " +
                             filename + "." + format;
  auto result = std::system(create_fig_from_dot.c_str());
  UNI_LOG_CHECK(result == 0, XIR_OPERATION_FAILED)
      << "command: \"" << create_fig_from_dot << "\", std::system exit("
      << result << ")";
  auto rm_dot = "rm " + file_name_dot;
  result = std::system(rm_dot.c_str());
  UNI_LOG_CHECK(result == 0, XIR_OPERATION_FAILED)
      << "command: \"" << rm_dot << "\", std::system exit(" << result << ")";
}

GraphTemplateImp::GraphType* GraphTemplateImp::get_boost_graph() {
  return graph_.get();
}

}  // namespace xir
