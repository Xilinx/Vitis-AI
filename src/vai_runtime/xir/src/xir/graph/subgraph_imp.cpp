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

#include <algorithm>

#include "UniLog/UniLog.hpp"
#include "xir/graph/subgraph_imp.hpp"
#include "xir/util/internal_util.hpp"
#include "xir/util/tool_function.hpp"

namespace xir {
SubgraphImp::SubgraphImp(GraphImp* graph, SubgraphImp* parent,
                         const std::set<Op*> ops)
    : graph_{graph}, parent_{parent}, ops_{ops} {
  attrs_ = Attrs::create();
  std::vector<Op*> ops_vec;
  std::for_each(ops.begin(), ops.end(), [&](Op* op) { ops_vec.push_back(op); });

  std::sort(ops_vec.begin(), ops_vec.end(), [](Op* op1, Op* op2) {
    return (op1->get_name() < op2->get_name());
  });
  if (ops.size()) {
    this->name_ = "subgraph_" + (*(ops_vec.begin()))->get_name();
  } else {
    this->name_ = "root";
  }
}

const std::string SubgraphImp::get_name() const { return this->name_; }

Subgraph* SubgraphImp::set_name(const std::string& subgraph_name) {
  this->name_ = subgraph_name;
  return this;
}

std::int32_t SubgraphImp::get_op_num() const { return ops_.size(); }

std::set<Op*> SubgraphImp::get_ops() { return ops_; }

const std::set<const Op*> SubgraphImp::get_ops() const {
  return internal::cast_to_const_set(ops_);
}

Op* SubgraphImp::get_tensor_producer(const Tensor* tensor) {
  return const_cast<Op*>(
      static_cast<const SubgraphImp&>(*this).get_tensor_producer(tensor));
}

const Op* SubgraphImp::get_tensor_producer(const Tensor* tensor) const {
  const Op* ret = nullptr;
  for (auto op : this->get_ops()) {
    if (op->get_output_tensor() == tensor) {
      ret = op;
      break;
    }
  }
  return ret;
}

std::set<Tensor*> SubgraphImp::get_input_tensors() {
  return internal::cast_from_const_set(
      static_cast<const SubgraphImp&>(*this).get_input_tensors());
}

const std::set<const Tensor*> SubgraphImp::get_input_tensors() const {
  auto ret = std::set<const Tensor*>();
  for (auto op : this->ops_) {
    auto input_ops = internal::vec_input_ops(op->get_input_ops());
    for (auto input_op : input_ops) {
      if (!(this->has_op(input_op))) {
        ret.insert(input_op->get_output_tensor());
      }
    }
  }
  return ret;
}

std::vector<Tensor*> SubgraphImp::get_sorted_input_tensors(
    const std::function<bool(Tensor*, Tensor*)>& compare) {
  return internal::cast_from_const_vector(
      static_cast<const SubgraphImp&>(*this).get_sorted_input_tensors(compare));
}

const std::vector<const Tensor*> SubgraphImp::get_sorted_input_tensors(
    const std::function<bool(Tensor*, Tensor*)>& compare) const {
  auto ret = std::vector<Tensor*>();
  auto ret_s = internal::cast_from_const_set(
      static_cast<const SubgraphImp&>(*this).get_input_tensors());
  ret.assign(ret_s.begin(), ret_s.end());
  sort(ret.begin(), ret.end(), compare);
  return internal::cast_to_const_vector(ret);
}

std::set<Tensor*> SubgraphImp::get_output_tensors() {
  return internal::cast_from_const_set(
      static_cast<const SubgraphImp&>(*this).get_output_tensors());
}

const std::set<const Tensor*> SubgraphImp::get_output_tensors() const {
  auto ret = std::set<const Tensor*>();
  for (auto op : this->ops_) {
    auto fanout_ops = op->get_fanout_ops();
    if (fanout_ops.size()) {
      for (auto fanout_op : fanout_ops) {
        if (!(this->has_op(fanout_op))) {
          ret.insert(op->get_output_tensor());
        }
      }
    } else {
      ret.insert(op->get_output_tensor());
    }
  }
  return ret;
}

std::vector<Tensor*> SubgraphImp::get_sorted_output_tensors(
    const std::function<bool(Tensor*, Tensor*)>& compare) {
  return internal::cast_from_const_vector(
      static_cast<const SubgraphImp&>(*this).get_sorted_output_tensors(
          compare));
}

const std::vector<const Tensor*> SubgraphImp::get_sorted_output_tensors(
    const std::function<bool(Tensor*, Tensor*)>& compare) const {
  auto ret = std::vector<Tensor*>();
  auto ret_s = internal::cast_from_const_set(
      static_cast<const SubgraphImp&>(*this).get_output_tensors());
  ret.assign(ret_s.begin(), ret_s.end());
  sort(ret.begin(), ret.end(), compare);
  return internal::cast_to_const_vector(ret);
}

std::int32_t SubgraphImp::count_op_(
    const std::set<std::string>& op_types) const {
  std::int32_t ret = 0;
  std::for_each(this->ops_.begin(), this->ops_.end(),
                [&ret, &op_types](Op* op) {
                  if (op_types.count(op->get_type())) {
                    ++ret;
                  }
                });
  return ret;
}

bool SubgraphImp::has_op(const std::string& op_name) const {
  auto ops = this->filter_op_by_name_(op_name);
  switch (ops.size()) {
    case 0:
      return false;
    case 1:
      return true;
    default: {
      std::stringstream ss;
      ss << "Find more than one op named as {" << op_name << "} in "
         << static_cast<const Subgraph*>(this)->to_string() << ": ";
      std::for_each(ops.begin(), --(ops.end()),
                    [&ss](const Op* op) { ss << op->to_string() << ", "; });
      ss << (*ops.rbegin())->to_string() << ".";
      UNI_LOG_FATAL(XIR_OP_NAME_CONFLICT) << ss.str();
    }
  }
}

bool SubgraphImp::has_op(const Op* op) const {
  return std::any_of(this->ops_.begin(), this->ops_.end(),
                     [&op](const Op* elem) -> bool { return (elem == op); });
}

Subgraph* SubgraphImp::find_op(const std::string& op_name) {
  return const_cast<Subgraph*>(
      static_cast<const SubgraphImp&>(*this).find_op(op_name));
}

const Subgraph* SubgraphImp::find_op(const std::string& op_name) const {
  Subgraph* ret = nullptr;
  if (this->has_op(op_name)) {
    for (auto& child : children_) {
      if (child->has_op(op_name)) {
        ret = child.get();
        break;
      }
    }
  }
  return ret;
}

Subgraph* SubgraphImp::find_op(const Op* op) {
  return const_cast<Subgraph*>(
      static_cast<const SubgraphImp&>(*this).find_op(op));
}

const Subgraph* SubgraphImp::find_op(const Op* op) const {
  Subgraph* ret = nullptr;
  for (auto& child : children_) {
    if (child->has_op(op)) {
      ret = child.get();
      break;
    }
  }
  return ret;
}

bool SubgraphImp::is_root() const { return parent_ == nullptr; }

bool SubgraphImp::is_leaf() const { return children_.size() == 0; }

std::int32_t SubgraphImp::get_depth() const {
  std::int32_t ret = 0;
  auto ptr = this;
  while (!ptr->is_root()) {
    ptr = static_cast<const SubgraphImp*>(ptr->get_parent());
    ++ret;
  }
  return ret;
}

Subgraph* SubgraphImp::get_root() {
  return const_cast<Subgraph*>(
      static_cast<const SubgraphImp&>(*this).get_root());
}

const Subgraph* SubgraphImp::get_root() const {
  const Subgraph* ret = this;
  while (!ret->is_root()) {
    ret = ret->get_parent();
  }
  return ret;
}

Subgraph* SubgraphImp::get_parent() { return parent_; }
const Subgraph* SubgraphImp::get_parent() const { return parent_; }

void SubgraphImp::create_children() {
  UNI_LOG_CHECK(this->is_leaf(), XIR_SUBGRAPH_CREATE_CHILDREN_FOR_NONLEAF)
      << "Cannot create children for non-leaf subgraph!";
  for (auto op : ops_) {
    children_.insert(
        std::make_unique<SubgraphImp>(this->graph_, this, std::set<Op*>{op}));
  }
  this->update_id_();
}

std::int32_t SubgraphImp::get_children_num() const { return children_.size(); }

std::set<Subgraph*> SubgraphImp::get_children() {
  return internal::cast_from_const_set(
      static_cast<const SubgraphImp&>(*this).get_children());
}

const std::set<const Subgraph*> SubgraphImp::get_children() const {
  auto ret = std::set<const Subgraph*>{};
  std::transform(children_.begin(), children_.end(),
                 std::inserter(ret, ret.end()),
                 [](const std::unique_ptr<SubgraphImp>& child) -> Subgraph* {
                   return child.get();
                 });
  return ret;
}

bool SubgraphImp::is_child(Subgraph* subgraph) const {
  auto target = static_cast<SubgraphImp*>(subgraph);
  return std::any_of(
      children_.begin(), children_.end(),
      [target](const std::unique_ptr<SubgraphImp>& child) -> bool {
        return target->get_id_() == child->get_id_();
      });
}

Subgraph* SubgraphImp::merge_children(std::set<Subgraph*> subgraph_list) {
  auto ops = std::set<Op*>{};
  for (auto subgraph : subgraph_list) {
    UNI_LOG_CHECK(this->is_child(subgraph),
                  XIR_SUBGRAPH_INVALID_MERGE_REQUEST_NONCHILD);
    UNI_LOG_CHECK(subgraph->is_leaf(),
                  XIR_SUBGRAPH_INVALID_MERGE_REQUEST_NONLEAF);
    auto target = static_cast<SubgraphImp*>(subgraph);
    auto iter_target = std::find_if(
        children_.begin(), children_.end(),
        [target](const std::unique_ptr<SubgraphImp>& child) -> bool {
          return target->get_id_() == child->get_id_();
        });
    auto target_ops = (*iter_target)->get_ops();
    std::move(target_ops.begin(), target_ops.end(),
              std::inserter(ops, ops.end()));
    children_.erase(iter_target);
  }
  auto result =
      children_.insert(std::make_unique<SubgraphImp>(this->graph_, this, ops));
  this->update_id_();
  return (*(result.first)).get();
}

Graph* SubgraphImp::get_graph() { return graph_; }

const Graph* SubgraphImp::get_graph() const { return graph_; }

Subgraph* SubgraphImp::get_subgraph(const std::string& name) {
  return const_cast<Subgraph*>(
      static_cast<const SubgraphImp&>(*this).get_subgraph(name));
}

const Subgraph* SubgraphImp::get_subgraph(const std::string& name) const {
  xir::Subgraph* ret = nullptr;
  for (auto& subg : children_) {
    if (subg->get_name() == name) return subg.get();
    ret = subg->get_subgraph(name);
    if (ret != nullptr) return ret;
  }
  return nullptr;
}

std::unique_ptr<Attrs> SubgraphImp::get_attrs() const {
  return Attrs::clone(attrs_.get());
}

bool SubgraphImp::has_attrs() const { return !(nullptr == attrs_); }

Subgraph* SubgraphImp::set_attrs(std::unique_ptr<Attrs> attrs) {
  attrs_ = std::move(attrs);
  return this;
}

bool SubgraphImp::has_attr(const std::string& key) const {
  if (nullptr == attrs_) {
    return false;
  }
  return attrs_->has_attr(key);
}

const xir::any SubgraphImp::get_attr(const std::string& key) const {
  return attrs_->get_attr(key);
}

Subgraph* SubgraphImp::set_attr(const std::string& key, const xir::any& value) {
  attrs_->set_attr(key, value);
  return this;
}

const std::uint32_t SubgraphImp::get_id_() const { return id_; }

void SubgraphImp::update_id_() {
  auto root = static_cast<SubgraphImp*>(this->get_root());
  root->update_id_helper_(0);
}

std::uint32_t SubgraphImp::update_id_helper_(std::uint32_t id) {
  id_ = id++;
  for (auto& child : children_) {
    id = child->update_id_helper_(id);
  }
  return id;
}

std::set<Op*> SubgraphImp::filter_op_by_name_(
    const std::string& op_name) const {
  std::set<Op*> ret{};
  std::for_each(this->ops_.begin(), this->ops_.end(), [&ret, &op_name](Op* op) {
    if (op->get_name() == op_name) {
      ret.insert(op);
    }
  });
  return ret;
}

void SubgraphImp::add_op(Op* op) {
  if (!is_leaf()) {
    this->children_.insert(
        std::make_unique<SubgraphImp>(this->graph_, this, std::set<Op*>{op}));
    this->update_id_();
  }
  add_op_helper_(op);
}

void SubgraphImp::add_op_helper_(Op* op) {
  ops_.insert(op);
  if (!is_root()) {
    parent_->add_op_helper_(op);
  }
}

void SubgraphImp::remove_op(Op* op) {
  UNI_LOG_CHECK(is_root(), XIR_REMOVE_OP_FAIL)
      << "Subgraph has to be root if you want to remove an op from it";
  UNI_LOG_CHECK(has_op(op), XIR_REMOVE_OP_FAIL)
      << "Subgraph doesn't have the op you want to remove";
  remove_op_helper_(op);
}

void SubgraphImp::remove_op_helper_(Op* op) {
  ops_.erase(op);
  if (!is_leaf()) {
    for (auto iter = children_.begin(); iter != children_.end(); ++iter) {
      if ((*iter)->has_op(op)) {
        if ((*iter)->get_op_num() == 1) {
          children_.erase(iter);
        } else {
          (*iter)->remove_op_helper_(op);
        }
        break;
      }
    }
  }
}

struct CycleDetector : public boost::dfs_visitor<> {
  CycleDetector(bool& has_cycle) : has_cycle_{has_cycle} {}

  template <class Edge, class Graph>
  void back_edge(Edge e, Graph& g) {
    has_cycle_ = true;
    auto op_source = g[boost::source(e, g)]->get_name();
    auto op_target = g[boost::target(e, g)]->get_name();
    UNI_LOG_DEBUG_WARNING << "Back edge from " << op_source << " to "
                          << op_target;
  }

 private:
  bool& has_cycle_;
};

bool EdgePredicate::operator()(const GraphImp::EdgeD& ed) const {
  auto vd_source = boost::source(ed, *(graph_->get_boost_graph()));
  auto vd_target = boost::target(ed, *(graph_->get_boost_graph()));
  auto b_source = std::any_of(ops_.begin(), ops_.end(), [vd_source](Op* op) {
    return static_cast<OpImp*>(op)->vd_ == vd_source;
  });
  auto b_target = std::any_of(ops_.begin(), ops_.end(), [vd_target](Op* op) {
    return static_cast<OpImp*>(op)->vd_ == vd_target;
  });
  return b_source && b_target;
}

bool VertexPredicate::operator()(const GraphImp::VertexD& vd) const {
  return std::any_of(ops_.begin(), ops_.end(), [vd](Op* op) {
    return static_cast<OpImp*>(op)->vd_ == vd;
  });
}

std::unique_ptr<SubgraphImp::FilteredGraphType>
SubgraphImp::get_filtered_graph() const {
  return std::make_unique<FilteredGraphType>(*(graph_->get_boost_graph()),
                                             EdgePredicate{graph_, ops_},
                                             VertexPredicate{ops_});
}

std::vector<Op*> SubgraphImp::topological_sort() {
  return internal::cast_from_const_vector(
      static_cast<const SubgraphImp&>(*this).topological_sort());
}

const std::vector<const Op*> SubgraphImp::topological_sort() const {
  auto ret_vd = std::vector<GraphImp::VertexD>{};
  auto filtered_graph = get_filtered_graph();
  boost::topological_sort(*filtered_graph, std::back_inserter(ret_vd));
  auto ret = std::vector<const Op*>{ret_vd.size()};
  std::transform(ret_vd.begin(), ret_vd.end(), ret.rbegin(),
                 [&filtered_graph](const GraphImp::VertexD& vd) {
                   return (*filtered_graph)[vd].get();
                 });
  return ret;
}

std::vector<Subgraph*> SubgraphImp::children_topological_sort() {
  return internal::cast_from_const_vector(
      static_cast<const SubgraphImp&>(*this).children_topological_sort());
}

const std::vector<const Subgraph*> SubgraphImp::children_topological_sort()
    const {
  using HelperGraph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                            Subgraph*, boost::no_property, boost::no_property>;
  using HelperVertexD = boost::graph_traits<HelperGraph>::vertex_descriptor;

  auto map_subgraph_vd = std::map<const Subgraph*, HelperVertexD>{};
  auto graph = HelperGraph{0};
  std::vector<SubgraphImp*> children_vec;

  for (auto& child : children_) {
    children_vec.push_back(child.get());
    int vec_num = children_vec.size();
    int preidx = vec_num - 2;
    auto current = child.get();
    while (preidx >= 0 &&
           children_vec[preidx]->get_name() > current->get_name()) {
      children_vec[preidx + 1] = children_vec[preidx];
      preidx--;
    }
    children_vec[preidx + 1] = current;
  }

  for (auto& child : children_vec) {
    auto vd = boost::add_vertex(graph);
    graph[vd] = child;
    map_subgraph_vd.emplace(static_cast<Subgraph*>(child), vd);
  }

  for (auto& child : children_vec) {
    std::vector<Op*> input_vec(0), output_vec(0);
    for (auto op : child->get_ops()) {
      auto input_ops = internal::vec_input_ops(op->get_input_ops());
      for (auto input_op : input_ops) {
        if (this->has_op(input_op) &&
            static_cast<Subgraph*>(child) != this->find_op(input_op)) {
          input_vec.push_back(input_op);
        }
      }
      auto fanout_ops = op->get_fanout_ops();
      for (auto fanout_op : fanout_ops) {
        if (this->has_op(fanout_op) &&
            static_cast<Subgraph*>(child) != this->find_op(fanout_op)) {
          output_vec.push_back(fanout_op);
        }
      }
    }

    auto op_compare = [](Op* op1, Op* op2) {
      return (op1->get_name() < op2->get_name());
    };
    std::sort(input_vec.begin(), input_vec.end(), op_compare);
    std::sort(output_vec.begin(), output_vec.end(), op_compare);
    for (auto op : input_vec) {
      boost::add_edge(map_subgraph_vd[this->find_op(op)],
                      map_subgraph_vd[static_cast<Subgraph*>(child)], graph);
    }
    for (auto op : output_vec) {
      boost::add_edge(map_subgraph_vd[static_cast<Subgraph*>(child)],
                      map_subgraph_vd[this->find_op(op)], graph);
    }
  }

  auto has_cycle = false;
  auto vis = CycleDetector{has_cycle};
  boost::depth_first_search(graph, boost::visitor(vis));
  UNI_LOG_CHECK(!has_cycle, XIR_SUBGRAPH_HAS_CYCLE);
  auto ret_vd = std::vector<HelperVertexD>{};
  boost::topological_sort(graph, std::back_inserter(ret_vd));
  auto ret = std::vector<const Subgraph*>{ret_vd.size()};
  std::transform(ret_vd.begin(), ret_vd.end(), ret.rbegin(),
                 [graph](const HelperVertexD& vd) { return graph[vd]; });
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

static void create_one_subgraph(boost::subgraph<BOOSTSubGraph>& BOOSTsub,
                                const xir::Subgraph* cur,
                                std::map<std::string, int>& op_idx, int& idx) {
  auto subops = cur->get_ops();
  std::for_each(subops.begin(), subops.end(), [&](auto op) {
    boost::add_vertex(op_idx[op->get_name()], BOOSTsub);
  });
  boost::get_property(BOOSTsub, boost::graph_vertex_attribute)["style"] =
      "rounded";
  boost::get_property(BOOSTsub, boost::graph_vertex_attribute)["shape"] =
      "box";  // style and shape here are used to create rounded-angle rectangle
  boost::get_property(BOOSTsub, boost::graph_vertex_attribute)["width"] =
      "x";  // width = x means that the width of rectangle is determined by
            // label
  boost::get_property(BOOSTsub, boost::graph_graph_attribute)["label"] =
      cur->get_name();  // label is subgraph name
  boost::get_property(BOOSTsub, boost::graph_name) =
      "clustersubgraph" +
      std::to_string(
          idx);  // we must add "cluster" in the front of graph name for
  // subgraph, otherwise, subgraph will not be drawn.
  if (cur->has_attr("device")) {
    if (cur->get_attr<std::string>("device") == "CPU")
      get_property(BOOSTsub, boost::graph_vertex_attribute)["color"] = "red";
    else if (cur->get_attr<std::string>("device") == "DPU")
      get_property(BOOSTsub, boost::graph_vertex_attribute)["color"] = "blue";
    else if (cur->get_attr<std::string>("device") == "USER")
      get_property(BOOSTsub, boost::graph_vertex_attribute)["color"] = "green";
  }
  idx++;
}

static void create_subgraphs(boost::subgraph<BOOSTSubGraph>& main,
                             const std::set<const xir::Subgraph*> cur,
                             std::map<std::string, int>& op_idx, int& idx) {
  std::for_each(cur.begin(), cur.end(), [&](auto sub) {
    if (sub->get_children_num() > 0) {
      boost::subgraph<BOOSTSubGraph>& submain = main.create_subgraph();
      create_one_subgraph(submain, sub, op_idx, idx);
      create_subgraphs(submain, sub->get_children(), op_idx, idx);
    } else {
      boost::subgraph<BOOSTSubGraph>& submain = main.create_subgraph();
      create_one_subgraph(submain, sub, op_idx, idx);
    }
  });
}

void SubgraphImp::save_to_dot(const std::string& file_path) const {
  auto ori = this;
  auto node_size = ori->get_op_num();
  boost::subgraph<BOOSTSubGraph> main(node_size);
  auto idx = 0;
  std::map<std::string, int> op_idx;
  std::map<int, const xir::Op*> idx_op;
  auto ops = ori->get_ops();
  std::for_each(ops.begin(), ops.end(), [&](auto op) {
    op_idx[op->get_name()] = idx;
    idx_op[idx] = op;
    idx++;
  });
  for (auto op : ops) {
    auto input_ops = internal::vec_input_ops(op->get_input_ops());
    for (auto in : input_ops) {
      add_edge(op_idx[in->get_name()], op_idx[op->get_name()], main);
    }
  }
  for (auto vd : boost::make_iterator_range(boost::vertices(main))) {
    std::string node_label =
        fold_str("Name: " + idx_op[vd]->get_name(), 50) +  //
        "\nType: " + idx_op[vd]->get_type() +              //
        fold_str("\nTensor: " + idx_op[vd]->get_output_tensor()->get_name(),
                 50) +  //
        "\nShape: " +
        xir::to_string(idx_op[vd]->get_output_tensor()->get_shape());
    put(get(boost::vertex_attribute, main), vd,
        GraphvizAttributes{{"label", node_label}});
  }
  boost::get_property(main, boost::graph_name) = "G0";
  boost::get_property(main, boost::graph_vertex_attribute)["style"] = "rounded";
  boost::get_property(main, boost::graph_vertex_attribute)["shape"] = "box";
  idx = 0;
  create_subgraphs(main, ori->get_children(), op_idx, idx);
  auto dir = file_path;
  boost::write_graphviz(dir, main);
}

const std::string SubgraphImp::to_string(
    const std::string& delimiter,     //
    const std::string& left_bracket,  //
    const std::string& right_bracket) const {
  std::ostringstream out;
  out << "xir::Subgraph" << left_bracket                           //
      << "name = " << this->get_name() << delimiter                //
      << " child_num = " << this->get_children_num() << delimiter  //
      << " ops_num = " << this->get_op_num()                       //
      << right_bracket;
  return out.str();
}

struct IsoVertexEquivalent {
  IsoVertexEquivalent(GraphTemplateImp* graph_small,
                      const SubgraphImp::FilteredGraphType& graph_large)
      : small_{*(graph_small->get_boost_graph())}, large_{graph_large} {}
  bool operator()(const GraphTemplateImp::VertexD vd_small,
                  const GraphImp::VertexD vd_large) const {
    auto template_types = small_[vd_small]->get_types();
    auto type = large_[vd_large]->get_type();
    return template_types.count(type) > 0;
  }

 private:
  const GraphTemplateImp::GraphType& small_;
  const SubgraphImp::FilteredGraphType& large_;
};

struct IsoEdgeEquivalent {
  IsoEdgeEquivalent(GraphTemplateImp* graph_small,
                    const SubgraphImp::FilteredGraphType& graph_large)
      : small_{*(graph_small->get_boost_graph())}, large_{graph_large} {}
  bool operator()(const GraphTemplateImp::EdgeD ed_small,
                  const GraphImp::EdgeD ed_large) const {
    auto ret = true;
    auto argument_name = small_[ed_small];
    if (argument_name != "") {
      auto op_source =
          static_cast<Op*>(large_[boost::source(ed_large, large_)].get());
      auto op_target =
          static_cast<Op*>(large_[boost::target(ed_large, large_)].get());
      auto ops = op_target->get_input_ops(argument_name);
      ret = std::count(ops.begin(), ops.end(), op_source) > 0;
    }
    return ret;
  }

 private:
  const GraphTemplateImp::GraphType& small_;
  const SubgraphImp::FilteredGraphType& large_;
};

struct IsoCallback {
  using result_type = std::vector<std::map<OpTemplate*, Op*>>;

  IsoCallback(GraphTemplateImp* graph_small,
              const SubgraphImp::FilteredGraphType& graph_large,
              std::shared_ptr<result_type> result)
      : small_{*(graph_small->get_boost_graph())},
        large_{graph_large},
        graph_filter_{graph_small->get_filter()},
        result_{result} {}

  template <typename CorrespondenceMap1To2, typename CorrespondenceMap2To1>
  bool operator()(CorrespondenceMap1To2 f, CorrespondenceMap2To1) {
    auto new_map = std::map<OpTemplate*, Op*>{};
    for (auto vd : boost::make_iterator_range(boost::vertices(small_))) {
      auto& filter = small_[vd].get()->get_filter();
      auto op = large_[boost::get(f, vd)].get();
      if (!filter(op)) return true;
      new_map.emplace(small_[vd].get(), static_cast<Op*>(op));
    }
    if (!graph_filter_(new_map)) return true;
    result_->push_back(new_map);
    return true;
  }

 private:
  const GraphTemplateImp::GraphType& small_;
  const SubgraphImp::FilteredGraphType& large_;
  const std::function<bool(std::map<OpTemplate*, Op*>)> graph_filter_;
  std::shared_ptr<result_type> result_;
};

std::vector<std::map<OpTemplate*, Op*>> SubgraphImp::isomorphism(
    GraphTemplate* graph_template) {
  auto graph_large = get_filtered_graph();
  auto graph_small = static_cast<GraphTemplateImp*>(graph_template);
  auto result = std::make_shared<std::vector<std::map<OpTemplate*, Op*>>>();

  GraphTemplateImp::GraphType& small_graph = *(graph_small->get_boost_graph());
  auto vertex_order_by_mult = boost::vertex_order_by_mult(small_graph);
  std::sort(vertex_order_by_mult.begin(), vertex_order_by_mult.end(),
            [this, &small_graph](GraphTemplateImp::VertexD vertex_x,
                                 GraphTemplateImp::VertexD vertex_y) {
              return this->count_op_(small_graph[vertex_x]->get_types()) <
                     this->count_op_(small_graph[vertex_y]->get_types());
            });

  boost::vf2_subgraph_iso(
      *(graph_small->get_boost_graph()), *graph_large,
      IsoCallback{graph_small, *graph_large, result}, vertex_order_by_mult,
      boost::edges_equivalent(IsoEdgeEquivalent{graph_small, *graph_large})
          .vertices_equivalent(IsoVertexEquivalent{graph_small, *graph_large}));

  return *result;
}
}  // namespace xir
