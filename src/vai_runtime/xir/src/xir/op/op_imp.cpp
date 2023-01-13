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

#include "xir/op/op_imp.hpp"
#include "xir/op/shape_inference.hpp"
#include "xir/util/internal_util.hpp"
#include "xir/util/tool_function.hpp"

namespace xir {

class TensorImp;

static void op_arg_occur_check(const OpArgDef& arg, std::uint32_t num) {
  if (arg.occur_type == OpArgDef::REQUIRED) {
    UNI_LOG_CHECK(num == 1, XIR_INVALID_ARG_OCCUR)
        << "Arg " << arg.name << " has type REQUIRED, but try set " << num
        << " elements";
  } else if (arg.occur_type == OpArgDef::OPTIONAL) {
    UNI_LOG_CHECK(num < 2, XIR_INVALID_ARG_OCCUR)
        << "Arg " << arg.name << " has type OPTIONAL, but try set " << num
        << " elements";
  } else if (arg.occur_type == OpArgDef::REPEATED) {
    ;
  } else if (arg.occur_type == OpArgDef::REQUIRED_AND_REPEATED) {
    UNI_LOG_CHECK(num > 0, XIR_INVALID_ARG_OCCUR)
        << "Arg " << arg.name << " has type REQUIRED_AND_REPEATED, but try set "
        << num << " elements";
  } else {
    UNI_LOG_FATAL(XIR_XIR_UNDEFINED_OPERATION);
  }
}

OpImp::OpImp(GraphImp::VertexD vd, const std::string& name,
             const std::string& type, std::unique_ptr<Attrs> attrs,
             const std::map<std::string, std::vector<Op*>>& input_ops,
             std::unique_ptr<Tensor> output_tensor, GraphImp* graph,
             const DataType& output_data_type)
    : vd_{vd},                //
      to_be_removed_{false},  //
      name_{name},            //
      type_{type},            //
      graph_{graph} {
  auto build_in_ops = op_def_factory()->get_registered_ops();
  if (std::find(build_in_ops.begin(), build_in_ops.end(), type) ==
      build_in_ops.end()) {
    xir::register_customized_operator_definition(name, type);
  }
  def_ = op_def_factory()->create(type);
  set_attrs(std::move(attrs));

  std::for_each(def_->input_args().begin(), def_->input_args().end(),
                [&input_ops](const OpArgDef& arg) {
                  auto num = input_ops.count(arg.name) > 0
                                 ? input_ops.at(arg.name).size()
                                 : 0;
                  op_arg_occur_check(arg, num);
                });
  for (auto const& pair : input_ops) {
    set_input_ops(pair.first, pair.second);
  }
  output_tensor_ = this->create_output_tensor_(output_data_type);
  this->shape_infer();
  auto constraints = this->def_->constraints();
  for (auto check : constraints) check(op_up_cast(this));
}

const std::string OpImp::get_name() const { return name_; }
const std::string OpImp::get_type() const { return type_; }

int OpImp::get_input_num() const {
  return boost::in_degree(vd_, *graph_->get_boost_graph());
}

int OpImp::get_input_num(std::string arg_name) const {
  return input_ops_.count(arg_name) > 0 ? input_ops_.at(arg_name).size() : 0;
}

std::vector<Op*> OpImp::get_input_ops(std::string arg_name) {
  return internal::cast_from_const_vector(
      static_cast<const OpImp&>(*this).get_input_ops(arg_name));
}

const std::vector<const Op*> OpImp::get_input_ops(std::string arg_name) const {
  auto ret = std::vector<const Op*>{};
  if (input_ops_.count(arg_name) > 0) {
    std::transform(input_ops_.at(arg_name).begin(),
                   input_ops_.at(arg_name).end(), std::back_inserter(ret),
                   op_up_cast);
  }
  return ret;
}

Op* OpImp::get_input_op(std::string arg_name, int idx) {
  return const_cast<Op*>(
      static_cast<const OpImp&>(*this).get_input_op(arg_name, idx));
}

const Op* OpImp::get_input_op(std::string arg_name, int idx) const {
  UNI_LOG_CHECK(input_ops_.count(arg_name) > 0, XIR_UNDEFINED_INPUT_ARG)
      << arg_name;
  UNI_LOG_CHECK(
      idx >= 0 && idx < static_cast<int>(input_ops_.at(arg_name).size()),
      XIR_OUT_OF_RANGE)
      << idx << " out of range. num of " << arg_name << " is "
      << input_ops_.at(arg_name).size();
  return op_up_cast(input_ops_.at(arg_name)[idx]);
}

std::map<std::string, std::vector<Op*>> OpImp::get_input_ops() {
  std::map<std::string, std::vector<Op*>> ret;
  for (auto it : this->input_ops_) {
    std::vector<Op*> ops;
    ops.reserve(it.second.size());
    for (auto& opimp : it.second) {
      ops.push_back(op_up_cast(opimp));
    }
    ret.emplace(it.first, ops);
  }
  return ret;
}

const std::map<std::string, std::vector<const Op*>> OpImp::get_input_ops()
    const {
  std::map<std::string, std::vector<const Op*>> ret;
  for (auto it : this->input_ops_) {
    std::vector<Op*> ops;
    ops.reserve(it.second.size());
    for (auto& opimp : it.second) {
      ops.push_back(op_up_cast(opimp));
    }
    ret.emplace(it.first, internal::cast_to_const_vector(ops));
  }
  return ret;
}

Op* OpImp::set_input_ops(std::string arg_name, std::vector<Op*> op_list) {
  auto iter_arg = std::find_if(
      def_->input_args().begin(), def_->input_args().end(),
      [arg_name](const OpArgDef& arg) -> bool { return arg_name == arg.name; });
  UNI_LOG_CHECK(iter_arg != def_->input_args().end(), XIR_UNREGISTERED_ARG)
      << arg_name;
  op_arg_occur_check(*iter_arg, op_list.size());

  for (auto input_op : input_ops_[arg_name]) {
    auto ed = boost::edge(input_op->vd_, vd_, *graph_->get_boost_graph()).first;
    boost::remove_edge(ed, *graph_->get_boost_graph());
  }
  input_ops_[arg_name].clear();
  std::transform(op_list.begin(), op_list.end(),
                 std::back_inserter(input_ops_[arg_name]), op_down_cast);
  for (auto input_op : input_ops_[arg_name]) {
    const auto op_range = vertices(*graph_->get_boost_graph());
    UNI_LOG_CHECK(std::find(op_range.first, op_range.second, input_op->vd_) !=
                      op_range.second,
                  XIR_UNDEFINED_OP)
        << input_op->get_name();
    boost::add_edge(input_op->vd_, vd_, *graph_->get_boost_graph());
  }
  return this;
}

void OpImp::replace_input_op(Op* op_old, Op* op_new) {
  const auto op_range = vertices(*graph_->get_boost_graph());
  UNI_LOG_CHECK(std::find(op_range.first, op_range.second,
                          static_cast<OpImp*>(op_new)->vd_) != op_range.second,
                XIR_UNDEFINED_OP)
      << op_new->to_string() << " is not in the current graph.";
  const auto& ori_shape = op_old->get_output_tensor()->get_shape();
  const auto& new_shape = op_new->get_output_tensor()->get_shape();
  UNI_LOG_CHECK(ori_shape == new_shape, XIR_SHAPE_UNMATCH)
      << "Using " << op_new->to_string() << " to replace " << this->to_string()
      << "'s input " << op_old->to_string()
      << " failed. Size is unmatching, the original input shape is "
      << xir::to_string(ori_shape) << " but the new input shape is "
      << xir::to_string(new_shape) << ".";
  for (auto& arg_ops : input_ops_) {
    for (auto iter = arg_ops.second.begin(); iter != arg_ops.second.end();
         ++iter) {
      if (op_old == static_cast<Op*>(*iter)) {
        auto ed =
            boost::edge((*iter)->vd_, vd_, *graph_->get_boost_graph()).first;
        boost::remove_edge(ed, *graph_->get_boost_graph());
        iter = arg_ops.second.erase(iter);
        iter = arg_ops.second.insert(iter, static_cast<OpImp*>(op_new));
        boost::add_edge((*iter)->vd_, vd_, *graph_->get_boost_graph());
      }
    }
  }
}

int OpImp::get_fanout_num() const {
  return boost::out_degree(vd_, *graph_->get_boost_graph());
}

std::vector<Op*> OpImp::get_fanout_ops() {
  return internal::cast_from_const_vector(
      static_cast<const OpImp&>(*this).get_fanout_ops());
}

const std::vector<const Op*> OpImp::get_fanout_ops() const {
  auto ret = std::vector<const Op*>{};
  for (auto ed : boost::make_iterator_range(
           boost::out_edges(vd_, *graph_->get_boost_graph()))) {
    ret.push_back(op_up_cast(
        (*graph_->get_boost_graph())[boost::target(ed, *graph_->get_boost_graph())].get()));
  }
  // drop duplicates 
  auto itr = ret.begin();
  std::set<const Op*> s;
  for (auto curr = ret.begin(); curr != ret.end(); ++curr) {
    if (s.insert(*curr).second) {
      *itr++ = *curr;
    }
  }
  ret.erase(itr, ret.end());
  return ret;
}

// input & output tensor
std::vector<Tensor*> OpImp::get_input_tensors() {
  return internal::cast_from_const_vector(
      static_cast<const OpImp&>(*this).get_input_tensors());
}

const std::vector<const Tensor*> OpImp::get_input_tensors() const {
  auto ret = std::vector<const Tensor*>{};
  auto ops = internal::vec_input_ops(get_input_ops());
  std::transform(
      ops.begin(), ops.end(), std::back_inserter(ret),
      [this](const Op* op) -> const Tensor* {
        return (*graph_->get_boost_graph())[op_down_cast_const(op)->vd_]
            ->get_output_tensor();
      });
  return ret;
}

std::vector<Tensor*> OpImp::get_input_tensors(std::string arg_name) {
  return internal::cast_from_const_vector(
      static_cast<const OpImp&>(*this).get_input_tensors(arg_name));
}

const std::vector<const Tensor*> OpImp::get_input_tensors(
    std::string arg_name) const {
  auto ret = std::vector<const Tensor*>{};
  auto ops = get_input_ops(arg_name);
  std::transform(
      ops.begin(), ops.end(), std::back_inserter(ret),
      [this](const Op* op) -> const Tensor* {
        return (*graph_->get_boost_graph())[op_down_cast_const(op)->vd_]
            ->get_output_tensor();
      });
  return ret;
}

Tensor* OpImp::get_input_tensor(std::string arg_name, int idx) {
  return const_cast<Tensor*>(
      static_cast<const OpImp&>(*this).get_input_tensor(arg_name, idx));
}

const Tensor* OpImp::get_input_tensor(std::string arg_name, int idx) const {
  return (*graph_->get_boost_graph())
      [op_down_cast_const(get_input_op(arg_name, idx))->vd_]
          ->get_output_tensor();
}

Tensor* OpImp::get_output_tensor() { return output_tensor_.get(); }
const Tensor* OpImp::get_output_tensor() const { return output_tensor_.get(); }

void OpImp::replace_output_tensor(std::unique_ptr<Tensor> tensor_new) {
  static_cast<TensorImp*>(tensor_new.get())->producer_ = static_cast<Op*>(this);
  output_tensor_ = std::move(tensor_new);
}

Graph* OpImp::get_graph() { return graph_; }
const Graph* OpImp::get_graph() const { return graph_; }

std::unique_ptr<Attrs> OpImp::get_attrs() const {
  if (nullptr == attrs_) {
    return Attrs::create();
  } else {
    return Attrs::clone(attrs_.get());
  }
}

bool OpImp::has_attrs() const { return !(attrs_ == nullptr); }

Op* OpImp::set_attrs(std::unique_ptr<Attrs> attrs) {
  std::for_each(def_->attrs().begin(), def_->attrs().end(),
                [&attrs, this](const AttrDef& attr) {
                  UNI_LOG_CHECK(attr.occur_type != AttrDef::REQUIRED ||
                                    attrs->has_attr(attr.name),
                                XIR_INVALID_ATTR_OCCUR)
                      << this->to_string() << " : Attr " << attr.name
                      << " has type REQUIRED, but not set.";
                  if ((!attrs->has_attr(attr.name)) &&
                      (attr.occur_type == AttrDef::OPTIONAL)) {
                    attrs->set_attr(attr.name, attr.default_value);
                  }
                });
  attrs_ = std::move(attrs);
  return this;
}

bool OpImp::has_attr(const std::string& key) const {
  if (nullptr == attrs_) {
    return false;
  }
  return attrs_->has_attr(key);
}

const xir::any OpImp::get_attr(const std::string& key) const {
  return attrs_->get_attr(key);
}

Op* OpImp::set_attr(const std::string& key, const xir::any& value) {
  if (nullptr == attrs_) {
    attrs_ = Attrs::create();
  }
  attrs_->set_attr(key, value);
  return this;
}

void OpImp::shape_infer() { def_->shape_infer()(op_up_cast(this)); }

const OpDef* OpImp::get_opdef() const { return def_; }

const std::string OpImp::to_string(const std::string& delimiter,     //
                                   const std::string& left_bracket,  //
                                   const std::string& right_bracket) const {
  std::ostringstream out;
  out << "xir::Op" << left_bracket                   //
      << "name = " << this->get_name() << delimiter  //
      << " type = " << this->get_type()              //
      << right_bracket;
  return out.str();
}

std::unique_ptr<Tensor> OpImp::create_output_tensor_(
    const DataType& output_data_type) {
  std::vector<std::int32_t> output_tensor_shape{1};
  DataType data_type;
  if ((!this->input_ops_.size()) || (!this->get_input_num("input"))) {
    UNI_LOG_CHECK(
        this->attrs_->has_attr("shape") && this->attrs_->has_attr("data_type"),
        XIR_INVALID_ATTR_OCCUR);
    output_tensor_shape =
        this->attrs_->get_attr<std::vector<std::int32_t>>("shape");
    data_type = DataType{this->attrs_->get_attr<std::string>("data_type")};
  } else {
    // output tensor shape will be deduced by the shape infer
    auto input_ref = this->get_input_op("input", 0)->get_output_tensor();
    data_type = input_ref->get_data_type();
  }
  if (output_data_type.valid()) {
    data_type = output_data_type;
  }
  auto tensorimp = new TensorImp(this->name_, output_tensor_shape, data_type);
  tensorimp->producer_ = this;
  auto ret = std::unique_ptr<Tensor>{static_cast<Tensor*>(tensorimp)};
  return ret;
}

Op* op_up_cast(OpImp* const& ptr) { return static_cast<Op*>(ptr); }

const OpImp* op_down_cast_const(const Op* ptr) {
  return static_cast<const OpImp*>(ptr);
}

OpImp* op_down_cast(Op* ptr) { return static_cast<OpImp*>(ptr); }

}  // namespace xir
