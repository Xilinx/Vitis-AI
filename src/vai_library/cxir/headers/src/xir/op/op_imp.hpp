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

#include "UniLog/UniLog.hpp"
#include "xir/graph/graph_imp.hpp"
#include "xir/op/op.hpp"
#include "xir/op/op_def.hpp"
#include "xir/op/op_def_factory_imp.hpp"
#include "xir/tensor/tensor_imp.hpp"

namespace xir {

struct DataType;

class OpImp : public Op {
 public:
  OpImp(GraphImp::VertexD vd, const std::string& name, const std::string& type,
        std::unique_ptr<Attrs> attrs,
        const std::map<std::string, std::vector<Op*>>& input_ops,
        std::unique_ptr<Tensor> output_tensor, GraphImp* graph,
        const DataType& output_data_type);
  OpImp(OpImp&&) = default;
  virtual ~OpImp() = default;
  const std::string get_name() const override;
  const std::string get_type() const override;

  // input & output op
  int get_input_num() const override;
  int get_input_num(std::string arg_name) const override;

  std::map<std::string, std::vector<Op*>> get_input_ops() override;
  const std::map<std::string, std::vector<const Op*>> get_input_ops()
      const override;
  std::vector<Op*> get_input_ops(std::string arg_name) override;
  const std::vector<const Op*> get_input_ops(
      std::string arg_name) const override;

  Op* get_input_op(std::string arg_name, int idx = 0) override;
  const Op* get_input_op(std::string arg_name, int idx = 0) const override;

  Op* set_input_ops(std::string arg_name, std::vector<Op*> op_list) override;
  void replace_input_op(Op* op_old, Op* op_new) override;

  int get_fanout_num() const override;
  std::vector<Op*> get_fanout_ops() override;
  const std::vector<const Op*> get_fanout_ops() const override;

  // input & output tensor
  std::vector<Tensor*> get_input_tensors() override;
  const std::vector<const Tensor*> get_input_tensors() const override;
  std::vector<Tensor*> get_input_tensors(std::string arg_name) override;
  const std::vector<const Tensor*> get_input_tensors(
      std::string arg_name) const override;
  Tensor* get_input_tensor(std::string arg_name, int idx = 0) override;
  const Tensor* get_input_tensor(std::string arg_name,
                                 int idx = 0) const override;
  Tensor* get_output_tensor() override;
  const Tensor* get_output_tensor() const override;
  void replace_output_tensor(std::unique_ptr<Tensor> tensor_new) override;

  Graph* get_graph() override;
  const Graph* get_graph() const override;

  // op properties
  std::unique_ptr<Attrs> get_attrs() const override;
  bool has_attrs() const override;
  Op* set_attrs(std::unique_ptr<Attrs> attrs) override;
  bool has_attr(const std::string& key) const override;
  const xir::any get_attr(const std::string& key) const override;
  Op* set_attr(const std::string& key, const xir::any& value) override;

  // infer shape
  void shape_infer() override;

  const OpDef* get_opdef() const override;

  const std::string to_string(
      const std::string& delimiter = ",",     //
      const std::string& left_bracket = "{",  //
      const std::string& right_bracket = "}") const override;

 public:
  const GraphImp::VertexD vd_;
  bool to_be_removed_;

 private:
  std::unique_ptr<Tensor> create_output_tensor_(
      const DataType& output_data_type);

 private:
  const std::string name_;
  const std::string type_;
  const OpDef* def_;
  std::unique_ptr<Attrs> attrs_;
  std::map<std::string, std::vector<OpImp*>> input_ops_;
  std::unique_ptr<Tensor> output_tensor_;
  friend class c_api;
  GraphImp* graph_;
};

Op* op_up_cast(OpImp* const& ptr);

const OpImp* op_down_cast_const(const Op* ptr);
OpImp* op_down_cast(Op* ptr);

}  // namespace xir
