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

#include <memory>
#include <vector>

#include "xir/attrs/attrs.hpp"
#include "xir/tensor/tensor.hpp"

namespace xir {

class OpDef;
class Graph;

/**
 * @brief  XIR OP interface
 *
 * This class defines the basic XIR OP Interface.
 */
class Op {
 public:
  /**
   * @brief Get name of the OP
   *
   * @return OP name
   */
  virtual const std::string get_name() const = 0;

  /**
   * @brief Get type of the OP
   *
   * @return OP type
   */
  virtual const std::string get_type() const = 0;

  /**
   * @brief Get total input number
   *
   * @return input number
   */
  virtual int get_input_num() const = 0;

  /**
   * @brief Get input number with specific arg_name
   *
   * @param arg_name Specific argument name
   *
   * @return input number
   */
  virtual int get_input_num(std::string arg_name) const = 0;

  /**
   * @brief Get all input OPs
   *
   * @return A map of input OPs
   */
  virtual std::map<std::string, std::vector<Op*>> get_input_ops() = 0;

  /**
   * @brief Get all input OPs
   *
   * @return A map of input OPs
   */
  virtual const std::map<std::string, std::vector<const Op*>> get_input_ops()
      const = 0;

  /**
   * @brief Get all input OPs with specific arg_name
   *
   * @param arg_name Specific argument name
   *
   * @return vector of input OPs, the order is guaranteed to be same as it was
   * set
   */
  virtual std::vector<Op*> get_input_ops(std::string arg_name) = 0;

  /**
   * @brief Get all input OPs with specific arg_name
   *
   * @param arg_name Specific argument name
   *
   * @return vector of input OPs, the order is guaranteed to be same as it was
   * set
   */
  virtual const std::vector<const Op*> get_input_ops(
      std::string arg_name) const = 0;

  /**
   * @brief Get input OP with specific arg_name and index
   *
   * @param arg_name Specific argument name
   *
   * @param idx Index of the input OP. The default value of idx is 0
   *
   * @return input OP
   */
  virtual Op* get_input_op(std::string arg_name, int idx = 0) = 0;

  /**
   * @brief Get input OP with specific arg_name and index
   *
   * @param arg_name Specific argument name
   *
   * @param idx Index of the input OP. The default value of idx is 0
   *
   * @return input OP
   */
  virtual const Op* get_input_op(std::string arg_name, int idx = 0) const = 0;

  /**
   * @brief Set all input OPs with specific arg_name
   *
   * @param arg_name Specific argument name
   *
   * @param op_list the input op list. The order will be guaranteed
   *
   * @return vector of input OPs
   */
  virtual Op* set_input_ops(std::string arg_name, std::vector<Op*> op_list) = 0;

  /**
   * @brief Replace an op's specific input op.
   *
   * @param op_old A raw pointer to the input op to be replaced.
   *
   * @param op_new A raw pointer to the new input op.
   */
  virtual void replace_input_op(Op* op_old, Op* op_new) = 0;

  /**
   * @brief Get fan-out OP number
   *
   * @details XIR graph doesn't allow that an OP has more than one output
   * tensor, but there may be different OPs which take the output tensor as
   * their input. We call those OPs fan-out OP. This function return the
   * number of fan-out OPs.
   *
   * @return fan-out number
   */
  virtual int get_fanout_num() const = 0;

  /**
   * @brief Get all fan-out OPs
   *
   * @return vector of fan-out OPs
   */
  virtual std::vector<Op*> get_fanout_ops() = 0;

  /**
   * @brief Get all fan-out OPs
   *
   * @return vector of fan-out OPs
   */
  virtual const std::vector<const Op*> get_fanout_ops() const = 0;

  /**
   * @brief Get all input tensors
   *
   * @return vector of input tensors
   */
  virtual std::vector<Tensor*> get_input_tensors() = 0;

  /**
   * @brief Get all input tensors
   *
   * @return vector of input tensors
   */
  virtual const std::vector<const Tensor*> get_input_tensors() const = 0;

  /**
   * @brief Get all input tensors with specific arg_name
   *
   * @param arg_name Specific argument name
   *
   * @return vector of input tensor, the order is guaranteed to be same as
   * input OPs
   */
  virtual std::vector<Tensor*> get_input_tensors(std::string arg_name) = 0;

  /**
   * @brief Get all input tensors with specific arg_name
   *
   * @param arg_name Specific argument name
   *
   * @return vector of input tensor, the order is guaranteed to be same as
   * input OPs
   */
  virtual const std::vector<const Tensor*> get_input_tensors(
      std::string arg_name) const = 0;

  /**
   * @brief Get input tensor with specific arg_name and index
   *
   * @param arg_name Specific argument name
   *
   * @param idx Index of the input tensor. The default value of idx is 0
   *
   * @return input tensor
   */
  virtual Tensor* get_input_tensor(std::string arg_name, int idx = 0) = 0;

  /**
   * @brief Get input tensor with specific arg_name and index
   *
   * @param arg_name Specific argument name
   *
   * @param idx Index of the input tensor. The default value of idx is 0
   *
   * @return input tensor
   */
  virtual const Tensor* get_input_tensor(std::string arg_name,
                                         int idx = 0) const = 0;

  /**
   * @brief Get output tensor
   *
   * @return output tensor
   */
  virtual Tensor* get_output_tensor() = 0;

  /**
   * @brief Get output tensor
   *
   * @return output tensor
   */
  virtual const Tensor* get_output_tensor() const = 0;

  /**
   * @brief Replace the op's output tensor.
   *
   * @param tensor_new A unique pointer to the new output tensor.
   */
  virtual void replace_output_tensor(std::unique_ptr<Tensor> tensor_new) = 0;

  /**
   * @brief Get the the graph to which the op belongs.
   *
   * @return A raw pointer to the graph.
   */
  virtual Graph* get_graph() = 0;

  /**
   * @brief Get the the graph to which the op belongs.
   *
   * @return A raw pointer to the graph.
   */
  virtual const Graph* get_graph() const = 0;

  /**
   * @brief Check the existence of the Attrs object.
   *
   * @return If this op has Attrs, return true, else false.
   */
  virtual bool has_attrs() const = 0;

  /**
   * @brief Get a copy of OP attributes
   *
   * @return OP attributes
   */
  virtual std::unique_ptr<Attrs> get_attrs() const = 0;

  /**
   * @brief Set OP attributes
   *
   * @param attrs OP attributes Attrs object.
   */
  virtual Op* set_attrs(std::unique_ptr<Attrs> attrs) = 0;

  /**
   * @brief Check the existence of the attribute indicated by key.
   *
   * @param key The attribute index name.
   *
   * @return If this op has this attribute return true, else false.
   */
  virtual bool has_attr(const std::string& key) const = 0;

  /**
   * @brief Get the attribute value indicated by key.
   *
   * @param key The attribute name.
   *
   * @return Attribute value.
   */
  virtual const xir::any get_attr(const std::string& key) const = 0;

  /**
   * @brief Set OP attribute indicated by <key, value> pair.
   *
   * @param key Attribute name.
   *
   * @param value Attribute value.
   *
   * @return A raw pointer to the OP.
   */
  virtual Op* set_attr(const std::string& key, const xir::any& value) = 0;

  /**
   * @brief Get the attribute value indicated by key.
   *
   * @param key The attribute name.
   *
   * @return Attribute value.
   */
  template <typename Dtype>
  const Dtype get_attr(const std::string& key) const {
    return xir::stdx::any_cast<Dtype>(this->get_attr(key));
  }

  /**
   * @brief Set OP attribute indicated by <key, value> pair.
   *
   * @param key Attribute name.
   *
   * @param value Attribute value.
   *
   * @return A raw pointer to the OP.
   */
  template <typename Dtype>
  Op* set_attr(const std::string& key, const Dtype& value) {
    this->set_attr(key, xir::any{value});
    return this;
  }

  /**
   * @brief Inference the output tensor shape.
   */
  virtual void shape_infer() = 0;

  /**
   * @brief Get the OpDeff of this Op.
   *
   * @return A raw pointer to the OpDef object.
   */
  virtual const OpDef* get_opdef() const = 0;

  /**
   * @brief Return the brief info of op in std::string format.
   *
   * @return Breif info of op in std::string format.
   */
  virtual const std::string to_string(
      const std::string& delimiter = ",",     //
      const std::string& left_bracket = "{",  //
      const std::string& right_bracket = "}") const = 0;

 public:
  virtual ~Op() = default;
};

}  // namespace xir
