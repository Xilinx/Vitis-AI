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
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "xir/graph/graph_template.hpp"
#include "xir/op/op.hpp"
#include "xir/util/any.hpp"

namespace xir {

class Graph;

/**
 * @class Subgraph
 *
 * @brief A class for subgraph
 */
class Subgraph {
 public:
  /**
   * @brief Get the name of subgraph.
   *
   * @return The name of the subgraph.
   */
  virtual const std::string get_name() const = 0;

  /**
   * @brief Set the name of subgraph.
   *
   * @param subgraph_name The name of the subgraph.
   */
  virtual Subgraph* set_name(const std::string& subgraph_name) = 0;

  /**
   * @brief Get the number of ops which belong to this subgraph.
   *
   * @return Number of ops.
   */
  virtual std::int32_t get_op_num() const = 0;

  /**
   * @brief Get all the ops which belong to this subgraph.
   *
   * @return A set of raw pointers.
   */
  virtual std::set<Op*> get_ops() = 0;

  /**
   * @brief Get all the ops which belong to this subgraph.
   *
   * @return A set of raw pointers.
   */
  virtual const std::set<const Op*> get_ops() const = 0;

  /**
   * @brief Find a tensor's producer op in this subgraph.
   *
   * @details If the producer doesn't exist or belongs to other subgraph, a
   * nullptr will be returned.
   *
   * @param tensor A raw pointer of a tensor.
   *
   * @return A raw pointer to the producer op.
   */
  virtual Op* get_tensor_producer(const Tensor* tensor) = 0;

  /**
   * @brief Find a tensor's producer op in this subgraph.
   *
   * @details If the producer doesn't exist or belongs to other subgraph, a
   * nullptr will be returned.
   *
   * @param tensor A raw pointer of a tensor.
   *
   * @return A raw pointer to the producer op.
   */
  virtual const Op* get_tensor_producer(const Tensor* tensor) const = 0;

  /**
   * @brief Get all the input tensors produced by other subgraph.
   *
   * @return A set of raw pointer to the input tensors.
   */
  virtual std::set<Tensor*> get_input_tensors() = 0;

  /**
   * @brief Get all the input tensors produced by other subgraph.
   *
   * @return A set of raw pointer to the input tensors.
   */
  virtual const std::set<const Tensor*> get_input_tensors() const = 0;

  /**
   * @brief Get all the tensors output to other subgraphs or dump out in current
   * subgraph.
   *
   * @details There are two parts inside the output tensors. First, the tensor
   * is passed to another subgraph as an input tensor; second, the tensor is
   * dump out in the current subgraph, for instance an op without fanout.
   *
   * @return A set of raw pointer to the output tensors.
   */
  virtual std::set<Tensor*> get_output_tensors() = 0;

  /**
   * @brief Get all the tensors output to other subgraphs or dump out in current
   * subgraph.
   *
   * @details There are two parts inside the output tensors. First, the tensor
   * is passed to another subgraph as an input tensor; second, the tensor is
   * dump out in the current subgraph, for instance an op without fanout.
   *
   * @return A set of raw pointer to the output tensors.
   */
  virtual const std::set<const Tensor*> get_output_tensors() const = 0;

  /**
   * @brief Check the existence of the op indicated by name.
   *
   * @param op_name The name of the op.
   *
   * @return True if exists, else false.
   */
  virtual bool has_op(const std::string& op_name) const = 0;

  /**
   * @brief Check the existence of the op indicated by a pointer of op.
   *
   * @param op A raw pointer of an op.
   *
   * @return True if exists, else false.
   */
  virtual bool has_op(const Op* op) const = 0;

  /**
   * @brief Find the child subgraph to which the op belongs.
   *
   * @details If there's no child subgraph or this op is from outside of the
   * current subgraph, a nullptr will be returned.
   *
   * @param op A raw pointer of an op.
   *
   * @return A raw pointer to the child subgraph.
   */
  virtual Subgraph* find_op(const Op* op) = 0;

  /**
   * @brief Find the child subgraph to which the op belongs.
   *
   * @details If there's no child subgraph or this op is from outside of the
   * current subgraph, a nullptr will be returned.
   *
   * @param op A raw pointer of an op.
   *
   * @return A raw pointer to the child subgraph.
   */
  virtual const Subgraph* find_op(const Op* op) const = 0;

  /**
   * @brief Find the child subgraph to which the op belongs.
   *
   * @details If there's no child subgraph or this op is from outside of the
   * current subgraph, a nullptr will be returned.
   *
   * @param op_name The name of an op.
   *
   * @return A raw pointer to the child subgraph.
   */
  virtual Subgraph* find_op(const std::string& op_name) = 0;

  /**
   * @brief Find the child subgraph to which the op belongs.
   *
   * @details If there's no child subgraph or this op is from outside of the
   * current subgraph, a nullptr will be returned.
   *
   * @param op_name The name of an op.
   *
   * @return A raw pointer to the child subgraph.
   */
  virtual const Subgraph* find_op(const std::string& op_name) const = 0;

  /**
   * @brief Check if this subgraph is a root subgraph.
   *
   * @return True if it's the root, else false
   */
  virtual bool is_root() const = 0;

  /**
   * @brief Check if this subgraph is a leaf subgraph.
   *
   * @return True if it's the leaf, else false.
   */
  virtual bool is_leaf() const = 0;

  /**
   * @brief Get the root subgraph of the current subgraph.
   *
   * @return A raw pointer to the root subgraph.
   */
  virtual Subgraph* get_root() = 0;

  /**
   * @brief Get the root subgraph of the current subgraph.
   *
   * @return A raw pointer to the root subgraph.
   */
  virtual const Subgraph* get_root() const = 0;

  /**
   * @brief Get the depth of the current subgraph.
   *
   * @return The depth of the current subgraph.
   */
  virtual std::int32_t get_depth() const = 0;

  /**
   * @brief Get the parent subgraph of the current subgraph.
   *
   * @return A raw pointer to the parent subgraph.
   */
  virtual Subgraph* get_parent() = 0;

  /**
   * @brief Get the parent subgraph of the current subgraph.
   *
   * @return A raw pointer to the parent subgraph.
   */
  virtual const Subgraph* get_parent() const = 0;

  /**
   * @brief Create children subgraph for the current subgraph.
   *
   * @details Create the children subgraph for the current subgraph while the
   * current subgraph is a leaf subgraph, if not, a fatal will of
   * XIR_SUBGRAPH_CREATE_CHILDREN_FOR_NONLEAF will be raised. And for the new
   * created children subgraphs, each of them only contains one op.
   */
  virtual void create_children() = 0;

  /**
   * @brief Get the number of children subgraphs.
   *
   * @return The number of the children subgraphs.
   */
  virtual std::int32_t get_children_num() const = 0;

  /**
   * @brief Get all the children subgraphs of the current subgraph.
   *
   * @return A set of raw pointer to the children subgraphs.
   */
  virtual std::set<Subgraph*> get_children() = 0;

  /**
   * @brief Get all the children subgraphs of the current subgraph.
   *
   * @return A set of raw pointer to the children subgraphs.
   */
  virtual const std::set<const Subgraph*> get_children() const = 0;

  /**
   * @brief Check if the input subgraph is a child of the current subgraph.
   *
   * @param subgraph A pointer to the input subgraph's.
   *
   * @return True if is a child, else false.
   */
  virtual bool is_child(Subgraph* subgraph) const = 0;

  /**
   * @brief Merge a set of child subraphs into one child subgraph.
   *
   * @param subgraph_list A set of child subgraphs to be merged.
   *
   * @return A raw pointer to the new merged subgraph.
   */
  virtual Subgraph* merge_children(std::set<Subgraph*> subgraph_list) = 0;

  /**
   * @brief Get the corresponding graph of the current subgraph.
   *
   * @return A raw pointer to the graph.
   */
  virtual Graph* get_graph() = 0;

  /**
   * @brief Get the corresponding graph of the current subgraph.
   *
   * @return A raw pointer to the graph.
   */
  virtual const Graph* get_graph() const = 0;

  /**
   * @brief Get the children subgraph with corresponding name from this
   * subgraph.
   *
   * @param name Name of the children subgraph.
   *
   * @return A raw pointer to the children subgraph.
   */
  virtual Subgraph* get_subgraph(const std::string& name) = 0;

  /**
   * @brief Get the children subgraph with corresponding name from this
   * subgraph.
   *
   * @param name Name of the children subgraph.
   *
   * @return A raw pointer to the children subgraph.
   */
  virtual const Subgraph* get_subgraph(const std::string& name) const = 0;

  /**
   * @brief Check the existence of the Attrs object in current subgraph.
   *
   * @return True if exist, else false.
   */
  virtual bool has_attrs() const = 0;

  /**
   * @brief Get all the attrs in the current subgraph.
   *
   * @return A unique pointer to the Attrs object.
   */
  virtual std::unique_ptr<Attrs> get_attrs() const = 0;

  /**
   * @brief Set an Attrs object to the current subgraph.
   *
   * @param attrs A unique pointer to the Attrs object to be set.
   */
  virtual Subgraph* set_attrs(std::unique_ptr<Attrs> attrs) = 0;

  /**
   * @brief Check the existence of the attribute indicated by key.
   *
   * @param key The attribute name.
   *
   * @return True if exist, else false.
   */
  virtual bool has_attr(const std::string& key) const = 0;

  /**
   * @brief Get the attribute value which indicated by key.
   *
   * @param key The attribute name.
   *
   * @return The attribute value.
   */
  virtual const xir::any get_attr(const std::string& key) const = 0;

  /**
   * @brief Set the <key, value> attribute pair.
   *
   * @param key The name of the attribute.
   *
   * @param value The value of the attribute.
   *
   * @return A raw pointer to the current subgraph.
   */
  virtual Subgraph* set_attr(const std::string& key, const xir::any& value) = 0;

  /**
   * @brief Get the attribute value which indicated by key.
   *
   * @param key The attribute name.
   *
   * @return The attribute value.
   */
  template <typename Dtype>
  const Dtype get_attr(const std::string& key) const {
    return stdx::any_cast<Dtype>(this->get_attr(key));
  }

  /**
   * @brief Set the <key, value> attribute pair.
   *
   * @param key The name of the attribute.
   *
   * @param value The value of the attribute.
   *
   * @return A raw pointer to the current subgraph.
   */
  template <typename Dtype>
  Subgraph* set_attr(const std::string& key, const Dtype& value) {
    this->set_attr(key, xir::any{value});
    return this;
  }

  /**
   * @brief Get all the ops of the current subgraph in the topological order.
   *
   * @return A vector of the raw pointer of ops.
   */
  virtual std::vector<Op*> topological_sort() = 0;

  /**
   * @brief Get all the ops of the current subgraph in the topological order.
   *
   * @return A vector of the raw pointer of ops.
   */
  virtual const std::vector<const Op*> topological_sort() const = 0;

  /**
   * @brief Get all the children subgraphs of the current subgraph in the
   * topological order.
   *
   * @return A vector of the raw pointer of children subgraphs.
   */
  virtual std::vector<Subgraph*> children_topological_sort() = 0;

  /**
   * @brief Get all the children subgraphs of the current subgraph in the
   * topological order.
   *
   * @return A vector of the raw pointer of children subgraphs.
   */
  virtual const std::vector<const Subgraph*> children_topological_sort()
      const = 0;

  /**
   * @brief Find all the isomorphism of the template in the current subgraph.
   *
   * @param graph_template The graph template used to find the isomorphism.
   *
   * @return A vetor that contains all the isomorphism structure, which is a map
   * indexed by the element of the graph template.
   */
  virtual std::vector<std::map<OpTemplate*, Op*>> isomorphism(
      GraphTemplate* graph_template) = 0;

  /**
   * @brief Save the subgraph into a dot file.
   *
   * @param file_path The path of the dot file.
   */
  virtual void save_to_dot(const std::string& file_path) const = 0;

  /**
   * @brief Return the brief info of subgraph in std::string format.
   *
   * @return Breif info of subgraph in std::string format.
   */
  virtual const std::string to_string(
      const std::string& delimiter = ",",     //
      const std::string& left_bracket = "{",  //
      const std::string& right_bracket = "}") const = 0;

 public:
  virtual ~Subgraph() = default;
};

}  // namespace xir
