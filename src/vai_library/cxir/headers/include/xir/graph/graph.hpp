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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "xir/graph/graph_template.hpp"
#include "xir/graph/subgraph.hpp"
#include "xir/op/op.hpp"
#include "xir/util/data_type.hpp"

namespace xir {

/**
 * @brief XIR graph interface
 *
 * @details This class defines the basic XIR graph Interface. Here are some
 * basic features of XIR Graph. First, the graph is constructed of OPs. Each OP
 * can have some pre-defined input arguments, such as "weights" and "input". And
 * those arguments have an occurance type to claim how many time they will show
 * up. For example, "conv2d" must have one "weights" argument and "eltwise" must
 * have at least one "input" arguments. Note that the actual number of input
 * tensors may be not equal to the number of pre-defined arguments for an OP.
 * You can refer OpDef for more infomation. And OP can only have one output
 * argument with OPTIONAL or REQUIRED type, which means it can only have zero or
 * one output tensor. This is a very strong constrain to avoid parallel edges in
 * graph. But we think it is acceptable. Then, the graph must have no ring. We
 * will support LSTM in another way rather than adding an edge directly.
 * Maintaining a graph with rings is much harder than acyclic graph.
 */
class Graph {
 public:
  /**
   * @brief Static function to create a graph with a name.
   *
   * @param name Name of the created graph.
   *
   * @return An instance of graph.
   */
  static std::unique_ptr<Graph> create(std::string name);

  /**
   * @brief Get name of the graph.
   *
   * @return Graph name.
   */
  virtual const std::string get_name() const = 0;

  /**
   * @brief Add an operator to the graph.
   *
   * @details Note that the parameters have to satisfy conditions descrided in
   * the parameter field, or the function will crash. Those conditions make sure
   * that there will never be an invalid status during the graph modification.
   * But that brings that you must build a graph carefully and add OPs in
   * topologic order.
   *
   * @param name Name of the OP. It has to be different from every existed OP
   * in the graph.
   *
   * @param type Type of the OP. It has to be registered into OpDefFactory.
   *
   * @param attrs Attributes of the OP. It has to contain all the required op
   * attributes which are defined in OpDef.
   *
   * @param input_ops_map Map of input operators where key is input argument
   * name defined in OpDef and value is vector of input operator pointer. The
   * number of the input operator has to be appropriate with the defination from
   * OpDef.
   *
   * @param subgraph The subgraph the new op belongs to.
   *
   * @return An instance of OP.
   */
  virtual Op* add_op(
      const std::string& name, const std::string& type,
      std::unique_ptr<Attrs> attrs,
      const std::map<std::string, std::vector<Op*>>& input_ops_map,
      Subgraph* subgraph = nullptr) = 0;

  /**
   * @brief Add an operator to the graph.
   *
   * @details Note that the parameters have to satisfy conditions descrided in
   * the parameter field, or the function will crash. Those conditions make sure
   * that there will never be an invalid status during the graph modification.
   * But that brings that you must build a graph carefully and add OPs in
   * topologic order.
   *
   * @param name Name of the OP. It has to be different from every existed OP
   * in the graph.
   *
   * @param type Type of the OP. It has to be registered into OpDefFactory.
   *
   * @param attrs Attributes of the OP. It has to contain all the required op
   * attributes which are defined in OpDef.
   *
   * @param input_ops_map Map of input operators where key is input argument
   * name defined in OpDef and value is vector of input operator pointer. The
   * number of the input operator has to be appropriate with the defination from
   * OpDef.
   *
   * @param output_data_type The data type of the output tensor.
   *
   * @param subgraph The subgraph the new op belongs to.
   *
   * @return An instance of OP.
   */
  virtual Op* add_op(
      const std::string& name, const std::string& type,
      std::unique_ptr<Attrs> attrs,
      const std::map<std::string, std::vector<Op*>>& input_ops_map,
      const DataType& output_data_type, Subgraph* subgraph = nullptr) = 0;

  /**
   * @brief Remove a operator from graph.
   *
   * @details For the same purpose as add_op, the OP you want to remove should
   * not be used by any other OPs, aka get_fanout_num() == 0. That will make
   * sure nobody will lose its input after an Op is removed.
   *
   * @param op The pointer of the OP you want to remove from the graph.
   */
  virtual void remove_op(Op* op) = 0;

  /**
   * @brief !!!experimental!!! Replace a group of ops with others
   *
   * @param replaced_ops Ops to be replaced
   *
   * @param creator funtion to create new Ops and return the correspondence of
   * tail ops
   *
   */
  virtual void replace_ops(
      std::map<std::string, Op*> replaced_ops,
      std::function<std::map<Op*, Op*>(Graph*, std::map<std::string, Op*>)>
          creator) = 0;

  /**
   * @brief Find op with a specific name.
   *
   * @param op_name OP name.
   *
   * @return pointer to the op with specific name, return nullptr if cannot find
   * it.
   */
  virtual Op* get_op(const std::string& op_name) = 0;

  /**
   * @brief Find op with a specific name.
   *
   * @param op_name OP name.
   *
   * @return pointer to the op with specific name, return nullptr if cannot find
   * it.
   */
  virtual const Op* get_op(const std::string& op_name) const = 0;

  /**
   * @brief Find tensor with a specific name.
   *
   * @param tensor_name Tensor name.
   *
   * @return pointer to the tensor with specific name, return nullptr if cannot
   * find it.
   */
  virtual Tensor* get_tensor(const std::string& tensor_name) = 0;

  /**
   * @brief Find tensor with a specific name.
   *
   * @param tensor_name Tensor name.
   *
   * @return pointer to the tensor with specific name, return nullptr if cannot
   * find it.
   */
  virtual const Tensor* get_tensor(const std::string& tensor_name) const = 0;

  /**
   * @brief Get number of OPs existed in the graph.
   *
   * @return OP number.
   */
  virtual int get_op_num() const = 0;

  /**
   * @brief Get all OP pointers in the graph.
   *
   * @return A vector of OP pointers.
   */
  virtual std::set<Op*> get_ops() = 0;

  /**
   * @brief Get all OP pointers in the graph.
   *
   * @return A set of OP pointers.
   */
  virtual const std::set<const Op*> get_ops() const = 0;

  /**
   * @brief Get all Tensor pointers in the graph.
   *
   * @return A vector of Tensor pointers.
   */
  virtual std::set<Tensor*> get_tensors() = 0;

  /**
   * @brief Get all Tensor pointers in the graph.
   *
   * @return A vector of Tensor pointers.
   */
  virtual const std::set<const Tensor*> get_tensors() const = 0;

  /**
   * @brief Get all OP pointers with no input OP.
   *
   * @return A set of OP pointers.
   */
  virtual std::set<Op*> get_head_ops() = 0;

  /**
   * @brief Get all OP pointers with no input OP.
   *
   * @return A set of OP pointers.
   */
  virtual const std::set<const Op*> get_head_ops() const = 0;

  /**
   * @brief Get all OP pointers with no fanout OP.
   *
   * @return A set of OP pointers.
   */
  virtual std::set<Op*> get_tail_ops() = 0;

  /**
   * @brief Get all OP pointers with no fanout OP.
   *
   * @return A set of OP pointers.
   */
  virtual const std::set<const Op*> get_tail_ops() const = 0;

  /**
   * @brief Get the tensor's producer Op.
   *
   * @details If the producer doesn't exist, a nullptr will be returned.
   *
   * @param tensor A raw pointer of a tensor.
   *
   * @return A raw pointer to the producer op, return nullptr if cannot find it.
   */
  virtual Op* get_tensor_producer(const Tensor* tensor) = 0;

  /**
   * @brief Get the tensor's producer Op.
   *
   * @details If the producer doesn't exist, a nullptr will be returned.
   *
   * @param tensor A raw pointer of a tensor.
   *
   * @return A raw pointer to the producer op, return nullptr if cannot find it.
   */
  virtual const Op* get_tensor_producer(const Tensor* tensor) const = 0;

  /**
   * @brief Get OPs in topological order
   *
   * @return A vector of OP pointers in topological order
   */
  virtual std::vector<Op*> topological_sort() = 0;

  /**
   * @brief Get OPs in topological order
   *
   * @return A vector of OP pointers in topological order
   */
  virtual const std::vector<const Op*> topological_sort() const = 0;

  /**
   * @brief Inference the shape for all the tensor
   */
  virtual void infer_shape() = 0;

  /**
   * @brief Find all the isomorphism of the template in the graph.
   *
   * @param graph_template The graph template used to find the isomorphism.
   *
   * @return A vetor that contains all the isomorphism structure, which is a map
   * indexed by the element of the graph template.
   */
  virtual std::vector<std::map<OpTemplate*, Op*>> isomorphism(
      GraphTemplate* graph_template) = 0;

  /**
   * @brief Save graph to dot format which could be visualized by graphviz.
   *
   * @param file_path The path of dot file.
   *
   */
  virtual void save_to_dot(const std::string& file_path) const = 0;

  /**
   * @brief Save graph to other format.
   *
   * @param file_path The path of the output picture.
   *
   * @param format such as "png", "svg".
   */
  virtual void visualize(const std::string& file_path,
                         const std::string& format) const = 0;

  /**
   * @brief Serialize the graph.
   *
   * @param file_path The path of output xmodel.
   *
   * @return A string storing the graph.
   */
  virtual void serialize(const std::string& file_path) const = 0;

  /**
   * @brief Serialize the graph.
   *
   * @return A string storing the graph.
   */
  virtual void serialize_to_string(std::string* str) const = 0;

  /**
   * @brief Deserializa a graph from a pb file.
   *
   * @param file_path The path of the xmodel.
   *
   * @return A unique pointer to the graph object.
   */
  static std::unique_ptr<Graph> deserialize(const std::string& file_path);

  /**
   * @brief Deserializa a graph from a string.
   *
   * @param str The generated string of pb file.
   *
   * @return A unique pointer to the graph object.
   */
  static std::unique_ptr<Graph> deserialize_from_string(const std::string& str);

  /**
   * @brief Get root subgraph of this graph.
   *
   * @return A pointer to root subgraph.
   */
  virtual const Subgraph* get_root_subgraph() const = 0;

  /**
   * @brief Get root subgraph of this graph.
   *
   * @return A pointer to root subgraph.
   */
  virtual Subgraph* get_root_subgraph() = 0;

  /**
   * @brief Get the leaf subgraph to which the op belongs.
   *
   * @param op A raw pointer to the op.
   *
   * @return A raw pointer to the subgraph.
   */
  virtual Subgraph* get_leaf_subgraph(const Op* op) = 0;

  /**
   * @brief Get the leaf subgraph to which the op belongs.
   *
   * @param op A raw pointer to the op.
   *
   * @return A raw pointer to the subgraph.
   */
  virtual const Subgraph* get_leaf_subgraph(const Op* op) const = 0;

  /**
   * @brief Get the subgraph with corresponding name from this graph.
   *
   * @param name Name of the subgraph.
   *
   * @return A raw pointer to the subgraph.
   */
  virtual Subgraph* get_subgraph(const std::string& name) = 0;

  /**
   * @brief Get the subgraph with corresponding name from this graph.
   *
   * @param name Name of the subgraph.
   *
   * @return A raw pointer to the subgraph.
   */
  virtual const Subgraph* get_subgraph(const std::string& name) const = 0;

  /**
   * @brief Check the existence of the Attrs object in current subgraph.
   *
   * @return True if exist, else false.
   */
  virtual bool has_attrs() const = 0;

  /**
   * @brief Get all the attrs in the current graph.
   *
   * @return A unique pointer to the Attrs object.
   */
  virtual std::unique_ptr<Attrs> get_attrs() const = 0;

  /**
   * @brief Set an Attrs object to the current graph.
   *
   * @param attrs A unique pointer to the Attrs object to be set.
   *
   * @return A raw pointer to the current graph.
   */
  virtual Graph* set_attrs(std::unique_ptr<Attrs> attrs) = 0;

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
   * @return A raw pointer to the current graph.
   */
  virtual Graph* set_attr(const std::string& key, const xir::any& value) = 0;

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
   * @return A raw pointer to the current graph.
   */
  template <typename Dtype>
  Graph* set_attr(const std::string& key, const Dtype& value) {
    this->set_attr(key, xir::any{value});
    return this;
  }

  /**
   * @brief Return the brief info of graph in std::string format.
   *
   * @return Breif info of graph in std::string format.
   */
  virtual const std::string to_string(
      const std::string& delimiter = ",",     //
      const std::string& left_bracket = "{",  //
      const std::string& right_bracket = "}") const = 0;

 public:
  virtual ~Graph() = default;
};

}  // namespace xir
