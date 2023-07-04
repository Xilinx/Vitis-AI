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
#include <stdint.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
/**
 * the obscure type for xir graph
 */
typedef void* xir_graph_t;

/**
 * the obscure type for xir subgraph
 */
typedef void* xir_subgraph_t;

/**
 * the obscure type for xir attrs
 */
typedef void* xir_attrs_t;

/**
 * the obscure type for xir Op
 */
typedef void* xir_op_t;

/**
 * the obscure type for xir op arg def
 */
typedef void* xir_op_arg_def_t;

/**
 * the obscure type for xir attr def
 */
typedef void* xir_attr_def_t;

/**
 * the obscure type for xir op def
 */
typedef void* xir_op_def_t;

/**
 * the obscure type for type_index
 */
typedef void* xir_type_index_t;

typedef struct {
  const char* data;
  size_t size;
} xir_string_t;
typedef struct {
  char* data;
  size_t size;
} xir_bytes_t;
typedef struct {
  int value;
} xir_bool_t;

typedef struct xir_attr_value_t xir_attr_value_t;
typedef struct {
  size_t lower_bound;
  int has_upper_bound;
  size_t upper_bound;
} xir_size_hint_t;

typedef struct {
  void* self;
  xir_attr_value_t (*next)(void* self);
  xir_size_hint_t (*size_hint)(void* self);
  void (*destroy)(void* self);
} xir_attr_value_iterator_t;

typedef struct xir_attr_pair_t xir_attr_pair_t;

typedef struct {
  void* self;
  xir_attr_pair_t (*next)(void* self);
  void (*destroy)(void* self);
} xir_attr_value_map_iterator_t;

/**
 * the obscure type for xir tensor
 */
typedef void* xir_tensor_t;

enum xir_tensor_data_type_t {
  XIR_INT,
  XIR_UINT,
  XIR_XINT,
  XIR_XUINT,
  XIR_FLOAT,
  UNKNOWN
};

/**
 * @brief create a graph with a name.
 *
 * @param name Name of the created graph.
 *
 * @return An instance of graph.
 */

xir_graph_t xir_graph_create(const char* name);

/**
 * @brief Deserializa a graph from a pb file.
 *
 * @param pb_fname The path of the pb file.
 *
 * @return A unique pointer to the graph object.
 */
xir_graph_t xir_graph_deserialize(const char* pb_fname);

/**
 * @brief Serialize the graph.
 *
 * @param file_path The path of output xmodel.
 *
 * @return A string storing the graph.
 */
void xir_graph_serialize(const xir_graph_t graph, const char* file_path);

/**
 * @brief destroy a graph
 *
 * @return return value is not in use yet. it is always zero.
 */
int xir_graph_destroy(xir_graph_t graph);
/**
 * @brief Get name of the graph.
 *
 * @return Graph name.
 */
const char* xir_graph_get_name(xir_graph_t graph);

/**
 * @brief Get root subgraph of this graph.
 *
 * @return A pointer to root subgraph.
 */
xir_subgraph_t xir_graph_get_root_subgraph(xir_graph_t graph);

/**
 * @brief Get the leaf subgraph to which the op belongs.
 *
 * @param op A raw pointer to the op.
 *
 * @return A raw pointer to the subgraph.
 */
xir_subgraph_t xir_graph_get_leaf_subgraph(xir_graph_t graph, xir_op_t op);

/**
 * @brief Get the subgraph with corresponding name from this graph.
 *
 * @param name Name of the subgraph.
 *
 * @return A raw pointer to the subgraph.
 */
xir_subgraph_t xir_graph_get_subgraph(xir_graph_t graph, xir_string_t name);

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
 * @param attrs Attributes of the OP. It has to contain all the
 * required op attributes which are defined in OpDef. The ownership is
 * taken away, the caller must not invoke xir_attrs_destroy() later on.
 *
 * @param input_ops_map Map of input operators where key is input argument
 * name defined in OpDef and value is vector of input operator pointer. The
 * number of the input operator has to be appropriate with the defination from
 * OpDef.
 *
 * @param tensor The ownership is taken so that the caller must not
 * invoke xir_tensor_destoy() later on.
 *
 * @return An instance of OP.
 */
struct xir_graph_input_ops_t {
  xir_string_t name;
  xir_op_t* ops;
  size_t num_of_ops;
};

xir_op_t xir_graph_add_op(xir_graph_t graph,                        //
                          xir_string_t name,                        //
                          xir_string_t type,                        //
                          xir_attrs_t attrs,                        //
                          struct xir_graph_input_ops_t* input_ops,  //
                          size_t num_of_ops,                        //
                          xir_subgraph_t subgraph                   //
);

/**
 * @brief Remove a operator from graph.
 *
 * @details For the same purpose as add_op, the OP you want to remove should
 * not be used by any other OPs, aka get_fanout_num() == 0. That will make
 * sure nobody will lose its input after an Op is removed.
 *
 * @param op The pointer of the OP you want to remove from the graph.
 */
void xir_graph_remove_op(xir_graph_t graph, xir_op_t op);

/**
 * @brief Get number of OPs existed in the graph.
 *
 * @return OP number.
 */
int xir_graph_get_op_num(xir_graph_t graph);

/**
 * @brief Get all OP pointers in the graph.
 *
 * @return A vector of OP pointers.
 */
void xir_graph_get_ops(xir_graph_t graph, xir_op_t ret[]);

/**
 * @brief Find op with a specific name.
 *
 * @param op_name OP name.
 *
 * @return pointer to the op with specific name, return nullptr if cannot find
 * it.
 */
xir_op_t xir_graph_get_op(xir_graph_t graph, xir_string_t op);

/**
 * @brief Get all Tensor pointers in the graph.
 *
 * @return A iterator of Tensor pointers.
 */
void xir_graph_get_tensors(xir_graph_t graph, xir_tensor_t ret[]);

/**
 * @brief Find tensor with a specific name.
 *
 * @param tensor_name Tensor name.
 *
 * @return pointer to the tensor with specific name, return nullptr if cannot
 * find it.
 */
xir_tensor_t xir_graph_get_tensor(xir_graph_t graph, xir_string_t tensor_name);

typedef void (*xir_get_op_callback_t)(void* data, xir_op_t op);
typedef void (*xir_get_tensor_callback_t)(void* data, xir_tensor_t tensor);

/**
 * @brief Get all OP pointers with no input OP.
 *
 * @return A set of OP pointers.
 */
void xir_graph_get_head_ops(xir_graph_t graph, void* data,
                            xir_get_op_callback_t cb);

/**
 * @brief Get all OP pointers with no fanout OP.
 *
 * @return A set of OP pointers.
 */
void xir_graph_get_tail_ops(xir_graph_t graph, void* data,
                            xir_get_op_callback_t cb);

/*
 * @brief Get the tensor's producer Op.
 *
 * @details If the producer doesn't exist, a nullptr will be returned.
 *
 * @param tensor A raw pointer of a tensor.
 *
 * @return A raw pointer to the producer op, return nullptr if cannot find it.
 */
xir_op_t xir_graph_get_tensor_producer(xir_graph_t graph, xir_tensor_t tensor);

/**
 * @brief Get OPs in topological order
 *
 * @return A vector of OP pointers in topological order
 */
void xir_graph_topological_sort(xir_tensor_t graph, void* data,
                                xir_get_op_callback_t cb);

/**
 * @brief Get all the attrs in the current graph.
 *
 * @return A unique pointer to the Attrs object.
 */
xir_attrs_t xir_graph_get_attrs(xir_graph_t graph);

/**
 * @brief Set an Attrs object to the current graph.
 *
 * @param attrs A unique pointer to the Attrs object to be set.
 */
void xir_graph_set_attrs(xir_graph_t graph, xir_attrs_t attrs);

/**
 * @breif Create a new empty Attrs object, and return a unique pointer
 * of the new object.
 *
 * @return A unique pointer to the created Attrs object.
 */

xir_attrs_t xir_attrs_create();
/**
 * @brief destroy the xir attrs
 */
void xir_attrs_destroy(xir_attrs_t);

// clang-format off
#define XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES2(p)     \
        p(bool, xir_bool_t, bool) \
                p(i8, int8_t, int8_t)               \
                p(i16, int16_t, int16_t)            \
                p(i32, int32_t, int32_t)            \
                p(i64, int64_t, int64_t)            \
                p(u8,  uint8_t,  uint8_t)           \
                p(u16, uint16_t, uint16_t)          \
        p(u32, uint32_t, uint32_t)                  \
        p(u64, uint64_t, uint64_t)                  \
        p(f32, float, float)                        \
        p(f64, double, double)                      \
        p(string, xir_string_t, std::string)       \
        p(bytes, xir_bytes_t, std::vector<char>)

// IMPORTANT: p(bytes, xir_bytes_t, std::vector<char>) vs p(bytes, xir_bytes_t, std::vector<int8_t>)
//   they are not same, vector<char> is not as same as vector<int8_t> i.e. vector<signed char>
//
//  p(type_name, c_type, c++ type )
// clang-format on
enum xir_attr_value_tag_t {
  XIR_ATTR_TYPE_TAG_NONE = 0,
  XIR_ATTR_TYPE_TAG_MAP = 1,
#define XIR_ATTRS_DECL_PRIMITIVE_TAG(name, c_type, cpp_type)                   \
  XIR_ATTR_TYPE_TAG_##name, XIR_ATTR_TYPE_TAG_VEC_##name,                      \
      XIR_ATTR_TYPE_TAG_MAP_##name, XIR_ATTR_TYPE_TAG_MAP_VEC_##name,
  XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES2(XIR_ATTRS_DECL_PRIMITIVE_TAG)
};

struct xir_attr_value_t {
  enum xir_attr_value_tag_t tag;
  union u {
#define XIR_ATTRS_DECL_PRIMITIVE_VALUES(name, c_type, cpp_type)                \
  c_type name##_value;
    XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES2(XIR_ATTRS_DECL_PRIMITIVE_VALUES)
    xir_attr_value_iterator_t* vec_value;
    xir_attr_value_map_iterator_t* map_value;
    xir_attr_value_map_iterator_t* map_vec_value;
  } u;
};

struct xir_attr_pair_t {
  xir_attr_value_t first;
  xir_attr_value_t second;
};

xir_attr_value_t xir2_attrs_get(xir_attrs_t attrs, xir_string_t key);
void xir2_attrs_set(xir_attrs_t attrs, xir_string_t key,
                    xir_attr_value_t value);

xir_attr_value_t xir2_attrs_keys(xir_attrs_t attrs);

/**
 * @brief Create a Tensor instance.
 *
 * @param name The name of the tensor.
 *
 * @param dims A array to indicate the tensor's dimensions.
 *
 * @param dim_num number of dimensions.
 *
 * @param data_type Indicates the type of the Tensor data.
 *
 * @param bit_width Indicates the bit width of the Tensor data.
 *
 * @return A unique pointer to the new Tensor object.
 */
xir_tensor_t xir_tensor_create(xir_string_t name, const int32_t* dims,
                               const int32_t dim_num,
                               enum xir_tensor_data_type_t data_type,
                               const int32_t bit_width);

/**
 * @brief destroy a tensor
 *
 * @return return value is not in use yet. it is always zero.
 */
int xir_tensor_destroy(xir_tensor_t tensor);

/**
 * @brief Get name of the tensor.
 * @param  tensor
 * @return The name of tensor .
 */
xir_string_t xir_tensor_get_name(xir_tensor_t tensor);
/**
 * @brief Get the bit width of the data in tensor.
 *
 * @return The bit width of the data in tensor.
 */
int32_t xir_tensor_get_bit_width(xir_tensor_t tensor);

/**
 * @brief Get the dimension size of one specific dimension indicated by idx.
 *
 * @param idx Indicate the dimension requested.
 *
 * @return The dimension size.
 */
int32_t xir_tensor_get_dim_size(xir_tensor_t tensor, int32_t idx);

/**
 * @brief Get the number of dimensions of the current Tensor object.
 *
 * @return The number of dimensions.
 */
int32_t xir_tensor_get_dim_num(xir_tensor_t tensor);

/**
 * @brief Get the data type of the tensor.
 *
 * @return Data type.
 */
enum xir_tensor_data_type_t xir_tensor_get_data_type(xir_tensor_t tensor);
/**
 * @brief Get the number of elements in the tensor.
 *
 * @return Number of elements.
 */
int64_t xir_tensor_get_element_num(xir_tensor_t tensor);
/**
 * @brief Get the number of data in the tensor.
 *
 * @return Number of data.
 */
uint64_t xir_tensor_get_data_size(xir_tensor_t tensor);
/**
 * @brief Get the Attrs object of the tensor.
 *
 * @return A unique pointer to the attrs
 */
xir_attrs_t xir_tensor_get_attrs(xir_tensor_t tensor);
/**
 * @brief Set the Attrs object of the tensor.
 */
void xir_tensor_set_attrs(xir_tensor_t tensor, xir_attrs_t attrs);

/**
 * @brief Get num of the keys in this Attrs object.
 *
 * @return num of keys
 */
size_t xir_attrs_get_num_of_keys(xir_attrs_t attrs);

/**
 * @brief Get a key
 *
 * @return the key
 */
const char* xir_attrs_get_key(xir_attrs_t attrs, size_t idx);
/**
 * @brief Get the name of subgraph.
 *
 * @return The name of the subgraph.
 */
xir_string_t xir_subgraph_get_name(xir_subgraph_t subgraph);

/**
 * @brief Set the name of subgraph.
 *
 * @param name The name of the subgraph.
 */
void xir_subgraph_set_name(xir_subgraph_t subgraph, xir_string_t subgraph_name);

/**
 * @brief Get the number of ops which belong to this subgraph.
 *
 * @return Number of ops.
 */
int32_t xir_subgraph_get_op_num(xir_subgraph_t subgraph);

/**
 * @brief Get all the ops which belong to this subgraph.
 *
 * @return A set of raw pointers.
 */
void xir_subgraph_get_ops(xir_subgraph_t subgraph, void* data,
                          xir_get_op_callback_t cb);

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
xir_op_t xir_subgraph_get_tensor_producer(xir_subgraph_t subgraph,
                                          xir_tensor_t tensor);

/**
 * @brief Get all the input tensors produced by other subgraph.
 *
 * @return A set of raw pointer to the input tensors.
 */
void xir_subgraph_get_input_tensors(xir_subgraph_t subgraph, void* data,
                                    xir_get_tensor_callback_t cb);

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
void xir_subgraph_get_output_tensors(xir_subgraph_t subgraph, void* data,
                                     xir_get_tensor_callback_t cb);

/**
 * @brief Check the existence of the op indicated by name.
 *
 * @param op_name The name of the op.
 *
 * @return True if exists, else false.
 */
int xir_subgraph_has_op_by_name(xir_subgraph_t subgraph, xir_string_t op_name);

/**
 * @brief Check the existence of the op indicated by a pointer of op.
 *
 * @param op A raw pointer of an op.
 *
 * @return True if exists, else false.
 */
int xir_subgraph_has_op(xir_subgraph_t subgraph, xir_op_t op);

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
xir_subgraph_t xir_subgraph_find_op_by_name(xir_subgraph_t subgraph,
                                            xir_string_t op_name);

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
xir_subgraph_t xir_subgraph_find_op(xir_subgraph_t subgraph, xir_op_t op);

/**
 * @brief Check if this subgraph is a root subgraph.
 *
 * @return True if it's the root, else false
 */
int xir_subgraph_is_root(xir_subgraph_t subgraph);

/**
 * @brief Check if this subgraph is a leaf subgraph.
 *
 * @return True if it's the leaf, else false.
 */
int xir_subgraph_is_leaf(xir_subgraph_t subgraph);

/**
 * @brief Get the root subgraph of the current subgraph.
 *
 * @return A raw pointer to the root subgraph.
 */
xir_subgraph_t xir_subgraph_get_root(xir_subgraph_t subgraph);

/**
 * @brief Get the depth of the current subgraph.
 *
 * @return The depth of the current subgraph.
 */
int32_t xir_subgraph_get_depth(xir_subgraph_t subgraph);

/**
 * @brief Get the parent subgraph of the current subgraph.
 *
 * @return A raw pointer to the parent subgraph.
 */
xir_subgraph_t xir_subgraph_get_parent(xir_subgraph_t subgraph);

/**
 * @brief Create children subgraph for the current subgraph.
 *
 * @details Create the children subgraph for the current subgraph while the
 * current subgraph is a leaf subgraph, if not, a fatal will of
 * XIR_SUBGRAPH_CREATE_CHILDREN_FOR_NONLEAF will be raised. And for the new
 * created children subgraphs, each of them only contains one op.
 */
void xir_subgraph_create_children(xir_subgraph_t subgraph);

/**
 * @brief Get the number of children subgraphs.
 *
 * @return The number of the children subgraphs.
 */
int32_t xir_subgraph_get_children_num(xir_subgraph_t subgraph);
/**
 * @brief Get the child subgraph of the current subgraph by idx.
 *
 * @return A raw pointer to a child subgraph.
 */
// xir_subgraph_t xir_subgraph_get_child(xir_subgraph_t subgraph, int32_t
// idx);

void xir_subgraph_get_children(xir_subgraph_t subgraph,
                               xir_subgraph_t children[]);

/**
 * @brief Check if the input subgraph is a child of the current subgraph.
 *
 * @param subgraph A pointer to the input subgraph's.
 *
 * @return True if is a child, else false.
 */
int xir_subgraph_is_child(xir_subgraph_t subgraph, xir_subgraph_t child);

/**
 * @brief Get the corresponding graph of the current subgraph.
 *
 * @return A raw pointer to the graph.
 */
xir_graph_t xir_subgraph_get_graph(xir_subgraph_t subgraph);

/**
 * @brief Check the existence of the Attrs object in current subgraph.
 *
 * @return True if exist, else false.
 */
int xir_subgraph_has_attrs(xir_subgraph_t subgraph);

/**
 * @brief Get all the attrs in the current subgraph.
 *
 * @return A unique pointer to the Attrs object.
 */
xir_attrs_t xir_subgraph_get_attrs(xir_subgraph_t subgraph);

/**
 * @brief Set an Attrs object to the current subgraph.
 *
 * @param attrs A unique pointer to the Attrs object to be set.
 */
void xir_subgraph_set_attrs(xir_subgraph_t subgraph, xir_attrs_t attrs);
/**
 * @brief Check the existence of the attribute indicated by key.
 *
 * @param key The attribute name.
 *
 * @return True if exist, else false.
 */
int xir_subgraph_has_attr(xir_subgraph_t subgraph, const char* key);

/**
 * @brief Get all the children subgraphs of the current subgraph in the
 * topological order.
 * @param subgraph  The current subgraph
 * @param children  All thr children subgraphs of the current subgraph int the
 * topological order.
 * @return A vector of the raw pointer of children subgraphs.
 */
void xir_subgraph_children_topological_sort(xir_subgraph_t subgraph,
                                            xir_subgraph_t children[]);
/**
 * @brief Save the subgraph into a dot file.
 *
 * @param file_name The file path of the file.
 */
void xir_subgraph_save_to_dot(xir_subgraph_t subgraph, const char* file_name);

/**
 * @brief Get name of the OP
 *
 * @return OP name
 */
xir_string_t xir_op_get_name(xir_op_t op);

/**
 * @brief Get type of the OP
 *
 * @return OP type
 */
xir_string_t xir_op_get_type(xir_op_t op);

/**
 * @brief Get total input number
 *
 * @return input number
 */
int xir_op_get_input_num(xir_op_t op);
/**
 * @brief Get input number with specific arg_name
 *
 * @param arg_name Specific argument name
 *
 * @return input number
 */
int xir_op_get_input_num_by_name(xir_op_t op, char* arg_name);

/**
 * @brief Get all input OPs with specific arg_name
 *
 * @param arg_name Specific argument name
 *
 * @return vector of input OPs, the order is guaranteed to be same as it was
 * set
 */
void xir_op_get_input_ops(xir_op_t op, xir_string_t arg_name, void* data,
                          xir_get_op_callback_t cb);

/**
 * @brief Get input OP with specific arg_name and index
 *
 * @param arg_name Specific argument name
 *
 * @param idx Index of the input OP. The default value of idx is 0
 *
 * @return input OP
 */
xir_op_t xir_op_get_input_op(xir_op_t op, xir_string_t arg_name, int idx);

/**
 * @brief Replace an op's specific input op.
 *
 * @param op_old A raw pointer to the input op to be replaced.
 *
 * @param op_new A raw pointer to the new input op.
 */
void xir_op_replace_input_op(xir_op_t op, xir_op_t old_op, xir_op_t new_op);

/**
 * @brief Get fan-out OP number
 *
 * @details XIR graph doesn't allow that an OP has more than one output
 * tensor, but there may be different OPs which take the output tensor as
 * their input. We call those OPs fan-out OP. This function return the number
 * of fan-out OPs.
 *
 * @return fan-out number
 */
int xir_op_get_fanout_num(xir_op_t op);

/**
 * @brief Get all fan-out OPs
 *
 * @return vector of fan-out OPs
 */
void xir_op_get_fanout_ops(xir_op_t op, void* data, xir_get_op_callback_t cb);

/**
 * @brief Get input tensor with specific arg_name and index
 *
 * @param arg_name Specific argument name
 *
 * @param idx Index of the input tensor. The default value of idx is 0
 *
 * @return input tensor
 */
xir_tensor_t xir_op_get_input_tensor(xir_op_t op, xir_string_t arg_name,
                                     int idx);

/**
 * @brief Get output tensor
 *
 * @return output tensor
 */
xir_tensor_t xir_op_get_output_tensor(xir_op_t op);

/**
 * @brief Replace the op's output tensor.
 *
 * @param tensor_new A unique pointer to the new output tensor.
 */
void xir_op_replace_output_tensor(xir_op_t op, xir_tensor_t tensor_new);

/**
 * @brief Get the the graph to which the op belongs.
 *
 * @return A raw pointer to the graph.
 */
xir_graph_t xir_op_get_graph(xir_op_t op);

/**
1 * @brief Check the existence of the Attrs object.
 *
 * @return If this op has Attrs, return true, else false.
 */
int xir_op_has_attrs(xir_op_t op);

/**
 * @brief Get a copy of OP attributes
 *
 * @return OP attributes
 */
xir_attrs_t xir_op_get_attrs(xir_op_t op);

/**
 * @brief Set OP attributes
 *
 * @param OP attributes Attrs object.
 */
xir_op_t xir_op_set_attrs(xir_op_t op, xir_attrs_t attrs);

/**
 * @brief Check the existence of the attribute indicated by key.
 *
 * @param key The attribute index name.
 *
 * @return If this op has this attribute return true, else false.
 */
int xir_op_has_attr(xir_op_t op, const char* key);

/**
 * @brief Inference the output tensor shape.
 */
void xir_op_shape_infer(xir_op_t op);

/** @brief return the op def */
xir_op_def_t xir_op_get_opdef(xir_op_t op);

/**
 * @brief Print the basic information of the Op.
 */
void xir_op_print_info(xir_op_t op);

typedef
    /**
     * @brief Element Occurence Specifier
     */
    enum xir_attr_def_occurence_type_t {
      /// Once and only once
      ATTR_REQUIRED,
      /// Never or once
      ATTR_OPTIONAL,

    } xir_attr_def_occurence_type_t;

/* @brief create a new attr arg def object */
xir_attr_def_t xir_attr_def_create(xir_string_t name,
                                   xir_type_index_t data_type1,
                                   xir_attr_def_occurence_type_t occurence_type,
                                   uint32_t list_length,
                                   xir_string_t annotation,
                                   xir_attr_value_t default_value);

/* @brief destroy an attr arg def object */
void xir_attr_def_destroy(xir_op_arg_def_t def);
xir_type_index_t XIR_TYPE_INDEX_BOOL();
xir_type_index_t XIR_TYPE_INDEX_INT8();
xir_type_index_t XIR_TYPE_INDEX_UINT8();
xir_type_index_t XIR_TYPE_INDEX_INT16();
xir_type_index_t XIR_TYPE_INDEX_UINT16();
xir_type_index_t XIR_TYPE_INDEX_INT32();
xir_type_index_t XIR_TYPE_INDEX_UINT32();
xir_type_index_t XIR_TYPE_INDEX_INT64();
xir_type_index_t XIR_TYPE_INDEX_UINT64();
xir_type_index_t XIR_TYPE_INDEX_FLOAT();
xir_type_index_t XIR_TYPE_INDEX_DOUBLE();
xir_type_index_t XIR_TYPE_INDEX_STRING();
xir_type_index_t XIR_TYPE_INDEX_BYTES();
xir_type_index_t XIR_TYPE_INDEX_BOOL_VEC();
xir_type_index_t XIR_TYPE_INDEX_INT8_VEC();
xir_type_index_t XIR_TYPE_INDEX_UINT8_VEC();
xir_type_index_t XIR_TYPE_INDEX_INT16_VEC();
xir_type_index_t XIR_TYPE_INDEX_UINT16_VEC();
xir_type_index_t XIR_TYPE_INDEX_INT32_VEC();
xir_type_index_t XIR_TYPE_INDEX_UINT32_VEC();
xir_type_index_t XIR_TYPE_INDEX_INT64_VEC();
xir_type_index_t XIR_TYPE_INDEX_UINT64_VEC();
xir_type_index_t XIR_TYPE_INDEX_FLOAT_VEC();
xir_type_index_t XIR_TYPE_INDEX_DOUBLE_VEC();
xir_type_index_t XIR_TYPE_INDEX_STRING_VEC();
xir_type_index_t XIR_TYPE_INDEX_BYTES_VEC();
xir_type_index_t XIR_TYPE_INDEX_MAP_STR_2_INT32();
xir_type_index_t XIR_TYPE_INDEX_MAP_STR_2_VEC_CHAR();
xir_type_index_t XIR_TYPE_INDEX_MAP_STR_2_STR();

/*
 *@struct OpArgDef
 *@brief Op argument definition
 *This struct defines an input argument of an op.
 */
typedef
    /**
     * @brief Element Occurence Specifier
     */
    enum xir_op_arg_def_occurence_type_t {
      /// Once and only once
      REQUIRED,
      /// Never or once
      OPTIONAL,
      /// No limitation
      REPEATED,
      /// At least once
      REQUIRED_AND_REPEATED,
      NUM
    } xir_op_arg_def_occurence_type_t;

/* @brief create a new op arg def object */
xir_op_arg_def_t xir_op_arg_def_create(
    xir_string_t name, xir_op_arg_def_occurence_type_t occurence_type,
    enum xir_tensor_data_type_t data_type, const int32_t bit_width,
    xir_string_t annotation);

/* @brief destroy an op arg def object */
void xir_op_arg_def_destroy(xir_op_arg_def_t opdef);

/* @brief get the name of a arg def object */
xir_string_t xir_op_arg_def_get_name(xir_op_arg_def_t self);

/* @brief create a new op def object */
xir_op_def_t xir_op_def_create(xir_string_t name);

/* @brief destroy an op def object */
void xir_op_def_destroy(xir_op_def_t opdef);

/* @brief get the name of a def object */
xir_string_t xir_op_def_get_name(xir_op_def_t self);

void xir_op_def_add_input_arg(xir_op_def_t self, xir_op_arg_def_t arg);
void xir_op_def_add_attr(xir_op_def_t self, xir_attr_def_t arg);
void xir_op_def_set_annotation(xir_op_def_t self, xir_string_t annotation);
typedef void (*op_callback_t)(void* self, xir_op_t);
void xir_op_def_set_shape_infer(xir_op_def_t self, op_callback_t fun,
                                void* data);
void xir_op_def_add_constraint(xir_op_def_t self, op_callback_t fun,
                               void* data);
#ifdef __cplusplus
}
#endif

/* Local Variables: */
/* mode:c */
/* c-basic-offset: 2 */
/* coding: undecided-unix */
/* End: */
