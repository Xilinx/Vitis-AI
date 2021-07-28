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
  const char* name;
  xir_op_t* ops;
  size_t num_of_ops;
};

xir_op_t xir_graph_add_op(xir_graph_t graph,                        //
                          const char* name,                         //
                          const char* type,                         //
                          xir_attrs_t attrs,                        //
                          struct xir_graph_input_ops_t* input_ops,  //
                          size_t num_of_ops,                        //
                          xir_tensor_t tensor,                      //
                          xir_subgraph_t subgraph                   //
);

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
typedef const char *xir_string_t;
typedef struct {
   int8_t * data;
   size_t size;
} xir_bytes_t;
#define XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(p)      \
  p(bool, int, bool)                                \
  p(i8, int8_t, int8_t)                             \
  p(i16, int16_t, int16_t)                          \
  p(i32, int32_t, int32_t)                          \
  p(i64, int64_t, int64_t)                          \
  p(u8,  uint8_t,  uint8_t)                         \
  p(u16, uint16_t, uint16_t)                        \
  p(u32, uint32_t, uint32_t)                        \
  p(u64, uint64_t, uint64_t)                        \
  p(f32, float, float)                              \
  p(f64, double, double)                            \
  p(string, xir_string_t, std::string)               \
  p(bytes, xir_bytes_t, std::vector<int8_t>)
// p(type_name, c_type, c++ type )
#define DECLARE_XIR_ATTRS_GET(type_name, c_type, cxx_type) \
c_type xir_attrs_get_##type_name (xir_attrs_t attrs, const char* name);
#define DECLARE_XIR_ATTRS_SET(type_name, c_type, cxx_type) \
void xir_attrs_set_##type_name (xir_attrs_t attrs, const char* name, c_type value);
#define DECLARE_XIR_ATTRS_HAS_ATTRS(type_name, c_type, cxx_type) \
int xir_attrs_has_##type_name (xir_attrs_t attrs, const char* name);
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES (DECLARE_XIR_ATTRS_GET)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES (DECLARE_XIR_ATTRS_SET)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES (DECLARE_XIR_ATTRS_HAS_ATTRS)

#define DECLARE_XIR_ATTRS_GET_VEC_SIZE(type_name, c_type, cxx_type) \
size_t xir_attrs_get_vec_size_ ##type_name (xir_attrs_t attrs, const char* name);
#define DECLARE_XIR_ATTRS_GET_VEC(type_name, c_type, cxx_type)          \
c_type xir_attrs_get_vec_ ##type_name (xir_attrs_t attrs, const char* name, size_t idx);
#define DECLARE_XIR_ATTRS_SET_VEC(type_name, c_type, cxx_type) \
  void xir_attrs_set_vec_ ##type_name (xir_attrs_t attrs, const char* name, size_t idx, c_type value);
#define DECLARE_XIR_ATTRS_HAS_VEC(type_name, c_type, cxx_type) \
int xir_attrs_has_vec_ ##type_name (xir_attrs_t attrs, const char* name);

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_GET_VEC_SIZE)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_GET_VEC)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_SET_VEC)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_HAS_VEC)

#define DECLARE_XIR_ATTRS_GET_MAP_SIZE(type_name, c_type, cxx_type)          \
size_t xir_attrs_get_map_size_ ##type_name (xir_attrs_t attrs, const char* name);
#define DECLARE_XIR_ATTRS_GET_MAP_KEYS(type_name, c_type, cxx_type)          \
void xir_attrs_get_map_keys_ ##type_name (xir_attrs_t attrs, const char* name,  const char *keys[]);
#define DECLARE_XIR_ATTRS_GET_MAP(type_name, c_type, cxx_type)          \
c_type xir_attrs_get_map_ ##type_name (xir_attrs_t attrs, const char* name, const char * key);
#define DECLARE_XIR_ATTRS_SET_MAP(type_name, c_type, cxx_type) \
  void xir_attrs_set_map_ ##type_name (xir_attrs_t attrs, const char* name, const char * key, c_type value);
#define DECLARE_XIR_ATTRS_HAS_MAP(type_name, c_type, cxx_type) \
int xir_attrs_has_map_ ##type_name (xir_attrs_t attrs, const char* name);

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_GET_MAP_SIZE)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_GET_MAP_KEYS)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_GET_MAP)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_SET_MAP)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_HAS_MAP)

#define DECLARE_XIR_ATTRS_GET_MAP_VEC_MSIZE(type_name, c_type, cxx_type)          \
size_t xir_attrs_get_map_vec_msize_ ##type_name (xir_attrs_t attrs, const char* name);
#define DECLARE_XIR_ATTRS_GET_MAP_VEC_VSIZE(type_name, c_type, cxx_type)          \
size_t xir_attrs_get_map_vec_vsize_ ##type_name (xir_attrs_t attrs, const char* name, const char * key);
#define DECLARE_XIR_ATTRS_GET_MAP_VEC_KEYS(type_name, c_type, cxx_type)          \
void xir_attrs_get_map_vec_keys_ ##type_name (xir_attrs_t attrs, const char* name,  const char *keys[]);
#define DECLARE_XIR_ATTRS_GET_MAP_VEC(type_name, c_type, cxx_type)          \
c_type xir_attrs_get_map_vec_ ##type_name (xir_attrs_t attrs, const char* name, const char * key, size_t idx);
#define DECLARE_XIR_ATTRS_SET_MAP_VEC(type_name, c_type, cxx_type) \
void xir_attrs_set_map_vec_ ##type_name (xir_attrs_t attrs, const char* name, const char * key, size_t idx, c_type value);
#define DECLARE_XIR_ATTRS_HAS_MAP_VEC(type_name, c_type, cxx_type) \
int xir_attrs_has_map_vec_ ##type_name (xir_attrs_t attrs, const char* name);

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_GET_MAP_VEC_MSIZE)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_GET_MAP_VEC_VSIZE)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_GET_MAP_VEC_KEYS)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_GET_MAP_VEC)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_SET_MAP_VEC)
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(DECLARE_XIR_ATTRS_HAS_MAP_VEC)

// clang-format on
/**
 * @brief Check the existence of the attribute indicated by key.
 *
 * @param key A string to indicate the key of the attribute to check.
 *
 * @return 1 for existing, otherwise for not.
 */
int xir_attrs_has_attr(xir_attrs_t attrs, const char* key);
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
xir_tensor_t xir_tensor_create(const char* name, const int32_t* dims,
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
const char* xir_tensor_get_name(xir_tensor_t tensor);
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
int32_t xir_tensor_get_element_num(xir_tensor_t tensor);
/**
 * @brief Get the number of data in the tensor.
 *
 * @return Number of data.
 */
int32_t xir_tensor_get_data_size(xir_tensor_t tensor);
/**
 * @brief Get the Attrs object of the tensor.
 *
 * @return A unique pointer to the attrs
 */
xir_attrs_t xir_tensor_get_attrs(xir_tensor_t tensor);

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
const char* xir_subgraph_get_name(xir_subgraph_t subgraph);
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
xir_subgraph_t xir_subgraph_get_child(xir_subgraph_t subgraph, int32_t idx);

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
#ifdef __cplusplus
}
#endif

/* Local Variables: */
/* mode:c */
/* c-basic-offset: 2 */
/* coding: undecided-unix */
/* End: */
