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

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <xir/xir.h>

int show_name(int argc, char* argv[]) {
  xir_graph_t g = xir_graph_deserialize(argv[2]);
  printf("graph name is %s\n", xir_graph_get_name(g));
  xir_graph_destroy(g);
  return 0;
}

int hello_attr(int argc, char* argv[]) {
  xir_attrs_t a = xir_attrs_create();
  xir_attrs_set_string(a, "hello", "world");
  if (xir_attrs_has_string(a, "hello")) {
    printf("has attrs(\"hello\") %s\n",
           xir_attrs_has_attr(a, "hello") ? "true" : "false");
  } else {
    assert(0 && "logical error: xir_attrs_has_string should return true");
  }
  printf("hello %s\n", xir_attrs_get_string(a, "hello"));
  /* test string array */
  xir_attrs_set_vec_string(a, "numbers", 0, "zero");
  // xir_attrs_set_vec_string(a, "numbers", 1, "one");
  xir_attrs_set_vec_string(a, "numbers", 2, "two");
  printf("has numbers? %s\n",
         xir_attrs_has_vec_string(a, "numbers") ? "true" : "false");
  size_t n_of_numbers = xir_attrs_get_vec_size_string(a, "numbers");
  for (size_t i = 0; i < n_of_numbers; ++i) {
    printf("numbers[%d] = %s\n", (int)i,
           xir_attrs_get_vec_string(a, "numbers", i));
  }
  /* test string mapping */
  xir_attrs_set_map_string(a, "month", "Jan", "Januray");
  xir_attrs_set_map_string(a, "month", "Feb", "Feberay");
  size_t n_of_months = xir_attrs_get_map_size_string(a, "month");
  const char* keys[n_of_months];
  xir_attrs_get_map_keys_string(a, "month", keys);
  for (size_t i = 0; i < n_of_months; ++i) {
    printf("%s => %s\n", keys[i],
           xir_attrs_get_map_string(a, "month", keys[i]));
  }
  /* test map_vec_string */
  xir_attrs_set_map_vec_string(a, "items", "colors", 0, "red");
  xir_attrs_set_map_vec_string(a, "items", "colors", 1, "green");
  xir_attrs_set_map_vec_string(a, "items", "colors", 2, "blue");
  xir_attrs_set_map_vec_string(a, "items", "shapes", 0, "circle");
  xir_attrs_set_map_vec_string(a, "items", "shapes", 1, "line");
  xir_attrs_set_map_vec_string(a, "items", "shapes", 2, "cross");
  printf("has items ? %s\n",
         xir_attrs_has_map_vec_string(a, "items") ? "true" : "false");
  size_t n_of_items = xir_attrs_get_map_vec_msize_string(a, "items");
  const char* item_keys[n_of_items];
  xir_attrs_get_map_vec_keys_string(a, "items", item_keys);
  for (size_t i = 0; i < n_of_items; ++i) {
    size_t n_of_x =
        xir_attrs_get_map_vec_vsize_string(a, "items", item_keys[i]);
    printf("%s => [", item_keys[i]);
    for (size_t j = 0; j < n_of_x; ++j) {
      if (j != 0) {
        printf(",");
      }
      printf("'%s'", xir_attrs_get_map_vec_string(a, "items", item_keys[i], j));
    }
    printf("]\n");
  }
  return 0;
}

int show_all_keys(int argc, char* argv[]) {
  xir_attrs_t a = xir_attrs_create();
  xir_attrs_set_string(a, "one", "1");
  xir_attrs_set_string(a, "two", "2");
  xir_attrs_set_string(a, "three", "3");
  xir_attrs_set_string(a, "four", "4");
  for (size_t i = 0u; i < xir_attrs_get_num_of_keys(a); ++i) {
    printf("%s => %s\n", xir_attrs_get_key(a, i),
           xir_attrs_get_string(a, xir_attrs_get_key(a, i)));
  }
  return 0;
}

xir_graph_t create_simple_graph() {
  xir_graph_t graph = xir_graph_create("graph_test");
  int data_dims[] = {1, 1, 1};
  xir_op_t data_op = xir_graph_add_op(
      graph, "data", "data", xir_attrs_create(), NULL, 0,
      xir_tensor_create("data", data_dims,
                        sizeof(data_dims) / sizeof(data_dims[0]), XIR_FLOAT,
                        32),
      NULL);
  (void)(data_op);
  return graph;
}

int simple_graph(int argc, char* argv[]) {
  xir_graph_t graph = create_simple_graph();
  printf("graph name is %s\n", xir_graph_get_name(graph));
  xir_graph_destroy(graph);
  return 0;
}

int show_tensor(int argc, char* argv[]) {
  int dims[] = {4, 480, 360, 3};
  xir_tensor_t tensor = xir_tensor_create(
      "tensor_test", dims, sizeof(dims) / sizeof(dims[0]), XIR_UINT, 8);
  printf("tensor name is %s\t tensor dims[ ", xir_tensor_get_name(tensor));
  int32_t dim_num = xir_tensor_get_dim_num(tensor);
  for (int32_t i = 0; i < dim_num; i++) {
    printf("%d ", xir_tensor_get_dim_size(tensor, i));
  }
  printf("]\t tensor bit_width %d\t tensor_data_type %d\n",
         xir_tensor_get_bit_width(tensor), xir_tensor_get_data_type(tensor));

  xir_tensor_destroy(tensor);
  return 0;
}

int show_subgraph_children(int argc, char* argv[]) {
  xir_graph_t g = xir_graph_deserialize(argv[2]);
  printf("graph name is %s\n", xir_graph_get_name(g));

  xir_subgraph_t root = xir_graph_get_root_subgraph(g);
  printf("root subgraph name is %s\n", xir_subgraph_get_name(root));

  int32_t children_num = xir_subgraph_get_children_num(root);
  printf("root subgraph children num is %d\n", children_num);

  for (int32_t i = 0; i < children_num; i++) {
    xir_subgraph_t subgraph = xir_subgraph_get_child(root, i);
    printf("children %d name is %s\n", i, xir_subgraph_get_name(subgraph));
  }
  printf("=== topological_sort_children===\n");
  xir_subgraph_t children[children_num];
  xir_subgraph_children_topological_sort(root, children);
  for (int32_t i = 0; i < children_num; i++) {
    printf("children %d name is %s\n", i, xir_subgraph_get_name(children[i]));
  }
  // free(children);
  xir_graph_destroy(g);
  return 0;
}

int main(int argc, char* argv[]) {
  const char* test_case = argv[1];
  if (strcmp(test_case, "show_name") == 0) {
    return show_name(argc, argv);
  } else if (strcmp(test_case, "hello_attr") == 0) {
    return hello_attr(argc, argv);
  } else if (strcmp(test_case, "show_all_keys") == 0) {
    return show_all_keys(argc, argv);
  } else if (strcmp(test_case, "simple_graph") == 0) {
    return simple_graph(argc, argv);
  } else if (strcmp(test_case, "show_tensor") == 0) {
    return show_tensor(argc, argv);
  } else if (strcmp(test_case, "show_subgraph_children") == 0) {
    return show_subgraph_children(argc, argv);
  } else {
    printf("unknown test case %s\n", test_case);
  }
  return 0;
}

/* Local Variables: */
/* mode:c */
/* c-basic-offset: 2 */
/* coding: utf-8-unix */
/* End: */
