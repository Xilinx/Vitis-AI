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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vart/vart.h>
static xir_subgraph_t find_subgraph(xir_graph_t graph, const char* name) {
  xir_graph_t ret = NULL;
  xir_subgraph_t root = xir_graph_get_root_subgraph(graph);
  int num_of_children = xir_subgraph_get_children_num(root);
  xir_graph_t children[num_of_children];
  xir_subgraph_children_topological_sort(root, children);
  for (int i = 0; i < num_of_children; ++i) {
    xir_string_t subgraph_name = xir_subgraph_get_name(children[i]);
    if (strncmp(name, subgraph_name.data, subgraph_name.size) == 0) {
      ret = children[i];
    }
  }
  if (ret == NULL) {
    fprintf(stderr, "cannot find subgraph %s\n", name);
    abort();
  }
  return ret;
}

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int main(int argc, char* argv[]) {
  const char* filename = argv[1];
  const char* kernel = argv[2];
  const char* input_file = argv[3];
  int runner_num = atoi(argv[4]);
  runner_num = MIN(100, MAX(0, runner_num));
  int count = atoi(argv[5]);
  count = MIN(100, MAX(0, count));
  vart_runner_t runners[count];
  xir_graph_t graph = xir_graph_deserialize(filename);
  xir_subgraph_t subgraph = find_subgraph(graph, kernel);
  for (int rr = 0; rr < runner_num; rr++) {
    runners[rr] = vart_create_runner(subgraph, "run");
  }
  for (int rr = 0; rr < runner_num; rr++) {
    vart_runner_t runner = runners[rr];
    int num_of_inputs = vart_runner_get_num_of_input_tensors(runner);
    vart_tensor_buffer_t inputs[num_of_inputs];
    vart_runner_get_inputs(runner, inputs);
    assert(num_of_inputs == 1);
    int num_of_outputs = vart_runner_get_num_of_output_tensors(runner);
    vart_tensor_buffer_t outputs[num_of_outputs];
    vart_runner_get_outputs(runner, outputs);
    size_t batch_size =
        xir_tensor_get_dim_size(vart_tensor_buffer_get_tensor(inputs[0]), 0);
    size_t size_per_batch =
        xir_tensor_get_data_size(vart_tensor_buffer_get_tensor(inputs[0])) /
        batch_size;
    for (size_t i = 0u; i < batch_size; ++i) {
      int dims[4] = {(int)i, 0, 0, 0};
      vart_tensor_buffer_address_t addr =
          vart_tensor_buffer_data(inputs[0], dims, 4);
      FILE* fp = fopen(input_file, "rb");
      if (!fp) {
        fprintf(stderr, "cannot open file %s\n", input_file);
      }
      size_t read_result =
          fread((void*)(uintptr_t)addr.addr, size_per_batch, 1u, fp);
      if (read_result <= 0u) {
        fprintf(stderr, "fread fail ! read_result is %d\n", (int)read_result);
      }
      assert(read_result > 0u);
      fclose(fp);
    }
    for (int i = 0; i < count; ++i) {
      vart_job_id_and_status_t r = vart_runner_execute_async(
          runner, inputs, num_of_inputs, outputs, num_of_outputs);
      vart_runner_wait(runner, r.job_id, 0);
    }
  }
  for (int rr = 0; rr < runner_num; rr++) {
    vart_destroy_runner(runners[rr]);
  }
  return 0;
}
