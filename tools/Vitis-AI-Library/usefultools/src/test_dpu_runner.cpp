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
#include <glog/logging.h>
#include <iostream>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vitis/ai/collection_helper.hpp>
#include <xir/graph/graph.hpp>
#include <xir/util/tool_function.hpp>

int g_subgraph_index = 0;
std::string g_xmodel_file = "";
auto g_input_files = std::vector<std::string>();
static void usage() {
  std::cout << "usage: xlinx_test_dpu_runner <xmodel> [-i <subgraph_index>] "
               "<input_bin> [input_bin]... \n"
            << std::endl;
}
inline void parse_opt(int argc, char* argv[]) {
  int opt = 0;

  while ((opt = getopt(argc, argv, "i:h:")) != -1) {
    switch (opt) {
      case 'i':
        g_subgraph_index = std::stoi(optarg);
        break;
      case 'h':
      default:
        usage();
        exit(1);
    }
  }
  if (optind >= argc) {
    usage();
    exit(1);
  }
  g_xmodel_file = argv[optind];
  for (auto i = optind + 1; i < argc; i++) {
    g_input_files.push_back(std::string(argv[i]));
  }
  return;
}

static xir::Subgraph* get_dpu_subgraph(xir::Graph* graph, const int idx) {
  auto root = graph->get_root_subgraph();
  xir::Subgraph* s = nullptr;
  int i = 0;
  for (auto c : root->children_topological_sort()) {
    if (c->get_attr<std::string>("device") == "DPU") {
      if (idx == i) {
        s = c;
        break;
      } else {
        i++;
      }
    }
  }
  return s;
}
static std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor) {
  auto idx = std::vector<int>(tensor->get_shape().size(), 0);
  return idx;
}

static std::string get_tensor_name(const xir::Tensor* tensor) {
  auto tensor_name = xir::remove_xfix(tensor->get_name());
  std::replace(tensor_name.begin(), tensor_name.end(), '/', '_');
  return tensor_name;
}

static void fillin_input(const std::string input_file, char* data,
                         int32_t size) {
  CHECK(std::ifstream(input_file).read((char*)data, size).good())
      << "fail to read! filename=" << input_file;
}

static void fillin_inputs(std::vector<std::string> input_files,
                          vart::TensorBuffer* input_tensor_buffer) {
  uint64_t data_in = 0u;
  size_t size_in = 0u;
  auto batch_size = input_tensor_buffer->get_tensor()->get_shape().at(0);
  auto size_per_batch =
      input_tensor_buffer->get_tensor()->get_data_size() / batch_size;
  for (auto batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    auto idx = get_index_zeros(input_tensor_buffer->get_tensor());
    idx[0] = batch_idx;
    std::tie(data_in, size_in) = input_tensor_buffer->data(idx);
    CHECK_NE(size_in, 0u);
    fillin_input(input_files[batch_idx % input_files.size()], (char*)data_in,
                 size_per_batch);
  }
}

static void dump_outputs(
    const std::vector<vart::TensorBuffer*>& output_tensor_buffers,
    const int batch_size) {
  auto mode = std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
  uint64_t data_out = 0u;
  size_t size_out = 0u;
  for (auto batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (auto i = 0u; i < output_tensor_buffers.size(); i++) {
      auto per_tensor_size =
          output_tensor_buffers[i]->get_tensor()->get_data_size() / batch_size;
      auto idx = get_index_zeros(output_tensor_buffers[i]->get_tensor());
      idx[0] = batch_idx;
      std::tie(data_out, size_out) = output_tensor_buffers[i]->data(idx);
      CHECK_NE(size_out, 0u);
      auto tensor_name =
          get_tensor_name(output_tensor_buffers[i]->get_tensor());
      auto output_tensor_file = std::to_string(batch_idx) + std::string(".") +
                                tensor_name + std::string(".bin");
      LOG(INFO) << "dump output to " << output_tensor_file;
      CHECK(std::ofstream(output_tensor_file, mode)
                .write((char*)data_out, per_tensor_size)
                .good())
          << " faild to write to " << output_tensor_file;
    }
  }
}
int main(int argc, char* argv[]) {
  parse_opt(argc, argv);

  // get subgraph
  auto graph = xir::Graph::deserialize(g_xmodel_file);
  auto subgraph = get_dpu_subgraph(graph.get(), g_subgraph_index);
  CHECK(subgraph != nullptr)
      << "cannot get subgraph[" << g_subgraph_index << "]";

  // create runner
  auto attrs = xir::Attrs::create();
  auto runner = vart::RunnerExt::create_runner(subgraph, attrs.get());

  // get input&output  tensor_buffers
  auto input_tensor_buffers = runner->get_inputs();
  auto output_tensor_buffers = runner->get_outputs();
  auto batch_size = input_tensor_buffers[0]->get_tensor()->get_shape().at(0);

  // fillin input data
  fillin_inputs(g_input_files, input_tensor_buffers[0]);

  for (auto& input : input_tensor_buffers) {
    input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                 input->get_tensor()->get_shape()[0]);
  }
  // run dpu
  auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);

  auto status = runner->wait((int)v.first, -1);
  CHECK_EQ(status, 0) << "failed to run dpu";

  for (auto& output : output_tensor_buffers) {
    output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                 output->get_tensor()->get_shape()[0]);
  }

  dump_outputs(output_tensor_buffers, batch_size);

  return 0;
}
