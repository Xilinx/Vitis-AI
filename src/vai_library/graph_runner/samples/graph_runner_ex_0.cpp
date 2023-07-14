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
#include <glog/logging.h>

#include <fstream>
#include <iostream>

#include "vart/runner_helper.hpp"
#include "vitis/ai/graph_runner.hpp"

std::string g_xmodel_file = "";
std::map<std::string, std::string> g_tensor_map;
auto g_input_files = std::vector<std::string>();

static void usage() {
  std::cout
      << "usage: graph_runner_ex_0 -m <xmodel>\n"
         "       -t <tensor_name>:<file_name> there are might many pairs.\n"
         "       -h for help"
      << std::endl;
}

std::pair<const std::string, std::string> to_pair(const std::string& name) {
  auto pos = name.find(':');
  CHECK(pos != std::string::npos) << "cannot find colon: name=" << name;
  return std::make_pair<const std::string, std::string>(name.substr(0, pos),
                                                        name.substr(pos + 1));
}

inline void parse_opt(int argc, char* argv[]) {
  int opt = 0;
  while ((opt = getopt(argc, argv, "m:t:h")) != -1) {
    switch (opt) {
      case 'm':
        g_xmodel_file = optarg;
        break;
      case 't':
        g_tensor_map.insert(to_pair(optarg));
        break;
      case 'h':
      default:
        usage();
        exit(1);
    }
  }
  return;
}

typedef struct {
  void* data;
  size_t size;
} tensor_buffer_data_t;

tensor_buffer_data_t get_tensor_buffer_data(vart::TensorBuffer* tensor_buffer,
                                            int batch_index) {
  tensor_buffer_data_t ret;
  auto idx = vart::get_index_zeros(tensor_buffer->get_tensor());
  idx[0] = batch_index;
  uint64_t data;
  std::tie(data, ret.size) = tensor_buffer->data(idx);
  ret.data = (void*)data;
  return ret;
}

std::string tensor_file(vart::TensorBuffer* tensor_buffer) {
  auto it = g_tensor_map.find(tensor_buffer->get_tensor()->get_name());
  CHECK(it != g_tensor_map.end())
      << "cannot find tensor file for " << tensor_buffer->to_string();
  return it->second;
}

static void read_input(vart::TensorBuffer* input) {
  auto filename = tensor_file(input);
  auto batch = input->get_tensor()->get_shape()[0];
  auto size = input->get_tensor()->get_data_size() / batch;
  for (auto i = 0; i < batch; ++i) {
    auto data = get_tensor_buffer_data(input, i);
    CHECK(std::ifstream(filename).read((char*)data.data, size).good())
        << "fail to read! filename=" << filename
        << ";tensor=" << input->get_tensor()->get_name() << std::endl;
    LOG(INFO) << "read " << filename << " to " << data.data << " size=" << size;
  }
}

static void read_inputs(std::vector<vart::TensorBuffer*>& inputs) {
  for (auto i : inputs) {
    read_input(i);
  }
}

static void dump_output(vart::TensorBuffer* output) {
  auto filename = tensor_file(output);
  auto batch = output->get_tensor()->get_shape()[0];
  auto size = output->get_tensor()->get_data_size() / batch;
  for (auto i = 0; i < batch; ++i) {
    auto data = get_tensor_buffer_data(output, i);
    CHECK(std::ofstream(filename).write((char*)data.data, size).good())
        << "fail to write! filename=" << filename
        << ";tensor=" << output->get_tensor()->get_name() << std::endl;
    LOG(INFO) << "write " << filename << " to " << data.data
              << " size=" << size;
  }
}

static void dump_outputs(std::vector<vart::TensorBuffer*>& outputs) {
  for (auto o : outputs) {
    dump_output(o);
  }
}

// clang-format off
/*
 example:
 graph_runner_ex_0 -m /usr/share/vitis_ai_library/models/resnet_v1_50_tf/resnet_v1_50_tf.xmodel \
 -t input/aquant:dump/subgraph_resnet_v1_50_block1_unit_1_bottleneck_v1_add/input/0.input_aquant.bin \
 -t resnet_v1_50/predictions/Reshape_1/aquant:dump/subgraph_resnet_v1_50_block1_unit_1_bottleneck_v1_add/output/0.resnet_v1_50_predictions_Reshape_aquant.bin
*/
// clang-format on

int main(int argc, char* argv[]) {
  parse_opt(argc, argv);

  //create graph runner
  auto graph = xir::Graph::deserialize(g_xmodel_file);
  auto attrs = xir::Attrs::create();  // for future
  auto runner =
      vitis::ai::GraphRunner::create_graph_runner(graph.get(), attrs.get());
  CHECK(runner != nullptr);

  //get input/output tensor buffers
  auto inputs = runner->get_inputs();
  auto outputs = runner->get_outputs();

  //fillin inputs
  read_inputs(inputs);

  //sync input tensor buffers
  for (auto& input : inputs) {
      input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                   input->get_tensor()->get_shape()[0]);
    }

  //run graph runner
  auto v = runner->execute_async(inputs, outputs);
  auto status = runner->wait((int)v.first, -1);
  CHECK_EQ(status, 0) << "failed to run the graph";

  //sync output tensor buffers
  for (auto output : outputs) {
      output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                   output->get_tensor()->get_shape()[0]);
    }

  //dump outputs
  dump_outputs(outputs);
  return 0;
}
