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

#include <iostream>
#include <vart/assistant/tensor_buffer_allocator.hpp>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vitis/ai/collection_helper.hpp>
#include <xir/graph/graph.hpp>
#include <xir/util/tool_function.hpp>

#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/runner_helper.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(USE_CPU_TASK, "1")
DEF_ENV_PARAM_2(MODE, "run", std::string)

int g_subgraph_index = 0;
std::string g_ref_dir = "ref";
std::string g_xmodel_file = "";
auto g_input_files = std::vector<std::string>();

static void usage() {
  std::cout << "usage: xlinx_test_dpu_runner <xmodel> [-i <subgraph_index>] "
               "<input_bin> [input_bin]... \n"
            << std::endl;
}

inline void parse_opt(int argc, char* argv[]) {
  int opt = 0;

  while ((opt = getopt(argc, argv, "i:h:r:")) != -1) {
    switch (opt) {
      case 'i':
        g_subgraph_index = std::stoi(optarg);
        break;
      case 'r':
        g_ref_dir = std::string(optarg);
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

static std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor) {
  auto idx = std::vector<int>(tensor->get_shape().size(), 0);
  return idx;
}

static std::string get_tensor_name(const xir::Tensor* tensor) {
  auto tensor_name = xir::remove_xfix(tensor->get_name());
  std::replace(tensor_name.begin(), tensor_name.end(), '/', '_');
  return tensor_name;
}

static std::string replace(const std::string& s) {
  const std::string pat = "/";
  std::ostringstream str;
  for (auto c : s) {
    if (pat.find(c) != std::string::npos) {
      str << "_";
    } else {
      str << c;
    }
  }
  return str.str();
}

static void fillin_input(const std::string refdir, vart::TensorBuffer* tb) {
  auto data = vart::get_tensor_buffer_data(tb, 0u);
  CHECK_EQ(data.size, tb->get_tensor()->get_data_size())
      << "must be continous tensor buffer";
  auto batch = tb->get_tensor()->get_shape()[0];
  for (auto b = 0; b < batch; ++b) {
    auto filename = refdir + "/" +
                    replace(xir::remove_xfix(tb->get_tensor()->get_name())) +
                    "_" + std::to_string(b) + ".bin";
    auto size_per_batch = data.size / batch;
    CHECK(std::ifstream(filename)
              .read((char*)data.data + size_per_batch * b, size_per_batch)
              .good())
        << "fail to read! filename=" << filename
        << ";tensor=" << tb->get_tensor()->get_name() << std::endl;
    LOG(INFO) << "read " << filename << " to " << data.data
              << " size=" << data.size;
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

std::vector<std::unique_ptr<vart::TensorBuffer>> create_tb(
    const std::vector<const xir::Tensor*>& tensors) {
  auto ret = std::vector<std::unique_ptr<vart::TensorBuffer>>();
  for (auto t : tensors) {
    // ret.emplace_back(std::make_unique<vart::mm::HostFlatTensorBuffer>(t));
    ret.emplace_back(vart::alloc_cpu_flat_tensor_buffer(t));
  }
  return ret;
}

int main(int argc, char* argv[]) {
  parse_opt(argc, argv);
  // get subgraph
  auto graph = xir::Graph::deserialize(g_xmodel_file);
  auto subgraph = graph->get_root_subgraph();

  if (!subgraph->has_attr("device")) {
    subgraph->set_attr<std::string>("device", "graph");
  }
  if (!subgraph->has_attr("runner")) {
    subgraph->set_attr<std::map<std::string, std::string>>(
        "runner", {{"ref", "libvitis_ai_library-graph_runner.so.3"},
                   {"sim", "libvitis_ai_library-graph_runner.so.3"},
                   {"run", "libvitis_ai_library-graph_runner.so.3"}});
  }
  if (g_subgraph_index >= 0) {
    auto all = subgraph->children_topological_sort();
    CHECK_LT((size_t)g_subgraph_index, all.size()) << "out of range";
    subgraph = all[g_subgraph_index];
  }

  if (ENV_PARAM(USE_CPU_TASK)) {
    if (subgraph->get_attr<std::string>("device") == "CPU") {
      subgraph->set_attr<std::map<std::string, std::string>>(
          "runner", {{"ref", "libvitis_ai_library-cpu_task.so.3"},
                     {"sim", "libvitis_ai_library-cpu_task.so.3"},
                     {"run", "libvitis_ai_library-cpu_task.so.3"}});
    }
  }
  auto runner = vart::Runner::create_runner(subgraph, ENV_PARAM(MODE));
  CHECK(runner != nullptr);
  // allocate tensor_buffers
  auto input_tensors = runner->get_input_tensors();
  auto output_tensors = runner->get_output_tensors();
  auto batch_size = input_tensors[0]->get_shape().at(0);

  auto input_tensor_buffers = create_tb(input_tensors);

  auto output_tensor_buffers = create_tb(output_tensors);

  // fillin input data
  for (auto& tb : input_tensor_buffers) {
    fillin_input(g_ref_dir, tb.get());
  }

  for (auto& input : input_tensor_buffers) {
    input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                 input->get_tensor()->get_shape()[0]);
  }
  // run dpu
  auto v = runner->execute_async(
      vitis::ai::vector_unique_ptr_get(input_tensor_buffers),
      vitis::ai::vector_unique_ptr_get(output_tensor_buffers));
  auto status = runner->wait((int)v.first, 1000000000);
  CHECK_EQ(status, 0) << "failed to run dpu";

  for (auto& output : output_tensor_buffers) {
    output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                 output->get_tensor()->get_shape()[0]);
  }

  dump_outputs(vitis::ai::vector_unique_ptr_get(output_tensor_buffers),
               batch_size);

  return 0;
}
