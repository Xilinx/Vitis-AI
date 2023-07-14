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
#include <fstream>
#include <memory>
#include <vitis/ai/env_config.hpp>
#include <vitis/dpurt/elf.hpp>
#include <xir/graph/graph.hpp>
DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0");

std::string g_xmodel_file_name = "test.xmodel";
std::string g_elf_file_name = "";
std::string g_kernel_prefix = "model";

static inline void usage(const char* progname) {
  std::cout << "usage: " << progname  //
            << " -t <num_of_threads>\n"
            << " <video file name>\n"
            << std::endl;
  return;
}

static inline void parse_opt(int argc, char* argv[]) {
  int opt = 0;
  optind = 1;
  while ((opt = getopt(argc, argv, "k:o:")) != -1) {
    switch (opt) {
      case 'o':
        g_elf_file_name = std::string(optarg);
        break;
      case 'k':
        g_kernel_prefix = std::string(optarg);
        break;
      default:
        usage(argv[0]);
        exit(1);
    }
  }
  if (optind < argc) {
    g_xmodel_file_name = std::string(argv[optind]);
  } else {
    std::cerr << "Expected argument after options\n";
    exit(EXIT_FAILURE);
  }
  return;
}
struct OutStream {
  OutStream(const std::string& filename) {
    fs_ = std::unique_ptr<std::ostream>{};
    if (!filename.empty()) {
      LOG(INFO) << "writing report to " << filename;
      fs_ = std::unique_ptr<std::ostream>{
          new std::ofstream(filename.c_str(), std::ofstream::out)};
    } else {
      LOG(INFO) << "writing report to <STDOUT>";
    }
  }
  std::ostream& get() {
    if (fs_) {
      return *fs_;
    }
    return std::cout;
  }
  std::unique_ptr<std::ostream> fs_;
};

static size_t get_kernel_io_size(const xir::Subgraph& subgraph) {
  CHECK(subgraph.has_attr("reg_id_to_context_type"));
  CHECK(subgraph.has_attr("reg_id_to_size"));
  auto reg_id_to_size =
      subgraph.get_attr<std::map<std::string, int>>("reg_id_to_size");
  auto reg_id_to_context_type =
      subgraph.get_attr<std::map<std::string, std::string>>(
          "reg_id_to_context_type");
  auto total = 0u;
  for (auto& reg : reg_id_to_context_type) {
    auto reg_id = reg.first;
    auto reg_type = reg.second;
    if (reg_type != "DATA") {
      continue;
    }
    auto it_size = reg_id_to_size.find(reg_id);
    CHECK(it_size != reg_id_to_size.end());
    auto size = it_size->second;
    total = total + size;
  }
  CHECK_GT(total, 0u) << "workspace size must not empty";
  return total;
}

static std::vector<char> load_parameter(const xir::Subgraph* subgraph_) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "loading parameter for " << subgraph_->get_graph()->get_name();
  CHECK(subgraph_->has_attr("reg_id_to_parameter_value"));
  auto reg_id_to_parameter_value =
      subgraph_->get_attr<std::map<std::string, std::vector<char>>>(
          "reg_id_to_parameter_value");

  CHECK(subgraph_->has_attr("reg_id_to_context_type"));
  auto reg_id_to_context_type =
      subgraph_->get_attr<std::map<std::string, std::string>>(
          "reg_id_to_context_type");
  std::vector<std::vector<char>> parameters;
  size_t total = 0;
  for (auto& reg : reg_id_to_context_type) {
    auto reg_id = reg.first;
    auto reg_type = reg.second;
    if (reg_type != "CONST") {
      continue;
    }
    auto it_value = reg_id_to_parameter_value.find(reg_id);
    CHECK(it_value != reg_id_to_parameter_value.end());
    total = total + it_value->second.size();
    parameters.emplace_back(std::move(it_value->second));
  }
  CHECK_EQ(parameters.size(), 1u) << "only support single parameter block.";

  return parameters[0];
}

enum RtTensorType {
  NormalTensor = (1 << 0),
  InputEdgeTensor = (1 << 1),
  OutputEdgeTensor = (1 << 2),
};

static vitis::dpurt::tensor convert_tensor(const xir::Tensor* tensor) {
  auto ret = vitis::dpurt::tensor{0u};
  CHECK_EQ(tensor->get_shape().size(), 4) << "only support 4 dims";
  ret.attr = 0u;
  ret.height = tensor->get_shape().at(1);
  ret.width = tensor->get_shape().at(2);
  ret.channel = tensor->get_shape().at(3);
  ret.addr_logical = tensor->get_attr<int32_t>("ddr_addr");
  ret.size = ret.height * ret.width * ret.channel;
  ret.fix_width = 8u;
  ret.fix_pos = tensor->get_attr<int32_t>("fix_point");
  ret.channel_stride = ret.channel;
  return ret;
}
static bool is_input_tensor(const xir::Subgraph* sg,
                            const xir::Tensor* tensor) {
  auto ret = false;
  for (auto t : sg->get_input_tensors()) {
    if (tensor == t) {
      ret = true;
    }
  }
  return ret;
}

static bool is_output_tensor(const xir::Subgraph* sg,
                             const xir::Tensor* tensor) {
  auto ret = false;
  for (auto t : sg->get_output_tensors()) {
    if (tensor == t) {
      ret = true;
    }
  }
  return ret;
}

static size_t build_tensor(const xir::Subgraph* sg, const xir::Tensor* tensor,
                           vitis::dpurt::DnncKernel* kernel) {
  auto location = tensor->get_attr<int32_t>("location");
  auto reg_id = (size_t)tensor->template get_attr<int>("reg_id");
  auto op = sg->get_graph()->get_tensor_producer(tensor);
  if (op->has_attr("fix_point")) {
    const_cast<xir::Tensor*>(tensor)->set_attr("fix_point",
                                               op->get_attr("fix_point"));
  } else if (op->has_attr("fix_point_output")) {
    const_cast<xir::Tensor*>(tensor)->set_attr(
        "fix_point", op->get_attr("fix_point_output"));
  } else {
    LOG(FATAL) << "cannot find fix info:"
               << "op_name " << op->get_name() << " "  //
        ;
  }
  if (location == 1 && reg_id == 1) {
    auto tmp_tensor = convert_tensor(tensor);
    if (is_input_tensor(sg, tensor)) {
      tmp_tensor.attr = InputEdgeTensor;
    } else if (is_output_tensor(sg, tensor)) {
      tmp_tensor.attr = OutputEdgeTensor;
    } else {
      tmp_tensor.attr = NormalTensor;
    }
    kernel->add_tensor(tmp_tensor);
    LOG(INFO) << "tensor: " << tensor->get_name() << " " << location;
  }
  return location == 1 ? 1u : 0u;
}

static void build_tensors(const xir::Subgraph* sg,
                          vitis::dpurt::DnncKernel* kernel) {
  auto inputs = sg->get_input_tensors();
  for (auto tensor : inputs) {
    build_tensor(sg, tensor, kernel);
  }
  for (auto op : sg->topological_sort()) {
    build_tensor(sg, op->get_output_tensor(), kernel);
  }
  kernel->build_tensor_section();
}

template <typename X>
void write(std::ostream& stream, const X& x) {
  CHECK(stream.write(reinterpret_cast<const char*>(&x), sizeof(x)).good());
}

enum RtNodeType { DpuCodeNode = 1, DpuVirtNode = 2, CpuNode = 0x80 };

static uint32_t get_node_type(const xir::Subgraph* layer) {  //
  return DpuCodeNode;
}

static uint32_t create_node_name(const xir::Subgraph* layer,
                                 vitis::dpurt::DnncKernel* kernel) {  //
  return kernel->deephi_allocate_string(layer->get_name());
}

static uint64_t get_node_workload(const xir::Subgraph* layer) {
  return 1976ull;
}

static void build_node_header(const xir::Subgraph* layer,
                              vitis::dpurt::DnncKernel* kernel,
                              std::ostream& stream) {
  write(stream, get_node_type(layer));
  write(stream, create_node_name(layer, kernel));
  write(stream, get_node_workload(layer));
}

static void build_node_regs(const xir::Subgraph* layer,
                            vitis::dpurt::DnncKernel* kernel,
                            std::ostream& stream) {
  uint32_t cnt = 4u;
  write(stream, cnt);
  write(stream, (uint16_t)2u);
  write(stream, (uint16_t)0u);
  write(stream, (uint16_t)0u);
  write(stream, (uint16_t)1u);
  write(stream, (uint16_t)3u);
  write(stream, (uint16_t)2u);
  write(stream, (uint16_t)1u);
  write(stream, (uint16_t)3u);
}

static void build_node_inputs(const xir::Subgraph* layer,
                              vitis::dpurt::DnncKernel* kernel,
                              std::ostream& stream) {
  auto tmp_inputs = layer->get_input_tensors();
  auto inputs =
      std::set<const xir::Tensor*>(tmp_inputs.begin(), tmp_inputs.end());
  uint32_t cnt = inputs.size();
  write(stream, cnt);
  LOG(INFO) << "dump " << layer->get_name();
  auto i = 0u;
  for (auto tensor : inputs) {
    LOG(INFO) << "input[" << i << "] : name= " << tensor->get_name();
    auto ddr_addr = tensor->get_attr<int32_t>("ddr_addr");
    auto input_tensor_id = kernel->find_tensor_by_ddr_addr(ddr_addr);
    write(stream, input_tensor_id);
    i = i + 1;
  }
}

static void build_node_outputs(const xir::Subgraph* layer,
                               vitis::dpurt::DnncKernel* kernel,
                               std::ostream& stream) {
  auto tmp_outputs = layer->get_output_tensors();
  auto outputs =
      std::set<const xir::Tensor*>(tmp_outputs.begin(), tmp_outputs.end());
  uint32_t cnt = outputs.size();
  write(stream, cnt);
  LOG(INFO) << "dump " << layer->get_name();
  auto i = 0u;
  for (auto tensor : outputs) {
    LOG(INFO) << "output[" << i << "] : name= " << tensor->get_name();
    auto ddr_addr = tensor->get_attr<int32_t>("ddr_addr");
    auto output_tensor_id = kernel->find_tensor_by_ddr_addr(ddr_addr);
    write(stream, output_tensor_id);
    i = i + 1;
  }
}

static void build_node_codes(const xir::Subgraph* layer,
                             vitis::dpurt::DnncKernel* kernel,
                             std::ostream& stream) {
  uint32_t cnt = 0u;
  write(stream, cnt);
}

static void build_node_params(const xir::Subgraph* layer,
                              vitis::dpurt::DnncKernel* kernel,
                              std::ostream& stream) {
  uint32_t cnt = 0u;
  write(stream, cnt);
}

static void build_node_pre_nodes(const xir::Subgraph* layer,
                                 vitis::dpurt::DnncKernel* kernel,
                                 std::ostream& stream) {
  uint32_t cnt = 0u;
  write(stream, cnt);
}

static void build_node_suc_nodes(const xir::Subgraph* layer,
                                 vitis::dpurt::DnncKernel* kernel,
                                 std::ostream& stream) {
  uint32_t cnt = 0u;
  write(stream, cnt);
}

static void build_node(const xir::Subgraph* layer,
                       vitis::dpurt::DnncKernel* kernel,  //
                       std::ostream& stream) {
  build_node_header(layer, kernel, stream);
  build_node_regs(layer, kernel, stream);
  build_node_inputs(layer, kernel, stream);
  build_node_outputs(layer, kernel, stream);
  build_node_codes(layer, kernel, stream);
  build_node_params(layer, kernel, stream);
  build_node_pre_nodes(layer, kernel, stream);
  build_node_suc_nodes(layer, kernel, stream);
}

static void build_node(const xir::Subgraph* layer,
                       vitis::dpurt::DnncKernel* kernel) {
  kernel->add_node([kernel, layer](std::ostream& stream) {
    build_node(layer, kernel, stream);
  });
}

static void build_nodes(const xir::Subgraph* sg,
                        vitis::dpurt::DnncKernel* kernel) {
  auto super_layers = sg->children_topological_sort();
  for (auto layer : super_layers) {
    build_node(layer, kernel);
  }
  kernel->set_num_of_nodes(super_layers.size());
}

static void convert_subgraph(const xir::Subgraph* sg,
                             vitis::dpurt::ElfBuilder* elf, size_t idx) {
  auto kernel_name = g_kernel_prefix + "_" + std::to_string(idx);
  auto kernel = std::make_unique<vitis::dpurt::DnncKernel>(kernel_name, elf);
  kernel->set_kernel_iosize(get_kernel_io_size(*sg));
  CHECK(sg->has_attr("mc_code"))
      << "sg->get_name() " << sg->get_name() << " "  //
      << "attrs: " << sg->get_attrs()->debug_info();
  auto& mc_code = sg->get_attr<std::vector<char>>("mc_code");
  kernel->build_code_section(mc_code);
  kernel->build_parameter_section(load_parameter(sg));
  build_tensors(sg, kernel.get());
  build_nodes(sg, kernel.get());
  kernel->build_meta_section();
}

static void convert_subgraphs(const xir::Graph* graph,
                              vitis::dpurt::ElfBuilder* elf) {
  auto root = graph->get_root_subgraph();
  auto children = root->children_topological_sort();
  auto idx = 0u;
  for (auto c : children) {
    CHECK(c->has_attr("device"));
    auto device = c->get_attr<std::string>("device");
    if (device == "DPU") {
      convert_subgraph(c, elf, idx);
      idx = idx + 1;
    }
  }
}

int main(int argc, char* argv[]) {
  parse_opt(argc, argv);
  auto out = std::make_unique<OutStream>(g_elf_file_name);
  auto graph = xir::Graph::deserialize(g_xmodel_file_name);
  auto elf = std::make_unique<vitis::dpurt::ElfBuilder>(out->get());
  convert_subgraphs(graph.get(), elf.get());
  elf->build();
  return 0;
}
