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
#include <experimental/xrt-next.h>
#include <glog/logging.h>
#include <google/protobuf/message.h>
#include <unistd.h>
#include <xrt.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vitis/ai/env_config.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/tool_function.hpp>
#include <xir/xrt_device_handle.hpp>

#include "vart/runner_ext.hpp"
#include "xir/buffer_object.hpp"
DEF_ENV_PARAM(DEVICE_ID, "0");
DEF_ENV_PARAM_2(CU_NAME, "DPU", std::string);
DEF_ENV_PARAM(CU_INDEX, "0");
DEF_ENV_PARAM(REC_INSTR, "0");
DEF_ENV_PARAM(START_INSTR, "0");
DEF_ENV_PARAM(STOP_INSTR, "0");
DEF_ENV_PARAM(DEBUG_XVDPU_PROFILER, "0");
using namespace std;

int g_subgraph_index = 1;
std::string g_xmodel_file = "";
std::string g_start_file = "";
auto g_input_files = std::vector<std::string>();

typedef struct {
  uint64_t size;
  uint64_t single_batch_size;
  size_t batch_size;
  std::vector<size_t> batch_num;
} bo_size_t;

static std::vector<std::string> instr{"all", "load", "save", "conv", "misc"};

template <typename T>
uint64_t data_slice(T source, int begin, int end) {
  return (source >> begin) & ((T(1) << int(end - begin)) - 1);
}

static uint64_t align(uint64_t a, uint64_t b) {
  if (a % b == 0) {
    return a;
  }
  return (a / b + 1) * b;
}

static int xrtXclRead(xclDeviceHandle handle, uint32_t ipIndex, uint64_t offset,
                      uint64_t offsetbase, uint32_t* datap) {
  return xclRegRead(handle, ipIndex, offset, datap);
}

static int xrtXclWrite(xclDeviceHandle handle, uint32_t ipIndex,
                       uint64_t offset, uint64_t offsetbase, uint32_t data) {
  return xclRegWrite(handle, ipIndex, offset, data);
}

static void set_reg(xclDeviceHandle xcl_handle, uint32_t ip_index,
                    uint64_t cu_base_addr, uint32_t offset, uint32_t value) {
  auto read_result =
      xrtXclWrite(xcl_handle, ip_index, offset, cu_base_addr, value);

  CHECK_EQ(read_result, 0) << "xrtXclWrite has error!";
}

static uint32_t get_reg(xclDeviceHandle xcl_handle, uint32_t ip_index,
                        uint64_t cu_base_addr, uint32_t offset) {
  uint32_t value = 0;
  auto read_result =
      xrtXclRead(xcl_handle, ip_index, offset, cu_base_addr, &value);

  CHECK_EQ(read_result, 0) << "xrtXclRead has error!";
  return value;
}

static bo_size_t get_bo_size(const std::string& cu_name, size_t index) {
  bo_size_t ret;
  auto h = xir::XrtDeviceHandle::get_instance();
  auto reg_44 =
      get_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
              h->get_cu_addr(cu_name, index), 0x44);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XVDPU_PROFILER))
      << "reg:0x44 : " << std::hex << "0x" << reg_44;
  auto batch_en = data_slice(reg_44, 0, 8);
  auto batch_size = 0;
  // batch_en is one-hot, up to 8 batch.
  for (auto i = 0; i < 8; i++) {
    if (batch_en & 0x1) {
      batch_size += 1;
      ret.batch_num.push_back(i);
    }
    batch_en = batch_en >> 1;
  }
  CHECK(batch_size != 0) << "no profiler enable!";
  for (auto b : ret.batch_num) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XVDPU_PROFILER))
        << "profiler enable : batch " << b;
  }
  auto profiler_depth = data_slice(reg_44, 16, 20);
  // single_batch_size = 4*2^(profiler_depth + 5)*4 bytes
  //                   = 2^(profiler_depth + 9) bytes
  uint64_t single_batch_size = 1 << (profiler_depth + 9);
  auto size = single_batch_size * batch_size;
  ret.size = align(size, 1024u);
  ret.single_batch_size = single_batch_size;
  ret.batch_size = batch_size;

  LOG_IF(INFO, ENV_PARAM(DEBUG_XVDPU_PROFILER))
      << "bo_size: " << std::hex << "0x" << ret.size
      << "   single_batch_size: " << std::hex << "0x" << ret.single_batch_size
      << "   batch_size: " << std::dec << "0x" << ret.batch_size;

  return ret;
}

static void init_profiler_reg(const std::string& cu_name, size_t index,
                              uint64_t phy_addr, uint64_t single_batch_size,
                              size_t rec_instr, size_t start_instr) {
  auto h = xir::XrtDeviceHandle::get_instance();
  // 0x58: PROF_DDR_ADDR_L
  set_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
          h->get_cu_addr(cu_name, index), 0x58, phy_addr & 0xFFFFFFFF);
  // 0x5C: PROF_DDR_ADDR_H
  set_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
          h->get_cu_addr(cu_name, index), 0x5C, (phy_addr >> 32) & 0xFFFFFFFF);
  // 0xE8: PROF_DDR_JUMP_L
  set_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
          h->get_cu_addr(cu_name, index), 0xE8, single_batch_size & 0xFFFFFFFF);
  // 0xEC: PROF_DDR_JUMP_H
  set_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
          h->get_cu_addr(cu_name, index), 0xEC,
          (single_batch_size >> 32) & 0xFFFFFFFF);
  // 0xE0: [7:4] PROF_REC_INSTR; [3:0] PROF_UP_MODE
  set_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
          h->get_cu_addr(cu_name, index), 0xE0,
          ((rec_instr << 4) + 1) & 0xFFFFFFFF);
  // 0xE4: PROF_START_INSTR
  set_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
          h->get_cu_addr(cu_name, index), 0xE4, start_instr & 0xFFFFFFFF);
  if (ENV_PARAM(DEBUG_XVDPU_PROFILER) >= 1) {
    auto reg_58 =
        get_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
                h->get_cu_addr(cu_name, index), 0x58);
    auto reg_5c =
        get_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
                h->get_cu_addr(cu_name, index), 0x5C);
    auto reg_e8 =
        get_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
                h->get_cu_addr(cu_name, index), 0xE8);
    auto reg_ec =
        get_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
                h->get_cu_addr(cu_name, index), 0xEC);
    auto reg_e0 =
        get_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
                h->get_cu_addr(cu_name, index), 0xE0);
    auto reg_e4 =
        get_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
                h->get_cu_addr(cu_name, index), 0xE4);
    LOG(INFO) << "reg:0x58 : " << std::hex << "0x" << reg_58;
    LOG(INFO) << "reg:0x5C : " << std::hex << "0x" << reg_5c;
    LOG(INFO) << "reg:0xE8 : " << std::hex << "0x" << reg_e8;
    LOG(INFO) << "reg:0xEC : " << std::hex << "0x" << reg_ec;
    LOG(INFO) << "reg:0xE0 : " << std::hex << "0x" << reg_e0;
    LOG(INFO) << "reg:0xE4 : " << std::hex << "0x" << reg_e4;
  }
}

static std::string get_tensor_name(const xir::Tensor* tensor) {
  auto tensor_name = xir::remove_xfix(tensor->get_name());
  std::replace(tensor_name.begin(), tensor_name.end(), '/', '_');
  return tensor_name;
}

static void run_dpu(vart::Runner* runner,
                    const std::vector<std::string>& input_files) {
  auto r = dynamic_cast<vart::RunnerExt*>(runner);
  CHECK(r != nullptr) << "not a dpu runner";
  auto input = r->get_inputs();
  auto output = r->get_outputs();
  size_t batch_size = input[0]->get_tensor()->get_shape()[0];
  for (auto input_idx = 0u; input_idx < input.size(); ++input_idx) {
    for (auto i = 0u; i < batch_size; ++i) {
      auto dims =
          vector<int>(input[input_idx]->get_tensor()->get_shape().size(), 0);
      dims[0] = (int)i;
      uint64_t input_data = 0u;
      auto input_size = 0u;
      auto size_per_batch =
          input[input_idx]->get_tensor()->get_data_size() / batch_size;
      std::tie(input_data, input_size) = input[input_idx]->data(dims);
      CHECK(std::ifstream(input_files[input_idx])
                .read((char*)input_data, size_per_batch)
                .good())
          << "fail to read! filename=" << input_files[input_idx];
    }
  }

  for (auto in : input) {
    in->sync_for_write(0, in->get_tensor()->get_data_size() /
                              in->get_tensor()->get_shape()[0]);
  }

  runner->execute_async(input, output);
  runner->wait(0, 0);
  for (auto out : output) {
    out->sync_for_read(0, out->get_tensor()->get_data_size() /
                              out->get_tensor()->get_shape()[0]);
  }
  if (ENV_PARAM(DEBUG_XVDPU_PROFILER) >= 2) {
    auto mode =
        std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
    for (auto output_idx = 0u; output_idx < output.size(); ++output_idx) {
      for (auto i = 0u; i < batch_size; ++i) {
        auto dims = vector<int>(
            output[output_idx]->get_tensor()->get_shape().size(), 0);
        dims[0] = (int)i;
        uint64_t output_data = 0u;
        auto output_size = 0u;
        auto size_per_batch =
            output[output_idx]->get_tensor()->get_data_size() / batch_size;
        std::tie(output_data, output_size) = output[output_idx]->data(dims);
        auto tensor_name = get_tensor_name(output[output_idx]->get_tensor());
        auto output_tensor_file = std::to_string(i) + std::string(".") +
                                  tensor_name + std::string(".bin");
        CHECK(std::ofstream(output_tensor_file, mode)
                  .write((char*)output_data, size_per_batch)
                  .good())
            << "failed to write to " << output_tensor_file;
      }
    }
  }
}

static bool check_finish_reg(const std::string& cu_name, size_t index) {
  auto h = xir::XrtDeviceHandle::get_instance();
  auto reg_44 =
      get_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
              h->get_cu_addr(cu_name, index), 0x44);
  return ((reg_44 & 0x00F00000) == 0x0);
}

// dump profiler_file, return instr end or not
static bool dump_profiler_file(xir::BufferObject* bo, bo_size_t bo_size,
                               size_t rec_instr, size_t start_instr,
                               const std::string& name = "") {
  auto buf = std::vector<char>(bo_size.size);
  bo->copy_to_host(&buf[0], bo_size.size, 0);
  auto mode = std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
  for (size_t i = 0; i < bo_size.batch_size; i++) {
    auto output_filename =
        std::string("profiler_") + name + std::string("_batch_") +
        std::to_string(bo_size.batch_num[i]) + std::string("_instr_") +
        instr[rec_instr] + std::string("_start_") +
        std::to_string(start_instr) + std::string(".bin");
    LOG(INFO) << "dump_profiler_file: " << output_filename;
    CHECK(std::ofstream(output_filename, mode)
              .write(&buf[0] + i * bo_size.single_batch_size,
                     bo_size.single_batch_size)
              .good())
        << " faild to write to " << output_filename;
  }

  // from the DDR Format define, if the buf[bo_size.single_batch_size -13] ==
  // 0x0, it means instr end.
  return (buf[bo_size.single_batch_size - 13] == 0x0);
}

static void usage() {
  std::cout << "usage: xvdpu_profiler <xmodel> [-i <subgraph_index>] [-t "
               "<start_file>] <input_bin> [input_bin]... \n"
            << std::endl;
}

inline void parse_opt(int argc, char* argv[]) {
  int opt = 0;

  while ((opt = getopt(argc, argv, "i:h:t:")) != -1) {
    switch (opt) {
      case 'i':
        g_subgraph_index = std::stoi(optarg);
        break;
      case 't':
        g_start_file = std::string(optarg);
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

static std::vector<std::pair<std::string, size_t>> load_start_file(
    const std::string& filename) {
  auto ret = std::vector<std::pair<std::string, size_t>>{};
  ifstream fin;
  fin.open(filename, ios_base::in);
  if (!fin) {
    std::cout << "Can't open the file " << filename << "\n";
    exit(-1);
  }
  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    istringstream ss(line);
    std::string name = "";
    size_t start_number = 0;
    ss >> name >> start_number;
    ret.push_back(std::make_pair(name, start_number));
  }
  fin.close();
  if (ENV_PARAM(DEBUG_XVDPU_PROFILER) >= 1) {
    for (const auto& r : ret) {
      LOG(INFO) << "layer name : " << r.first
                << "   start instr : " << r.second;
    }
  }
  return ret;
}

int main(int argc, char* argv[]) {
  size_t device_id = (size_t)ENV_PARAM(DEVICE_ID);
  size_t cu_index = (size_t)ENV_PARAM(CU_INDEX);
  size_t rec_instr = (size_t)ENV_PARAM(REC_INSTR);
  CHECK(rec_instr >= 0 && rec_instr <= 4)
      << "rec_instr out of range, 0:all, 1:load, 2:save, 3:conv, 4:misc";
  size_t start_instr = (size_t)ENV_PARAM(START_INSTR);
  size_t stop_instr = (size_t)ENV_PARAM(STOP_INSTR);
  if (stop_instr <= 0) {
    stop_instr = std::numeric_limits<uint32_t>::max();
  }
  const string& cu_name = ENV_PARAM(CU_NAME);

  parse_opt(argc, argv);
  auto graph = xir::Graph::deserialize(g_xmodel_file);
  auto subgraph = graph->get_root_subgraph();
  auto all = subgraph->children_topological_sort();
  CHECK(g_subgraph_index >= 0 && (size_t)g_subgraph_index < all.size())
      << "subgraph_index out of range";
  subgraph = all[g_subgraph_index];

  if (!g_start_file.empty()) {
    auto start_data = load_start_file(g_start_file);
    for (const auto& start_data_i : start_data) {
      auto bo_size = get_bo_size(cu_name, cu_index);
      auto bo = xir::BufferObject::create(bo_size.size, device_id, cu_name);
      LOG_IF(INFO, ENV_PARAM(DEBUG_XVDPU_PROFILER))
          << "bo_phy_addr: " << std::hex << "0x" << bo->phy();
      init_profiler_reg(cu_name, cu_index, bo->phy(), bo_size.single_batch_size,
                        rec_instr, start_data_i.second);
      auto attrs = xir::Attrs::create();
      std::unique_ptr<vart::RunnerExt> runner =
          vart::RunnerExt::create_runner(subgraph, attrs.get());
      run_dpu(runner.get(), g_input_files);
      // maybe write profiler data to DDR not completed, so check_finish_reg
      // here.
      while (!check_finish_reg(cu_name, cu_index)) {
        usleep(10 * 1000);
      }
      dump_profiler_file(bo.get(), bo_size, rec_instr, start_data_i.second,
                         start_data_i.first);
    }
  } else {
    bool end_flag = 1;
    do {
      auto bo_size = get_bo_size(cu_name, cu_index);
      auto bo = xir::BufferObject::create(bo_size.size, device_id, cu_name);
      LOG_IF(INFO, ENV_PARAM(DEBUG_XVDPU_PROFILER))
          << "bo_phy_addr: " << std::hex << "0x" << bo->phy();
      init_profiler_reg(cu_name, cu_index, bo->phy(), bo_size.single_batch_size,
                        rec_instr, start_instr);
      auto attrs = xir::Attrs::create();
      std::unique_ptr<vart::RunnerExt> runner =
          vart::RunnerExt::create_runner(subgraph, attrs.get());
      run_dpu(runner.get(), g_input_files);
      // maybe write profiler data to DDR not completed, so check_finish_reg
      // here.
      while (!check_finish_reg(cu_name, cu_index)) {
        usleep(10 * 1000);
      }
      end_flag = dump_profiler_file(bo.get(), bo_size, rec_instr, start_instr);
      // The PROF_START_INSTR_JUMP could be a bit smaller than RAM_DEPTH,
      // introducing a gap area, to avoid missing some parallel instrs.
      // RAM_DEPTH is bo_size.single_batch_size >> 4
      start_instr = start_instr + (size_t)(bo_size.single_batch_size >> 4) - 20;
    } while (!end_flag && (start_instr < stop_instr));
  }
  return 0;
}
