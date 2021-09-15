/*
 * Copyright 2021 Xilinx Inc.
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

#include "rnn_controller_u25v2.hpp"

#include <algorithm>
#include <fstream>
#include <memory>
#include <numeric>
#include <utility>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/weak.hpp>
#include <xir/xrt_device_handle.hpp>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename T>
static std::string dump_binary_data(std::string filename, T* data,
                                    size_t size) {
  std::ofstream of(filename);
  for (size_t i = 0; i < size; ++i) {
    of << std::hex << data[i] << '\n';
  }
  of << std::dec;
  return " Done";
}

namespace vart {
namespace xrnn {

std::vector<uint32_t> RnnControllerU25v2::get_reg_data(int frame,
                                                       int thread_index) {
  std::vector<uint32_t> regs_array(MAX_REG_ADDR / 4, 0);

  std::string board = get_board_name();

  // TODO : How I/O address changes w.r.t core-id ?
  if (board == "u25") {
    regs_array[REG_FRAME_LEN / 4] = frame;
    regs_array[REG_INSTR_BASE_ADDR_L / 4] = 0x0700'0000;
    regs_array[REG_MODEL_BASE_ADDR_L / 4] = 0x0000'0000;
    regs_array[REG_PROF_START_ADDR_L / 4] = 0x0800'0000;
    regs_array[REG_PROF_EN / 4] = 0x0000'0001;
    regs_array[REG_INSTR_BASE_ADDR_H / 4] = 0x0000'0008;
    regs_array[REG_INPUT_BASE_ADDR_H / 4] = 0x0000'0008;
    regs_array[REG_PROF_START_ADDR_H / 4] = 0x0000'0008;
    regs_array[REG_INPUT_BASE_ADDR_L / 4] = thread_index * THREAD_STEP;
    regs_array[REG_OUTPUT_BASE_ADDR_L / 4] = thread_index * THREAD_STEP;
  }
  return regs_array;
}

std::string RnnControllerU25v2::get_board_name() { return "u25"; }

size_t RnnControllerU25v2::get_base_addr(unsigned batch_num) {
  std::string addr_name = get_addr_name();
  CHECK(batch_addr_map.count(addr_name) == 1)
      << "Can Not Find Addr For " << addr_name;
  std::vector<size_t>& base_addrs = *batch_addr_map[addr_name];
  CHECK(batch_num < base_addrs.size()) << "Invalid batch_num: " << batch_num;

  return base_addrs[batch_num];
}

const std::vector<size_t>& RnnControllerU25v2::get_init_addr() {
  std::string addr_name = get_addr_name();
  CHECK(init_addr_map.count(addr_name) == 1)
      << "Can Not Find Addr For " << addr_name;
  const std::vector<size_t>& base_addrs = *init_addr_map[addr_name];
  return base_addrs;
}

int RnnControllerU25v2::get_batch_size() { return batch_; }

RnnControllerU25v2::RnnControllerU25v2(size_t device_core_id,
                                       std::unique_ptr<xir::XrtCu>&& xrt_cu)
    : RnnControllerU50LV(device_core_id, std::move(xrt_cu)) {
  auto core_id = xrt_cu_->get_core_id(idx_);
  batch_ = 1;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "Initialised RnnControllerU25v2 : idx_" << idx_ << "; core_id "
      << core_id << "; batch " << batch_;
}

RnnControllerU25v2::~RnnControllerU25v2() {}

// REGISTER_RNN_CONTROLLER("U25v2", RnnControllerU25v2)
struct RegisterU25v2 {
  RegisterU25v2() {
    RnnControllerCreate::Register(
        "U25", [](unsigned int device_core_id, std::string device) {
          return std::make_unique<RnnControllerU25v2>(
              device_core_id, std::make_unique<xir::XrtCu>(device));
        });
  }
};
static RegisterU25v2 dummyRegU25v2;

}  // namespace xrnn
}  // namespace vart
