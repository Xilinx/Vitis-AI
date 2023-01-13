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

#include "rnn_controller_u50lv.hpp"

#include <algorithm>
#include <fstream>
#include <memory>
#include <numeric>
#include <utility>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/weak.hpp"
#include "xir/xrt_device_handle.hpp"

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename T>
static std::string dump_binary_data(std::string filename, T* data, size_t size,
                                    bool hexify = true) {
  std::ofstream of(filename);
  if (hexify) {
    of << std::hex;
  }
  for (size_t i = 0; i < size; ++i) {
    of << data[i] << '\n';
  }
  of << std::dec;
  return " Done";
}

namespace vart {
namespace xrnn {

std::vector<uint32_t> RnnControllerU50LV::get_reg_data(int frame,
                                                       int thread_index) {
  std::vector<uint32_t> regs_array(MAX_REG_ADDR / 4, 0);

  std::string board = get_board_name();

  // TODO : How I/O address changes w.r.t core-id ?
  if (board == "u50") {
    regs_array[REG_FRAME_LEN / 4] = frame;
    regs_array[REG_INSTR_BASE_ADDR_L / 4] = 0x0700'0000;
    regs_array[REG_MODEL_BASE_ADDR_L / 4] = 0x0000'0000;
    regs_array[REG_PROF_START_ADDR_L / 4] = 0x0800'0000;
    regs_array[REG_PROF_EN / 4] = 0x0000'0001;
    if (idx_ == 0) {
      regs_array[REG_INSTR_BASE_ADDR_H / 4] = 0x0000'0001;
      regs_array[REG_INPUT_BASE_ADDR_H / 4] = 0x0000'0001;
      regs_array[REG_PROF_START_ADDR_H / 4] = 0x0000'0001;
    } else if (idx_ == 1) {
      regs_array[REG_INSTR_BASE_ADDR_H / 4] = 0x0000'0000;
      regs_array[REG_INPUT_BASE_ADDR_H / 4] = 0x0000'0000;
      regs_array[REG_PROF_START_ADDR_H / 4] = 0x0000'0000;
    }
    regs_array[REG_INPUT_BASE_ADDR_L / 4] = thread_index * THREAD_STEP;
    regs_array[REG_OUTPUT_BASE_ADDR_L / 4] = thread_index * THREAD_STEP;
  }
  return regs_array;
}

std::string RnnControllerU50LV::get_board_name() {
  // auto kernel_name = xrt_cu_->get_kernel_name(idx_);
  // return kernel_name.find("slr") != kernel_name.npos ? "u50" : "u25";
  return "u50";
}

std::string RnnControllerU50LV::get_addr_name() {
  std::string board = get_board_name();
  return board + "_cu" + std::to_string(idx_);
}

size_t RnnControllerU50LV::get_base_addr(unsigned batch_num) {
  std::string addr_name = get_addr_name();
  CHECK(batch_addr_map.count(addr_name) == 1)
      << "Can Not Find Addr For " << addr_name;
  std::vector<size_t>& base_addrs = *batch_addr_map[addr_name];
  CHECK(batch_num < base_addrs.size()) << "Invalid batch_num: " << batch_num;

  return base_addrs[batch_num];
}

const std::vector<size_t>& RnnControllerU50LV::get_init_addr() {
  std::string addr_name = get_addr_name();
  CHECK(init_addr_map.count(addr_name) == 1)
      << "Can Not Find Addr For " << addr_name;
  const std::vector<size_t>& base_addrs = *init_addr_map[addr_name];
  return base_addrs;
}

int RnnControllerU50LV::get_batch_size() { return batch_; }

RnnControllerU50LV::RnnControllerU50LV(size_t device_core_id,
                                       std::unique_ptr<xir::XrtCu>&& xrt_cu)
    : idx_{device_core_id},
      xrt_cu_{std::move(xrt_cu)},
      memory_{vitis::ai::WeakStore<size_t, xir::DeviceMemory>::create(
          xrt_cu_->get_device_id(idx_), xrt_cu_->get_device_id(idx_))} {
  batch_ = idx_ == 0 ? 3 : 4;
  auto core_id = xrt_cu_->get_core_id(idx_);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "Initialised RnnControllerU50LV : idx_" << idx_ << "; core_id "
      << core_id << "; batch " << batch_;
}

void RnnControllerU50LV::run(char* in, uint64_t isize, char* out,
                             uint64_t osize, int batch, int frame,
                             int thread_index) {
  CHECK(batch <= batch_) << "Invalid Batch Size";

  auto core_id = xrt_cu_->get_core_id(idx_);

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "device_core_id " << idx_ << " core_id " << core_id;

  __TIC__(REG_COMPUTE)
  std::vector<uint32_t> reg_data = get_reg_data(frame, thread_index);
  __TOC__(REG_COMPUTE)
  LOG_IF(INFO, ENV_PARAM(DEBUG_DUMP_DATA))
      << "Dumping basic register to reg_v2.log ..."
      << dump_binary_data("reg_v2.log", reg_data.data(), reg_data.size());

  __TIC__(INPUT_UPLOAD)
  for (auto i = 0; i < batch; i++) {
    auto base_addr = get_base_addr(i);

    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
        << "vetor input addr: " << std::hex
        << (base_addr + ADDR(VECTOR) + thread_index * THREAD_STEP);

    LOG_IF(INFO, ENV_PARAM(DEBUG_DUMP_DATA))
        << "Writing input to in_v2_b" << i << ".log... " << std::hex
        << dump_binary_data("in_v2_b" + std::to_string(i) + ".log",
                            reinterpret_cast<int16_t*>(in + i * isize),
                            isize / 2, /*hexify*/ false);
    memory_->upload(
        (void*)(in + i * isize),
        (size_t)(base_addr + ADDR(VECTOR) + thread_index * THREAD_STEP),
        (size_t)isize);
  }
  __TOC__(INPUT_UPLOAD)

  auto func = [=](ert_start_kernel_cmd* ecmd) -> void {
    auto rsz = (MAX_REG_ADDR / 4 + 1) + 1;  // regmap array size
    ecmd->count = 1 + rsz;

    ecmd->data[0x00] = 0x00;

    for (unsigned i = 1; i < reg_data.size(); i++) {
      ecmd->data[i] = reg_data[i];
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
        << "reg size" << reg_data.size();
  };

  auto out_addr = nlayers_ % 2 ? ADDR(RESL) : ADDR(VECTOR);

  xrt_cu_->run(
      idx_, func,
      // on_success
      [=](xclDeviceHandle handle, uint64_t cu_addr) -> void {
        LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
            << "xrnn excute done! "
            << "core_id = " << core_id << " thread = " << thread_index;

        LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
            << "get result len: " << osize << "(0x" << std::hex << osize << ")";

        __TIC__(OUTPUT_DOWNLOAD)
        for (auto i = 0; i < batch; i++) {
          auto base_addr = get_base_addr(i);
          LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
              << "result output addr: " << std::hex
              << (base_addr + out_addr + thread_index * THREAD_STEP);
          memory_->download(
              (void*)(out + i * osize),
              (size_t)(base_addr + out_addr + thread_index * THREAD_STEP),
              (size_t)osize);
          LOG_IF(INFO, ENV_PARAM(DEBUG_DUMP_DATA))
              << "Writing output to out_v2_b" << i << ".log... " << std::hex
              << dump_binary_data("out_v2_b" + std::to_string(i) + ".log",
                                  reinterpret_cast<int16_t*>(out + i * osize),
                                  osize / 2, /*hexify*/ false);
        }
        __TOC__(OUTPUT_DOWNLOAD)
      },
      // on failure
      [core_id](xclDeviceHandle handle, uint64_t cu_addr) -> void {
        LOG(INFO) << "xrnn controller timeout! "
                  << "core_id = " << core_id << "\n";
      });
}

RnnControllerU50LV::~RnnControllerU50LV() {}

void RnnControllerU50LV::init(char* ddr, uint64_t size) {
  const std::vector<size_t>& init_addrs = get_init_addr();

  for (unsigned i = 0; i < init_addrs.size(); i++) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
        << "ddr_init at: " << std::hex << init_addrs[i];
    memory_->upload((void*)ddr, (size_t)init_addrs[i], (size_t)size);
  }
}

void RnnControllerU50LV::update(int frame, ModelConfig* mc, uint32_t* p_ddr,
                                size_t size) {
  __TIC__(INSTR_COMPUTE)
  constexpr uint32_t NREGS_PER_ROW = 4;
  constexpr uint32_t HEADER_REGS = 8;   // 2 rows (head_len && instr_count)
  constexpr uint32_t L0_SKIP_REGS = 4;  // Layer 0 has extra row for end_instr

  constexpr int reg_step = 0x30;
  constexpr int reg_load0 = 0x00;
  constexpr int reg_load1 = 0x04;
  // constexpr int reg_save0 = 0x20;
  constexpr int reg_save1 = 0x24;
  nlayers_ = mc->get_layer_num();

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER)) << "layers: " << nlayers_;

  const auto& ddr_layout = mc->get_ddr_layout();
  for (int i = 0; i < nlayers_; i++) {
    int ddr_regs_ptr = ddr_layout[i * 3] * NREGS_PER_ROW + HEADER_REGS;
    if (i == 0) {
      ddr_regs_ptr += L0_SKIP_REGS;
    }

    int dir_val = (mc->get_reg_dir(i, CONFIG_NAME::LOAD0) == 1) ? 0 : 1;
    int offset_load0 =
        dir_val * (frame - 1) * mc->get_reg_size(i, CONFIG_NAME::LOAD0) * 2;
    int offset_load1 =
        dir_val * (frame - 1) * mc->get_reg_size(i, CONFIG_NAME::LOAD1) * 2;
    int offset_save1 =
        dir_val * (frame - 1) * mc->get_reg_size(i, CONFIG_NAME::SAVE1) * 2;

    for (int b = 0; b < batch_; b++) {
      uint32_t batch_base = get_base_addr(b) & 0xFFFFFFFF;
      if (i % 2 == 0) {
        p_ddr[ddr_regs_ptr + (reg_load0 + reg_step * b) / 4] =
            batch_base + ADDR(VECTOR) + offset_load0;
        p_ddr[ddr_regs_ptr + (reg_load1 + reg_step * b) / 4] =
            batch_base + ADDR(RESL) + offset_load1;
        p_ddr[ddr_regs_ptr + (reg_save1 + reg_step * b) / 4] =
            batch_base + ADDR(RESL) + offset_save1;
      } else {
        p_ddr[ddr_regs_ptr + (reg_load0 + reg_step * b) / 4] =
            batch_base + ADDR(RESL) + offset_load0;
        p_ddr[ddr_regs_ptr + (reg_load1 + reg_step * b) / 4] =
            batch_base + ADDR(VECTOR) + offset_load1;
        p_ddr[ddr_regs_ptr + (reg_save1 + reg_step * b) / 4] =
            batch_base + ADDR(VECTOR) + offset_save1;
      }
    }
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER)) << "update ddr";
  const std::vector<size_t>& init_addrs = get_init_addr();

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "instruction: " << std::hex << init_addrs[0] + ADDR(INSTR);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "p_ddr @" << p_ddr << "size: " << size;
  LOG_IF(INFO, ENV_PARAM(DEBUG_DUMP_DATA))
      << "Dumping ddr register to ddr_reg_v2.log ..."
      << dump_binary_data("ddr_reg_v2.log", (uint32_t*)p_ddr, size / 4);
  __TOC__(INSTR_COMPUTE)

  __TIC__(INSTR_UPLOAD)
  memory_->upload((void*)p_ddr, (size_t)(init_addrs[0] + ADDR(INSTR)),
                  (size_t)size);
  __TOC__(INSTR_UPLOAD)
}

// REGISTER_RNN_CONTROLLER("U50LV", RnnControllerU50LV)
struct RegisterU50LV {
  RegisterU50LV() {
    RnnControllerCreate::Register(
        "U50", [](unsigned int device_core_id, std::string device) {
          return std::make_unique<RnnControllerU50LV>(
              device_core_id, std::make_unique<xir::XrtCu>(device));
        });
  }
};
static RegisterU50LV dummyRegU50LV;

}  // namespace xrnn
}  // namespace vart
