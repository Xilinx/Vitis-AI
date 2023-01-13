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
#include "./dpu_control_xrt_xv_dpu.hpp"

#include <ert.h>
#include <glog/logging.h>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <mutex>
#include <vitis/ai/env_config.hpp>
#ifndef _WIN32
#include <vitis/ai/trace.hpp>
#endif
#include "./dpu_edge.hpp"
DEF_ENV_PARAM(DEBUG_DPU_CONTROLLER, "0");
DEF_ENV_PARAM(DISABLE_DPU_CONTROLLER_XRT, "0");
DEF_ENV_PARAM(XLNX_SHOW_DPU_COUNTER, "0");

DEF_ENV_PARAM(DEBUG_AP_START_CU_XVDPU, "0");
DEF_ENV_PARAM_2(XLNX_DIRTY_HACK_XVDPU_GEN_BASE, "0x200", uint32_t);
#define DOMAIN xclBOKind(1)

DpuControllerXrtXvDpu::DpuControllerXrtXvDpu(
    std::unique_ptr<xir::XrtCu>&& xrt_cu)
    : xir::DpuController{},  //
      xrt_cu_{std::move(xrt_cu)} {
  for (size_t i = 0; i < get_num_of_dpus(); i++) {
    auto cu_device_id = xrt_cu_->get_device_id(i);
    auto cu_core_id = xrt_cu_->get_core_id(i);
    auto cu_name = xrt_cu_->get_instance_name(i);
    auto cu_full_name = xrt_cu_->get_full_name(i);
    auto cu_fingerprint = xrt_cu_->get_fingerprint(i);
    auto cu_batch = get_batch_size(i);
#ifndef _WIN32
    vitis::ai::trace::add_info("dpu-controller", TRACE_VAR(cu_device_id),
                               TRACE_VAR(cu_core_id), TRACE_VAR(cu_batch),
                               TRACE_VAR(cu_name), TRACE_VAR(cu_full_name),
                               TRACE_VAR(cu_fingerprint));
#endif
  }
}

DpuControllerXrtXvDpu::~DpuControllerXrtXvDpu() {}

std::string DpuControllerXrtXvDpu::xdpu_get_counter(size_t device_core_id) {
  std::ostringstream str;
  struct {
    char name[64];
    uint32_t addr;
  } regs[] = {
      {"AP", 0x0},                            //
      {"LSTART", 0x180},  {"LEND", 0x184},    //
      {"CSTART", 0x188},  {"CEND", 0x18C},    //
      {"SSTART", 0x190},  {"SEND", 0x194},    //
      {"MSTART", 0x198},  {"MEND", 0x19C},    //
      {"CYCLE_L", 0x1A0}, {"CYCLE_H", 0x1A4}  //
  };
  int cnt = 0;
  for (const auto& reg : regs) {
    auto value = xrt_cu_->read_register(device_core_id, reg.addr);
    str << " " << reg.name << " "  //
        << value << " "            //
        ;
    cnt++;
  }
  return str.str();
}
static std::string dump_gen_reg(const std::vector<uint64_t>& gen_reg) {
  std::ostringstream str;
  str << std::hex;
  for (const auto& v : gen_reg) {
    str << " 0x" << v;
  }
  return str.str();
}
static size_t get_offset(size_t batch, size_t reg_id) {
  auto base = ENV_PARAM(XLNX_DIRTY_HACK_XVDPU_GEN_BASE);
  auto offset = base + batch * 0x40u + reg_id * 0x8u;
  return offset;
}

void DpuControllerXrtXvDpu::run(size_t core_idx, const uint64_t code,
                                const std::vector<uint64_t>& gen_reg) {
  static std::vector<std::mutex> mutexes(xrt_cu_->get_num_of_cu());
  auto num_of_cu = xrt_cu_->get_num_of_cu();
  core_idx = core_idx % num_of_cu;

  // std::lock_guard<std::mutex> lock(mutexes[core_idx]);

  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
      << std::hex                                          //
      << "code 0x" << code << " "                          //
      << "core_idx " << core_idx << " "                    //
      << "gen_reg: " << dump_gen_reg(gen_reg) << std::dec  //
      ;
  auto size_of_gen_regs = get_size_of_gen_regs(core_idx);
  auto batch_size = get_batch_size(core_idx);
  auto func = [code, &gen_reg, size_of_gen_regs,
               batch_size](ert_start_kernel_cmd* ecmd) -> void {
    if (ENV_PARAM(DEBUG_AP_START_CU_XVDPU)) {
      ecmd->state = ERT_CMD_STATE_NEW;
      ecmd->opcode = ERT_START_CU;
      ecmd->data[XDPU_CONTROL_AP] = 0x0;
      ecmd->data[XDPU_CONTROL_IER / 4] =
          0x1;  // must enable this, otherwise, DPU can only used once.
      ecmd->data[XDPU_CONTROL_PROF_ENA / 4] = 0x1;
      ecmd->data[1] = 0x1;  // GLBL_IRQ_ENA(Global Interrupt Enable Register)
      ecmd->data[2] = 0x1;  // IP_IRQ_ENA(IP Interrupt Enable Register)
      ecmd->data[3] = 0x0;  // IP_IRQ_STS(IP Interrupt Status Register)
      ecmd->data[0x44 / 4] = 0x0;  // PROF_ENA
      ecmd->data[XDPU_CONTROL_HP / 4] = 0x07070f0f;

      ecmd->data[XDPU_CONTROL_ADDR_INSTR_L / 4] = code & 0xFFFFFFFF;
      ecmd->data[XDPU_CONTROL_ADDR_INSTR_L / 4 + 1] = (code >> 32) & 0xFFFFFFFF;
      size_t max_offset = 0u;
      for (auto batch_id = 0u; batch_id < batch_size; batch_id++) {
        for (auto reg_id = 0u; reg_id < size_of_gen_regs; ++reg_id) {
          auto offset = get_offset(batch_id, reg_id);
          auto reg_value = gen_reg[batch_id * size_of_gen_regs + reg_id];
          if (reg_value != ((uint64_t)-1)) {
            ecmd->data[offset / 4u] = reg_value & 0xFFFFFFFF;
            ecmd->data[offset / 4u + 1] = (reg_value >> 32) & 0xFFFFFFFF;
            max_offset = std::max(max_offset, offset / 4u + 1);
          }
        }
      }
      // don't ask me why +2, refer to /opt/xilinx/xrt/include/xrt.h
      ecmd->count = max_offset + 2;
    } else {
      ecmd->state = ERT_CMD_STATE_NEW;
      ecmd->opcode = ERT_EXEC_WRITE;
      auto p = ecmd->extra_cu_masks;
      // ecmd->data[0] = 0;  // [0] APCTL=0 = XDPU_CONTROL_AP_START=0
      // ecmd->data[1] = 1;  // [1] GIE =1
      // ecmd->data[2] = 1;  // [2] IER = 1
      /* ecmd->data[p++] = 0;
      ecmd->data[p++] = 0;
      ecmd->data[p++] = 1;
      ecmd->data[p++] = 1;
      ecmd->data[p++] = 2;
      ecmd->data[p++] = 1; */
      ecmd->data[p++] = 0x40;  // CLEAR INTERRUPT
      ecmd->data[p++] = 1;
      ecmd->data[p++] = 0x44;  // PROF_EN=0
      ecmd->data[p++] = 0;
      ecmd->data[p++] = XDPU_CONTROL_HP;  // PROF_EN=0
      ecmd->data[p++] = 0x07070f0f;
      ecmd->data[p++] = XDPU_CONTROL_ADDR_INSTR_L;
      ecmd->data[p++] = code & 0xFFFFFFFF;
      ecmd->data[p++] = XDPU_CONTROL_ADDR_INSTR_L + 4;
      ecmd->data[p++] = (code >> 32) & 0xFFFFFFFF;
      for (auto batch_id = 0u; batch_id < batch_size; batch_id++) {
        for (auto reg_id = 0u; reg_id < size_of_gen_regs; ++reg_id) {
          auto offset = get_offset(batch_id, reg_id);
          auto reg_value = gen_reg[batch_id * size_of_gen_regs + reg_id];
          if (reg_value != ((uint64_t)-1)) {
            ecmd->data[p++] = offset;
            ecmd->data[p++] = reg_value & 0xFFFFFFFF;
            ecmd->data[p++] = offset + 4;
            ecmd->data[p++] = (reg_value >> 32) & 0xFFFFFFFF;
          }
        }
      }
      ecmd->count = p + 1;
    }
  };

#ifndef _WIN32
  vitis::ai::trace::add_trace("dpu-controller", vitis::ai::trace::func_start,
                              core_idx, 0);
  //vitis::ai::trace::lock(core_idx);
#endif
  xrt_cu_->run(
      core_idx, func,
      // on_success
      [core_idx, this](xclDeviceHandle handle, uint64_t cu_addr) -> void {
        if (ENV_PARAM(XLNX_SHOW_DPU_COUNTER)) {
          std::cout << "core_idx = " << core_idx << " "
                    << xdpu_get_counter(core_idx) << std::endl;
        }
      },
      // on failure
      [core_idx, this](xclDeviceHandle handle, uint64_t cu_addr) -> void {
        LOG(FATAL) << "dpu timeout! "
                   << "core_idx = " << core_idx << "\n"
                   << xdpu_get_counter(core_idx);
      });
  auto hwconuter = get_device_hwconuter(core_idx);
#ifndef _WIN32
  //vitis::ai::trace::unlock(core_idx);
  vitis::ai::trace::add_trace("dpu-controller", vitis::ai::trace::func_end,
                              core_idx, hwconuter);
#endif
}
size_t DpuControllerXrtXvDpu::get_num_of_dpus() const {
  return xrt_cu_->get_num_of_cu();
}
size_t DpuControllerXrtXvDpu::get_device_id(size_t device_core_id) const {
  return xrt_cu_->get_device_id(device_core_id);
}
size_t DpuControllerXrtXvDpu::get_core_id(size_t device_core_id) const {
  return xrt_cu_->get_core_id(device_core_id);
}
uint64_t DpuControllerXrtXvDpu::get_fingerprint(size_t device_core_id) const {
  return xrt_cu_->get_fingerprint(device_core_id);
}

uint64_t  DpuControllerXrtXvDpu::get_device_hwconuter(size_t device_core_id) const {
  uint32_t cycle_l_addr = 0x1A0;
  uint32_t cycle_h_addr = 0x1A4;

  auto value_l = xrt_cu_->read_register(device_core_id, cycle_l_addr);
  auto value_h = xrt_cu_->read_register(device_core_id, cycle_h_addr);

  uint64_t value = ((uint64_t)value_h << 32) | value_l;
  return value;
}

size_t DpuControllerXrtXvDpu::get_batch_size(size_t device_core_id) const {
  // see
  auto value = xrt_cu_->read_register(device_core_id, 0x134);
  auto ret = 1;
  if (value >= 1u && value <= 8u) {
    ret = (int)value;
  }
  return ret;
}

size_t DpuControllerXrtXvDpu::get_size_of_gen_regs(
    size_t device_core_id) const {
  return 8u;
}
std::string DpuControllerXrtXvDpu::get_full_name(size_t device_core_id) const {
  return xrt_cu_->get_full_name(device_core_id);
}
std::string DpuControllerXrtXvDpu::get_kernel_name(
    size_t device_core_id) const {
  return xrt_cu_->get_kernel_name(device_core_id);
}
std::string DpuControllerXrtXvDpu::get_instance_name(
    size_t device_core_id) const {
  return xrt_cu_->get_instance_name(device_core_id);
}
