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
#include "./dpu_control_xrt_cloud.hpp"

#include <glog/logging.h>

#include <bitset>
#include <iostream>
#include <mutex>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>

#include "dpu_cloud.hpp"
#include "ert.h"
DEF_ENV_PARAM(DEBUG_DPU_CONTROLLER, "0")
DEF_ENV_PARAM(DISABLE_DPU_CONTROLLER_XRT, "0")
DEF_ENV_PARAM(XLNX_SHOW_DPU_COUNTER, "0");
DEF_ENV_PARAM(XLNX_ENBALE_AP_START_CU_CLOUD, "-1");
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

DpuControllerXrtCloud::DpuControllerXrtCloud(
    std::unique_ptr<xir::XrtCu>&& xrt_cu, DPU_CLOUD_TYPE dpu_cloud_type)
    : xir::DpuController{},  //
      xrt_cu_{std::move(xrt_cu)},
      dpu_cloud_type_{dpu_cloud_type} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
      << "creating dpu controller: "  //
      << " this=" << (void*)this      //
      ;
}

DpuControllerXrtCloud::~DpuControllerXrtCloud() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
      << "destroying dpu controller: "  //
      << " this=" << (void*)this        //
      ;
}

std::string DpuControllerXrtCloud::xdpu_get_counter(size_t device_core_id) {
  std::ostringstream str;
  struct {
    char name[64];
    uint32_t addr;
  } regs[] = {
      {"APSTART", 0x00}, {"DONE", 0x80},  //
      {"LSTART", 0xA0},  {"LEND", 0x90},  //
      {"CSTART", 0x98},  {"CEND", 0x88},  //
      {"SSTART", 0x9C},  {"SEND", 0x8C},  //
      {"PSTART", 0x94},  {"PEND", 0x84},  //
      {"CYCLE", 0xA8},
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

static int get_ap_start_cu(DPU_CLOUD_TYPE type) {
  auto ap_start_cu = ENV_PARAM(XLNX_ENBALE_AP_START_CU_CLOUD);
  if (ap_start_cu == -1) {
    ap_start_cu = type == DPU_CLOUD_TYPE::V4E ? 1 : 0;
  }
  return ap_start_cu;
}

void DpuControllerXrtCloud::run(size_t device_core_idx, const uint64_t code,
                                const std::vector<uint64_t>& gen_reg) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
      << std::hex                                          //
      << "code 0x" << code << " "                          //
      << "device_core_idx " << device_core_idx << " \n"    //
      << "gen_reg: " << dump_gen_reg(gen_reg) << std::dec  //

      ;
  auto ap_start_cu = get_ap_start_cu(dpu_cloud_type_);
  auto func = [device_core_idx, code, ap_start_cu,
               &gen_reg](ert_start_kernel_cmd* ecmd) -> void {
    if (ap_start_cu) {
      ecmd->state = ERT_CMD_STATE_NEW;
      ecmd->opcode = ERT_START_CU;

      ecmd->data[XDPU_CONTROL_AP] = XDPU_CONTROL_AP_START;
      ecmd->data[XDPU_CONTROL_GIE / 4] = XDPU_GLOBAL_INT_ENABLE;
      ecmd->data[XDPU_CONTROL_START / 4] = 0x00;
      ecmd->data[XDPU_CONTROL_RESET / 4] = 0x1;
      ecmd->data[XDPU_CONTROL_MEAN0 / 4] = 0x0;
      ecmd->data[XDPU_CONTROL_INSTR_L / 4] = code & 0xFFFFFFFF;
      ecmd->data[XDPU_CONTROL_INSTR_L / 4 + 1] = (code >> 32) & 0xFFFFFFFF;

      const int* offset = &GEN_REG_OFFSET[0];
      size_t size = GEN_REG_OFFSET.size();
      size_t i;
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
          << "offset[0] " << offset[0] << " "            //
          << "gen_reg.size() " << gen_reg.size() << " "  //
          << "size " << size;
      for (i = 0u; i < gen_reg.size() && i < size; ++i) {
        ecmd->data[offset[i]] = gen_reg[i] & 0xFFFFFFFF;
        ecmd->data[offset[i] + 1] = (gen_reg[i] >> 32) & 0xFFFFFFFF;
      }
      // TODO: error when code or param after offset[i-1]
      ecmd->count = offset[i - 1] + 1;
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
          << "offset[0] " << offset[0] << " "            //
          << "gen_reg.size() " << gen_reg.size() << " "  //
          << "size " << size << " "                      //
          << "ecmd->count " << ecmd->count;
      ;

    } else {
      ecmd->state = ERT_CMD_STATE_NEW;
      ecmd->opcode = ERT_EXEC_WRITE;
      ecmd->data[XDPU_CONTROL_AP] = XDPU_CONTROL_AP_START;  // [0] APCTL=0,
      ecmd->data[XDPU_CONTROL_GIE / 4] = XDPU_GLOBAL_INT_ENABLE;  // [1] GIE =1
      ecmd->data[XDPU_CONTROL_IER / 4] = 1;                       // [2] IER = 1
      auto p = 6;
      ecmd->data[p++] = XDPU_CONTROL_START;
      ecmd->data[p++] = 0x00;
      ecmd->data[p++] = 0x14;
      ecmd->data[p++] = 0x200;
      ecmd->data[p++] = XDPU_CONTROL_RESET;
      ecmd->data[p++] = 0x01;
      // ecmd->data[p++] = XDPU_CONTROL_HP;
      // ecmd->data[p++] = 0x204040;
      ecmd->data[p++] = XDPU_CONTROL_INSTR_L;
      ecmd->data[p++] = code & 0xFFFFFFFF;
      ecmd->data[p++] = XDPU_CONTROL_INSTR_H;
      ecmd->data[p++] = (code >> 32) & 0xFFFFFFFF;
      auto offset = &GEN_REG_OFFSET[0];
      auto size = GEN_REG_OFFSET.size();

      for (auto i = 0u; i < gen_reg.size() && i < size; ++i) {
        if (gen_reg[i] != 0xffffffffffffffff) {
          ecmd->data[p++] = offset[i] * 4;
          ecmd->data[p++] = gen_reg[i] & 0xFFFFFFFF;
          ecmd->data[p++] = (offset[i] + 1) * 4;
          ecmd->data[p++] = (gen_reg[i] >> 32) & 0xFFFFFFFF;
        }
      }
      ecmd->count = p + 1;
    }
  };
  xrt_cu_->run(
      device_core_idx, func,
      // on_success
      [device_core_idx, this](xclDeviceHandle handle,
                              uint64_t cu_addr) -> void {
        if (ENV_PARAM(XLNX_SHOW_DPU_COUNTER)) {
          LOG(INFO) << "device_core_idx = " << device_core_idx << " "
                    << xdpu_get_counter(device_core_idx) << std::endl;
        }
      },
      // on failure
      [device_core_idx, this](xclDeviceHandle handle,
                              uint64_t cu_addr) -> void {
        LOG(FATAL) << "dpu timeout! "
                   << "device_core_idx = " << device_core_idx << "\n"
                   << xdpu_get_counter(device_core_idx);
      });
}

size_t DpuControllerXrtCloud::get_num_of_dpus() const {
  return xrt_cu_->get_num_of_cu();
}

size_t DpuControllerXrtCloud::get_device_id(size_t device_core_id) const {
  return xrt_cu_->get_device_id(device_core_id);
}
size_t DpuControllerXrtCloud::get_core_id(size_t device_core_id) const {
  return xrt_cu_->get_core_id(device_core_id);
}
uint64_t DpuControllerXrtCloud::get_fingerprint(size_t device_core_id) const {
  return xrt_cu_->get_fingerprint(device_core_id);
}

size_t DpuControllerXrtCloud::get_batch_size(size_t device_core_id) const {
  // see
  // https://confluence.xilinx.com/display/~xfeng/V4E+CSR+Specification
  // 12‘h1EC r dpu0_hardware_engine_num 32'h0 [3:0]: DPU engine number,
  // possible rang is 1~8 [31:4]: reserved.
  auto value = xrt_cu_->read_register(device_core_id, 0x1ec);
  auto ret = 1;
  if (value >= 1u && value <= 8u) {
    ret = (int)value;
  }
  return ret;
}

std::string DpuControllerXrtCloud::get_full_name(size_t device_core_id) const {
  return xrt_cu_->get_full_name(device_core_id);
}
std::string DpuControllerXrtCloud::get_kernel_name(
    size_t device_core_id) const {
  return xrt_cu_->get_kernel_name(device_core_id);
}
std::string DpuControllerXrtCloud::get_instance_name(
    size_t device_core_id) const {
  return xrt_cu_->get_instance_name(device_core_id);
}
