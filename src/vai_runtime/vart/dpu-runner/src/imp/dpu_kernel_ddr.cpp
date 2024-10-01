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
#include "./dpu_kernel_ddr.hpp"

#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/xxd.hpp>

DEF_ENV_PARAM(XLNX_ENABLE_CODE_UPLODING, "1");
DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0");
DEF_ENV_PARAM(XLNX_SHORT_CIRCUIT_DPU_CODE, "0");
namespace vart {
namespace dpu {
DpuKernelDdr::DpuKernelDdr(const std::string& filename,
                           const std::string& kernel,
                           xir::DpuController* dpu_controller,
                           size_t device_core_id)
    : vart::dpu::DpuKernel(filename, kernel),
      device_core_id_(device_core_id),  //
      cu_full_name_(dpu_controller->get_full_name(device_core_id)),
      device_id_(dpu_controller->get_device_id(device_core_id)) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << " create dpu kernel @" << (void*)this  //
      << " cu=" << cu_full_name_                //
      << " device_id=" << device_id_            //
      << " device_core_id=" << device_core_id_  //
      ;
}

DpuKernelDdr::DpuKernelDdr(const xir::Subgraph& sg, xir::Attrs* attrs,
                           xir::DpuController* dpu_controller,
                           size_t device_core_id)
    : vart::dpu::DpuKernel(sg, attrs),
      device_core_id_(device_core_id),  //
      cu_full_name_(dpu_controller->get_full_name(device_core_id)),
      device_id_(dpu_controller->get_device_id(device_core_id)) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << " create dpu kernel @" << (void*)this  //
      << " cu=" << cu_full_name_                //
      << " device_id=" << device_id_            //
      << " device_core_id=" << device_core_id_  //
      ;
}

void DpuKernelDdr::initialize() {  //
  vart::dpu::DpuKernel::initialize();
}

static std::unique_ptr<xir::BufferObject> create_buffer_object(
    size_t size, size_t device_id, const std::string& cu_name) {
  return xir::BufferObject::create(size, device_id, cu_name);
}

void DpuKernelDdr::load_parameter(
    const std::vector<vart::dpu::DpuReg>& parameters) {
  // do nothing
}

DpuKernelDdr::~DpuKernelDdr() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER)) << "kernel destoryed";
}

void DpuKernelDdr::load_code(const vart::dpu::DpuReg& code) {
  size_t device_id = device_id_;
  std::string cu_name = cu_full_name_;
  auto& mc_code = code.value_;
  auto mc_code_size = mc_code.size();
  codes_.emplace_back(create_buffer_object(mc_code_size, device_id, cu_name));
  auto& code_ = codes_.back();

  if (ENV_PARAM(XLNX_SHORT_CIRCUIT_DPU_CODE)) {
    LOG(WARNING) << "XLNX_SHORT_CIRCUIT_DPU_CODE=1 is applied, result might "
                    "not be correct, check "
                 << "offset " << std::hex << "0x" << code_->phy() << std::dec
                 << " "
                 << "size " << code_->size() << " "  //
        ;
    *((uint32_t*)(&code.value_[0])) =
        0x72200000u;  // SINGLE DPU END INSTRUCTION
  }
  if (!ENV_PARAM(XLNX_ENABLE_CODE_UPLODING)) {
    LOG(WARNING)
        << "code upload is cancelled because XLNX_ENABLE_CODE_UPLODING=1";
  } else {
    code_->copy_from_host(&mc_code[0], mc_code.size(), 0u);
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "loading release code  " << mc_code.size() << " bytes to " << std::hex
      << "0x" << code_->phy() << std::dec;
  // << vitis::ai::xxd((unsigned char*)&mc_code[0], 160, 16, 2);
}

std::map<std::string, uint64_t> DpuKernelDdr::get_parameter(
    size_t device_core_id /*not used*/) const {
  auto ret = std::map<std::string, uint64_t>{};
  // TO BE DELETED
  return ret;
}

std::vector<vart::dpu::DpuKernel::SubgraphCode> DpuKernelDdr::get_code(
    size_t device_core_id) const {
  auto size = super_layer_subgraph_.size();
  CHECK(size == codes_.size());

  auto subgraphcodes = std::vector<vart::dpu::DpuKernel::SubgraphCode>();
  subgraphcodes.reserve(size);
  auto subgraph_id = 0ul;
  for (const auto& subgraph : super_layer_subgraph_) {
    auto code_addr = codes_[subgraph_id]->phy();
    subgraphcodes.push_back(
        vart::dpu::DpuKernel::SubgraphCode{subgraph, code_addr});
    subgraph_id++;
  }
  return subgraphcodes;
}
}  // namespace dpu
}  // namespace vart
