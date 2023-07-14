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
#include "./dpu_kernel_hbm.hpp"

#include <glog/logging.h>

#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>
#include <vitis/ai/xxd.hpp>

#include "./hbm_config.hpp"
DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0");
DEF_ENV_PARAM(XLNX_ENABLE_CODE_UPLODING, "1");
DEF_ENV_PARAM(XLNX_SHORT_CIRCUIT_DPU_CODE, "0");
DEF_ENV_PARAM(XLNX_ENABLE_WEIGHT_SPLIT, "1");
namespace vart {
namespace dpu {

// get_hbm_managers()[reg_id, e.g. "D0"]
static std::map<std::string, std::shared_ptr<vart::dpu::HbmManager>>
get_hbm_managers(xir::DpuController* dpu_controller, size_t device_core_id) {
  auto ret = std::map<std::string, std::shared_ptr<vart::dpu::HbmManager>>();
  auto core_id = dpu_controller->get_core_id(device_core_id);
  auto chunk_defs = vart::dpu::get_hbm(core_id);
  auto mm = std::map<std::string, std::shared_ptr<vart::dpu::HbmManager>>();
  for (const auto& chunk_def : chunk_defs) {
    const auto& name = chunk_def.first;
    const auto& def = chunk_def.second;
    auto id =
        std::string("Dpu[") + std::to_string(device_core_id) + "]." + name;
    mm[name] = vitis::ai::WeakStore<std::string, vart::dpu::HbmManager>::create(
        id, def);
  }
  return mm;
}

DpuKernelHbm::DpuKernelHbm(const std::string& filename,
                           const std::string& kernel,
                           xir::DpuController* dpu_controller,
                           size_t device_core_id)
    : DpuKernel(filename, kernel),
      device_core_id_(device_core_id),  //
      cu_full_name_(dpu_controller->get_full_name(device_core_id)),
      device_id_(dpu_controller->get_device_id(device_core_id)),
      hbm_managers_(get_hbm_managers(dpu_controller, device_core_id)),  //
      code_chunks_(),
      parameter_chunks_() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << " create dpu kernel @" << (void*)this  //
      << " device_core_id=" << device_core_id   //
      << " cu=" << cu_full_name_                //
      << " device_id=" << device_id_            //
      ;

  auto device_id = dpu_controller->get_device_id(device_core_id);
  device_memory_ = vitis::ai::WeakStore<size_t, xir::DeviceMemory>::create(
      device_id, device_id);
}

DpuKernelHbm::DpuKernelHbm(const xir::Subgraph& sg, xir::Attrs* attrs,
                           xir::DpuController* dpu_controller,
                           size_t device_core_id)
    : DpuKernel(
          sg,
          attrs),  //
                   /// device_cu_(vart::dpu::XrtDeviceCu::get_instance()), //
      device_core_id_(device_core_id),
      cu_full_name_(dpu_controller->get_full_name(device_core_id)),
      device_id_(dpu_controller->get_device_id(device_core_id)),
      hbm_managers_(get_hbm_managers(dpu_controller, device_core_id)),  //
      code_chunks_{},
      parameter_chunks_{} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << " create dpu kernel @" << (void*)this  //
      << " device_core_id=" << device_core_id   //
      << " cu=" << cu_full_name_                //
      << " device_id=" << device_id_            //
      ;
  auto device_id = dpu_controller->get_device_id(device_core_id);
  device_memory_ = vitis::ai::WeakStore<size_t, xir::DeviceMemory>::create(
      device_id, device_id);
}

DpuKernelHbm::~DpuKernelHbm() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << " destroy dpu kernel @" << (void*)this  //
      << " device_core_id=" << device_core_id_   //
      << " cu=" << cu_full_name_                 //
      << " device_id=" << device_id_             //
      ;
}
// ret [device_core_id]
vart::dpu::HbmManager* DpuKernelHbm::get_code_hbm_manager() {
  auto it = hbm_managers_.find(std::string("I"));
  CHECK(it != hbm_managers_.end())
      << "cannot find code register definition. check HARDWARE_DEF.";
  return it->second.get();
}

vart::dpu::HbmManager* DpuKernelHbm::get_parameter_hbm_managers(
    const std::string& reg_id /*W0 or W1*/) {
  auto it = hbm_managers_.find(reg_id);
  CHECK(it != hbm_managers_.end())
      << "cannot find code register definition. check HARDWARE_DEF."  //
      << " reg_id=" << reg_id;
  return it->second.get();
}

static const xir::Subgraph* find_subgraph(const xir::Subgraph* sg) {
  if (sg->has_attr("reg_id_to_hw_segment")) {
    return sg;
  }
  if (!sg->is_root()) {
    return find_subgraph(const_cast<xir::Subgraph*>(sg)->get_parent());
  }
  return nullptr;
}

void DpuKernelHbm::load_parameter(
    const std::vector<vart::dpu::DpuReg>& parameters) {
  const bool enable_weight_split = ENV_PARAM(XLNX_ENABLE_WEIGHT_SPLIT) != 0;
  auto subgraph = find_subgraph(get_subgraph1(0));
  auto reg_id_to_hw_segment =
      subgraph->template get_attr<std::map<std::string, std::string>>(
          "reg_id_to_hw_segment");
  CHECK(subgraph != nullptr);

  size_t total = 0u;
  for (const auto& reg : parameters) {
    auto reg_id = reg.name_;
    auto reg_size = reg.size_;
    auto& weight_or_bias = reg.value_;
    // reg_id = "REG_0", "REG_1", or "REG_2" etc.
    // hw_reg_id = "W0", "W1"
    CHECK(!weight_or_bias.empty()) << "empty parameter! reg_id=" << reg_id;
    auto hw_reg_id = reg_id_to_hw_segment.find(reg_id);
    auto found_hw_reg_id = hw_reg_id != reg_id_to_hw_segment.end();
    CHECK(found_hw_reg_id) << "cannot find hw_reg_id! reg_id=" << reg_id;
    auto const_parameter_id =
        enable_weight_split ? hw_reg_id->second : std::string("W0");
    auto chunk =
        get_parameter_hbm_managers(const_parameter_id)->allocate(reg_size);
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
        << "loading parameter for device_core_id_ = " << device_core_id_
        << " reg_id = " << reg_id << " size= " << reg_size
        << " const_parameter_id=" << const_parameter_id
        << " allocated chunk = " << chunk->to_string();
    CHECK(chunk != nullptr) << " out of memory for parameter";
    chunk->upload(device_memory_.get(), &weight_or_bias[0], 0ul,
                  weight_or_bias.size());
    parameter_chunks_[reg_id] = std::move(chunk);
    total += weight_or_bias.size();
  }
  LOG_IF(WARNING, total == 0) << "zeros no parameter loaded";
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "total " << total << " bytes parameter loaded";
}

void DpuKernelHbm::load_code(const vart::dpu::DpuReg& code) {  //
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "loading code " << code.size_ << " bytes";
  {
    auto chunk = get_code_hbm_manager()->allocate(code.size_);
    CHECK(chunk != nullptr) << "out of memory for code";
    if (ENV_PARAM(XLNX_SHORT_CIRCUIT_DPU_CODE)) {
      LOG(WARNING) << "XLNX_SHORT_CIRCUIT_DPU_CODE=1 is applied, result might "
                      "not be correct, check "
                   << "offset " << chunk->get_offset() << " "
                   << "size " << chunk->get_size() << " "  //
          ;
      *((uint32_t*)(&code.value_[0])) =
          0x72200000u;  // SINGLE DPU END INSTRUCTION
    }
    if (!ENV_PARAM(XLNX_ENABLE_CODE_UPLODING)) {
      LOG(WARNING)
          << "code upload is cancelled because XLNX_DISABLE_CODE_UPLODING=1";
    } else {
      chunk->upload(device_memory_.get(), &code.value_[0], 0ul, code.size_);
    }
    code_chunks_.emplace_back(std::move(chunk));
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << " code loded. " << code.size_ << " bytes to " << std::hex << "0x"
      << code_chunks_.back()->get_offset() << std::dec << std::dec;
}
// return 8 reg, nullptr means not set.
std::map<std::string, uint64_t> DpuKernelHbm::get_parameter(
    size_t device_core_id) const {
  CHECK_EQ(device_core_id, device_core_id_) << "logical error";
  auto ret = std::map<std::string, uint64_t>();
  auto& chunks = parameter_chunks_;
  for (const auto& chunk : chunks) {
    const auto& reg_id = chunk.first;
    CHECK(chunk.second != nullptr);
    const auto& offset = chunk.second->get_offset();
    ret.emplace(std::make_pair(reg_id, offset));
  }
  return ret;
}
std::vector<vart::dpu::DpuKernel::SubgraphCode> DpuKernelHbm::get_code(
    size_t device_core_id) const {
  CHECK_EQ(device_core_id, device_core_id_) << "logical error";
  auto size = super_layer_subgraph_.size();
  auto subgraphcodes = std::vector<vart::dpu::DpuKernel::SubgraphCode>();
  subgraphcodes.reserve(size);
  auto subgraph_id = 0ul;
  for (const auto& subgraph : super_layer_subgraph_) {
    auto code_addr = code_chunks_[subgraph_id]->get_offset();
    subgraphcodes.push_back(
        vart::dpu::DpuKernel::SubgraphCode{subgraph, code_addr});
    subgraph_id++;
  }
  return subgraphcodes;
}
}  // namespace dpu
}  // namespace vart
