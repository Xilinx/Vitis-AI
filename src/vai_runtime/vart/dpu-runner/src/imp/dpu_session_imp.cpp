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

#include "./dpu_session_imp.hpp"

#include <cmath>
#include <vart/tensor_buffer.hpp>  // vitis
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>
#include <xir/dpu_controller.hpp>
#include <UniLog/UniLog.hpp>

#include "../../runner/src/runner_helper.hpp"
#include "./dpu_kernel_ddr.hpp"
#include "./dpu_kernel_hbm.hpp"
#include "./dpu_runner_ddr.hpp"
#include "./dpu_runner_hbm.hpp"
#include "vart/assistant/tensor_buffer_allocator.hpp"

DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0");
DEF_ENV_PARAM_2(XLNX_DDR_OR_HBM, "", std::vector<std::string>);

#if IS_EDGE
DEF_ENV_PARAM(XLNX_TENSOR_BUFFER_LOCATION, "1" /* HOST_PHY */);
#else
DEF_ENV_PARAM(XLNX_TENSOR_BUFFER_LOCATION, "2" /* DEVICE */);
#endif

namespace vart {
namespace dpu {

static bool is_ddr(size_t device_id) {
  // TODO: xrt_device_handler_imp.cpp should detect the HW platform
  // and set this variable properly.
  // see xrt_device_handle_imp.cpp
  auto& xlnx_ddr_or_hbm = ENV_PARAM(XLNX_DDR_OR_HBM);
  if (xlnx_ddr_or_hbm.empty()) {
    // dirty hack: For vivado flow, there is no device detected,
    // XLNX_DDR_OR_HBM.empty() == true, xrt does not exists at
    // all. so return DDR directly
    return true;
  }
  CHECK_LT(device_id, xlnx_ddr_or_hbm.size())
      << " we must detect hbm or ddr somewhere, or by settting env variable "
         "XLNX_DDR_OR_HBM. "
      << "for example: XLNX_DDR_OR_HBM=DDR,HBM";
  return xlnx_ddr_or_hbm[device_id] == "DDR";
}

DpuSessionImp::DpuSessionImp(const std::string& filename,
                             const std::string& kernel)
    : vart::dpu::DpuSessionBaseImp(nullptr) {
  auto dpu_name = dpu_controller_->get_kernel_name(get_device_core_id());
  auto device_id = dpu_controller_->get_device_id(device_core_id_);
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "create dpu session @" << (void*)this << " "
      << "device_core_id_ " << get_device_core_id() << " "  //
      << "device_id " << device_id << " "
      << "is_ddr " << is_ddr(device_id) << " "
      << "dpu_name " << dpu_name;
  if (is_ddr(device_id)) {
    // for ddr arch, we only load kernel per device;
    const std::string key =
        filename + ":" + kernel + ":" + std::to_string(device_id);
    kernel_ =
        vitis::ai::WeakStore<std::string, vart::dpu::DpuKernelDdr>::create(
            key, filename, kernel, dpu_controller_.get(), get_device_core_id());
  } else {
    // for hbm arch, we only load kernel per device_core;
    const std::string key =
        filename + ":" + kernel + ":" + std::to_string(device_core_id_);
    kernel_ =
        vitis::ai::WeakStore<std::string, vart::dpu::DpuKernelHbm>::create(
            key, filename, kernel,
            // dpu_controller and device_core_id are not used yet
            dpu_controller_.get(), get_device_core_id());
  }
}

DpuSessionImp::DpuSessionImp(const xir::Subgraph* subgraph, xir::Attrs* attrs)
    : vart::dpu::DpuSessionBaseImp(attrs) {
  auto dpu_name = dpu_controller_->get_kernel_name(get_device_core_id());
  auto device_id = dpu_controller_->get_device_id(device_core_id_);
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "create dpu session @" << (void*)this << " "
      << "device_core_id_ " << get_device_core_id() << " "  //
      << "device_id " << device_id << " "
      << "is_ddr " << is_ddr(device_id) << " "
      << "dpu_name " << dpu_name;
  const std::string key = std::string("subgraph:") +
                          std::to_string((uintptr_t)subgraph) +
                          std::to_string(get_device_core_id());
  if (is_ddr(device_id)) {
    // for ddr arch, we only load kernel per device;
    const std::string key = std::string("subgraph:") +
                            std::to_string((uintptr_t)subgraph) +
                            std::to_string(device_id);
    kernel_ =
        vitis::ai::WeakStore<std::string, vart::dpu::DpuKernelDdr>::create(
            key, *subgraph, attrs, dpu_controller_.get(), get_device_core_id());
  } else {
    // for hbm arch, we only load kernel per device_core;
    const std::string key = std::string("subgraph:") +
                            std::to_string((uintptr_t)subgraph) +
                            std::to_string(get_device_core_id());
    kernel_ =
        vitis::ai::WeakStore<std::string, vart::dpu::DpuKernelHbm>::create(
            key, *subgraph, attrs, dpu_controller_.get(), get_device_core_id());
  }
}

DpuSessionImp::~DpuSessionImp() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << " destroy dpu session @" << (void*)this;
}

static std::vector<std::string> get_tensor_names(
    const std::vector<const xir::Tensor*>& tensors) {
  auto ret = std::vector<std::string>(tensors.size());
  for (auto i = 0u; i < tensors.size(); ++i) {
    ret[i] = tensors[i]->get_name();
  }
  return ret;
}

void DpuSessionImp::initialize() {
  DpuSessionBaseImp::initialize();
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << " create dpu runner @ " << (void*)this                             //
      << " device_id= " << dpu_controller_->get_device_id(device_core_id_)  //
      << " device_core_id=" << device_core_id_                              //
      ;

  set_subgraph_specific_attrs();
  all_tensor_buffers_ = init_tensor_buffer(my_all_tensors_);

  input_tensor_buffers_ =
      find_tensor_buffer(get_tensor_names(get_input_tensors()));
  output_tensor_buffers_ =
      find_tensor_buffer(get_tensor_names(get_output_tensors()));
  reg_base_ = find_reg_tensor_buffer();
}

std::unique_ptr<vart::Runner> DpuSessionImp::create_runner() {
  std::unique_ptr<vart::Runner> ret = nullptr;
  auto device_id = dpu_controller_->get_device_id(device_core_id_);
  if (is_ddr(device_id)) {
    ret = std::make_unique<vart::dpu::DpuRunnerDdr>(get_input_tensors(),
                                                    get_output_tensors(), this);
  } else {
    ret = std::make_unique<vart::dpu::DpuRunnerHbm>(get_input_tensors(),
                                                    get_output_tensors(),  //
                                                    get_device_core_id(), this);
  }
  return ret;
}

std::vector<vart::TensorBuffer*> DpuSessionImp::get_inputs() {
  return input_tensor_buffers_;
}

std::vector<vart::TensorBuffer*> DpuSessionImp::get_outputs() {
  return output_tensor_buffers_;
}

void DpuSessionImp::set_subgraph_specific_attrs() {
  UNI_LOG_CHECK(attrs_ != nullptr, VART_NULL_PTR);
  auto device_id = dpu_controller_->get_device_id(device_core_id_);
  attrs_->set_attr<int>(
      kernel_->get_subgraph()->get_name() + ":__tensor_buffer_location__",
      is_ddr(device_id) ? (int)vart::TensorBuffer::location_t::HOST_PHY
                        : /* TODO: HBM should return DEVICE_? in the future */
          (int)vart::TensorBuffer::location_t::HOST_VIRT);
  attrs_->set_attr<std::string>(
      kernel_->get_subgraph()->get_name() + ":__cu_name__",
      dpu_controller_->get_full_name(device_core_id_));
}

std::vector<std::unique_ptr<vart::TensorBuffer>>
DpuSessionImp::init_tensor_buffer(std::vector<my_tensor_t>& my_tensors) {
  auto ret = std::vector<std::unique_ptr<vart::TensorBuffer>>();
  ret.reserve(my_tensors.size());
  auto xir_tensors = std::vector<const xir::Tensor*>();
  xir_tensors.reserve(my_tensors.size());
  for (auto i = 0u; i < my_tensors.size(); ++i) {
    auto tensor = my_tensors[i].get_tensor();
    if (tensor->has_attr("reg_id") && tensor->has_attr("ddr_addr")) {
      xir_tensors.emplace_back(const_cast<xir::Tensor*>(tensor));
    }
  }
  auto allocator = vart::assistant::TensorBufferAllocator::create(attrs_);
  return allocator->allocate(kernel_->get_subgraph(), xir_tensors, {}).first;
}

std::vector<vart::TensorBuffer*> DpuSessionImp::find_tensor_buffer(
    const std::vector<std::string>& names) {
  auto ret = std::vector<vart::TensorBuffer*>(names.size());
  for (auto i = 0u; i < names.size(); ++i) {
    for (auto j = 0u; j < all_tensor_buffers_.size(); ++j) {
      if (names[i] == all_tensor_buffers_[j]->get_tensor()->get_name()) {
        ret[i] = all_tensor_buffers_[j].get();
        break;
      }
    }
    UNI_LOG_CHECK(ret[i], VART_TENSOR_INFO_ERROR)
      << "cannot find tensor name. name=" << names[i];
  }
  return ret;
}

std::vector<vart::TensorBuffer*> DpuSessionImp::find_reg_tensor_buffer() {
  auto ret = std::vector<vart::TensorBuffer*>();
  ret.reserve(8u);
  for (auto j = 0u; j < all_tensor_buffers_.size(); ++j) {
    if (all_tensor_buffers_[j]->get_tensor()->get_name().find("__reg__") == 0) {
      ret.emplace_back(all_tensor_buffers_[j].get());
    }
  }
  return ret;
}

}  // namespace dpu
}  // namespace vart

DECLARE_INJECTION(vart::dpu::DpuSession, vart::dpu::DpuSessionImp,
                  const std::string&, const std::string&);
DECLARE_INJECTION(vart::dpu::DpuSession, vart::dpu::DpuSessionImp,
                  const xir::Subgraph*&, xir::Attrs*&);
