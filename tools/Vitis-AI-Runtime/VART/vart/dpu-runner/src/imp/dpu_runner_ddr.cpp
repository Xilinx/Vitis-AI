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
#include "dpu_runner_ddr.hpp"

#include <glog/logging.h>

#include <chrono>
#include <limits>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/weak.hpp>

#include "./dpu_session_imp.hpp"
#include "vart/tensor_buffer.hpp"
DEF_ENV_PARAM(XLNX_ENABLE_DUMP, "0");
DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0");

namespace vart {
namespace dpu {

DpuRunnerDdr::DpuRunnerDdr(const std::vector<const xir::Tensor*> input_tensors,
                           const std::vector<const xir::Tensor*> output_tensors,
                           DpuSessionBaseImp* session)
    : vart::dpu::DpuRunnerBaseImp(input_tensors, output_tensors, session) {}

static int find_tensor_index(std::vector<vart::TensorBuffer*> tensor_buffers,
                             const std::string& name) {
  int ret = -1;
  for (auto i = 0u; i < tensor_buffers.size(); ++i) {
    if (tensor_buffers[i]->get_tensor()->get_name().find(name) !=
        std::string::npos) {
      ret = (int)i;
      break;
    }
  }
  CHECK_NE(ret, -1) << "cannot find tensor! name=" << name;
  return ret;
}

static vart::TensorBuffer::location_t get_location(
    const std::vector<vart::TensorBuffer*>& input) {
  CHECK(!input.empty());
  auto ret = input[0]->get_location();
  for (auto i = 1u; i < input.size(); ++i) {
    CHECK_EQ((int)ret, (int)input[i]->get_location())
        << " all tensor buffer must have same location: tensor="
        << input[i]->get_tensor()->to_string();
  }
  return ret;
}

std::vector<vart::TensorBuffer*> DpuRunnerDdr::prepare_input(
    const std::vector<vart::TensorBuffer*>& input) {
  auto my_input_tensor_buffers = session_->get_inputs();
  auto session = dynamic_cast<DpuSessionImp*>(session_);
  CHECK(session != nullptr) << "session = " << (void*)session_;
  auto reg_base = session->get_reg_base();
  auto location = get_location(input);
  if (location == TensorBuffer::location_t::HOST_VIRT) {
    for (auto input_idx = 0u; input_idx < input.size(); ++input_idx) {
      auto& input_bo = input[input_idx];
      auto my_input_idx = find_tensor_index(my_input_tensor_buffers,
                                            input_bo->get_tensor()->get_name());
      auto dpu_tensor_buffer = my_input_tensor_buffers[my_input_idx];
      copy_data_for_input(input_bo, dpu_tensor_buffer);
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
          << "copy input:" << dpu_tensor_buffer->to_string();
    }
  } else {
    // zero copy, do nothing except sanity checking
    auto device_id = 0u;
    if (location == vart::TensorBuffer::location_t::HOST_PHY) {
      // only support single device on edge
      device_id = 0u;
    } else {
      device_id =
          (size_t)location - (size_t)vart::TensorBuffer::location_t::DEVICE_0;
    }
    auto device_core_id = session_->get_device_core_id();
    CHECK_EQ(device_id,
             session_->get_dpu_controller()->get_device_id(device_core_id));
  }
  // first add my bases, and overwrite by `input` if any
  auto ret = std::vector<vart::TensorBuffer*>{};
  ret.insert(ret.end(), reg_base.begin(), reg_base.end());
  ret.insert(ret.end(), input.begin(), input.end());
  return ret;
}

std::pair<uint32_t, int> DpuRunnerDdr::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  __TIC__(DPU_RUNNER_COPY_INPUT);
  CHECK(my_input_.empty());
  my_input_ = prepare_input(input);
  __TOC__(DPU_RUNNER_COPY_INPUT);
  __TIC__(DPU_RUNNER)
  start_dpu2(session_->get_device_core_id());
  __TOC__(DPU_RUNNER)
  __TIC__(DPU_RUNNER_COPY_OUTPUT);
  prepare_output(output);
  __TOC__(DPU_RUNNER_COPY_OUTPUT);
  my_input_.clear();
  return std::make_pair<uint32_t, int>(1u, 0);
}

void DpuRunnerDdr::prepare_output(
    const std::vector<vart::TensorBuffer*>& output) {
  auto my_output_tensor_buffers =
      dynamic_cast<vart::dpu::DpuSession*>(session_)->get_outputs();
  auto location = get_location(output);
  auto ret = std::vector<vart::TensorBuffer*>();
  if (location == TensorBuffer::location_t::HOST_VIRT) {
    for (auto output_idx = 0u; output_idx < output.size(); ++output_idx) {
      auto& output_bo = output[output_idx];
      auto my_output_idx = find_tensor_index(
          my_output_tensor_buffers, output_bo->get_tensor()->get_name());
      auto dpu_tensor_buffer = my_output_tensor_buffers[my_output_idx];
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
          << "copy_output:" << dpu_tensor_buffer->to_string();
      copy_data_for_output(output_bo, dpu_tensor_buffer);
      ret.emplace_back(dpu_tensor_buffer);
    }
  } else {
    // TODO: check device id == location
    // do nothing for zero copy
  }
}

int DpuRunnerDdr::wait(int jobid, int timeout) { return 0; }
static size_t parse_reg_id(const std::string& name) {
  // it looks like "__reg__1__"
  if (name.find("__reg__") == 0) {
    auto ret = (int)(name[sizeof("__reg__") - 1] - '0');
    CHECK(ret >= 0 && ret < 16) << "ret = " << ret << " name=" << name;
    return (size_t)ret;
  }
  return std::numeric_limits<size_t>::max();
}

void DpuRunnerDdr::fill_gen_reg(size_t device_core_id,
                                std::vector<uint64_t>& gen_reg) {
  CHECK_NE(my_input_.size(), 0u);
  auto num_of_batch = (int)session_->get_num_of_engines();
  auto num_of_regs =
      const_cast<const xir::DpuController*>(session_->get_dpu_controller())
          ->get_size_of_gen_regs(device_core_id);
  auto& reg_base = my_input_;
  for (auto reg_base_idx = 0u; reg_base_idx < reg_base.size(); ++reg_base_idx) {
    auto& reg = reg_base[reg_base_idx];
    auto dims = reg->get_tensor()->get_shape();
    auto dim_idx = std::vector<int32_t>(dims.size());
    auto reg_id = parse_reg_id(reg->get_tensor()->get_name());
    if (reg_id >= 0 && reg_id <= 16) {
      for (auto batch_idx = 0; batch_idx < num_of_batch; ++batch_idx) {
        auto idx2 = std::min(batch_idx, reg->get_tensor()->get_shape()[0] - 1);
        dim_idx[0] = idx2;
        uint64_t base;
        size_t size;
        std::tie(base, size) = reg->data_phy(dim_idx);
        auto reg_idx = batch_idx * num_of_regs + reg_id;
        // allocator need overwrite reg infos
        // CHECK_EQ(gen_reg[reg_idx], std::numeric_limits<uint64_t>::max())
        //    << "cannot write the reg twice."
        //    << "reg_idx " << reg_idx << " "  //
        //     << " batch_idx=" << batch_idx    //
        //    << " reg_id=" << reg_id          //
        // ;
        //
        gen_reg[reg_idx] = base;
      }
    }
  }
}

void DpuRunnerDdr::copy_data_for_input(vart::TensorBuffer* tb_from,
                                       vart::TensorBuffer* tb_to) {
  vart::TensorBuffer::copy_tensor_buffer(tb_from, tb_to);
  return;
}

void DpuRunnerDdr::copy_data_for_output(vart::TensorBuffer* tb_to,
                                        vart::TensorBuffer* tb_from) {
  vart::TensorBuffer::copy_tensor_buffer(tb_from, tb_to);
  return;
}

}  // namespace dpu
}  // namespace vart
