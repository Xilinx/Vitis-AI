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

#include <UniLog/UniLog.hpp>
#include <chrono>
#include <limits>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/weak.hpp>
#include <xir/tensor/tensor.hpp>

#include "./dpu_session_imp.hpp"
#include "vart/tensor_buffer.hpp"
#include "vart/zero_copy_helper.hpp"
DEF_ENV_PARAM(XLNX_ENABLE_DUMP, "0");
DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0");
DEF_ENV_PARAM(DEBUG_DPU_WARMUP, "0");

namespace vart {
namespace dpu {
static void init_tensor_buffer(vart::TensorBuffer* tb, int is_output) {
  auto shape = tb->get_tensor()->get_shape();
  for (auto i = 0; i < shape[0]; ++i) {
    auto index = std::vector<int>(shape.size(), 0);
    index[0] = i;
    int64_t data;
    size_t size;
    std::tie(data, size) = tb->data(index);
    if (0) {
      for (auto s = 0u; s < size; ++s) {
        char* p = reinterpret_cast<char*>(data);
        p[s] = (char)(int)s;
      }
    }
    if (1) {
      if (is_output) {
        tb->sync_for_read(0u, size);
      } else {
        tb->sync_for_write(0u, size);
      }
    }
  }
}

DpuRunnerDdr::DpuRunnerDdr(const std::vector<const xir::Tensor*> input_tensors,
                           const std::vector<const xir::Tensor*> output_tensors,
                           DpuSessionBaseImp* session)
    : vart::dpu::DpuRunnerBaseImp(input_tensors, output_tensors, session) {
  for (auto i = 0; i < ENV_PARAM(DEBUG_DPU_WARMUP); ++i) {
    if (0)
      for (auto tb : session_->get_inputs()) {
        init_tensor_buffer(tb, 0);
      }
    if (0) execute_async(session_->get_inputs(), session_->get_outputs());
    if (1)
      for (auto tb : session_->get_outputs()) {
        init_tensor_buffer(tb, 1);
      }
    if (0)
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_WARMUP))
          << "dpu runner warmup " << (void*)this                          //
          << " device_core_id " << session_->get_device_core_id() << " "  //
          ;
  }
}

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
  UNI_LOG_CHECK(ret != -1, VART_TENSOR_INFO_ERROR)
      << "cannot find tensor! name=" << name;
  return ret;
}

static vart::TensorBuffer::location_t get_location(
    const std::vector<vart::TensorBuffer*>& input) {
  UNI_LOG_CHECK(!input.empty(), VART_SIZE_MISMATCH) << "input tensor is empty!";
  auto ret = input[0]->get_location();
  for (auto i = 1u; i < input.size(); ++i) {
    UNI_LOG_CHECK((int)ret == (int)input[i]->get_location(),
                  VART_TENSOR_INFO_ERROR)
        << " all tensor buffer must have same location: tensor="
        << input[i]->get_tensor()->to_string();
  }
  return ret;
}

void DpuRunnerDdr::maybe_copy_input(
    vart::TensorBuffer::location_t location,
    const std::vector<vart::TensorBuffer*>& input) {
  auto my_input_tensor_buffers = session_->get_inputs();
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
  }
}

void DpuRunnerDdr::prepare_input_for_reg(
    vart::TensorBuffer::location_t location,
    const std::vector<vart::TensorBuffer*>& tensor_buffers,
    std::vector<vart::TensorBuffer*>& ret) {
  if (location == TensorBuffer::location_t::HOST_VIRT) {
  } else {
    // zero copy, do nothing except sanity checking
    auto device_id = 0u;
    if (location == vart::TensorBuffer::location_t::HOST_PHY) {
      // only support single device on edge
      device_id = 0u;
    } else {
      // TODO: on board zero copy is not implemented well yet. there
      // is no such user scenario yet.
      //
      // no need to check all inputs and outputs must be on the same
      // board, because location is the same as location_output.
      //
      device_id =
          (size_t)location - (size_t)vart::TensorBuffer::location_t::DEVICE_0;
    }
    auto device_core_id = session_->get_device_core_id();
    UNI_LOG_CHECK(device_id == session_->get_dpu_controller()->get_device_id(
                                   device_core_id),
                  VART_DEVICE_MISMATCH);
    ret.insert(ret.end(), tensor_buffers.begin(), tensor_buffers.end());
  }
}

std::vector<vart::TensorBuffer*> DpuRunnerDdr::prepare_input(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  auto ret = std::vector<vart::TensorBuffer*>{};
  auto session = dynamic_cast<DpuSessionImp*>(session_);
  UNI_LOG_CHECK(session != nullptr, VART_NULL_PTR)
      << "session = " << (void*)session_;
  auto reg_base = session->get_reg_base();
  // first add my bases, and overwrite by `input` or `output` if any,
  // see fillin_reg_reg for detail
  ret.insert(ret.end(), reg_base.begin(), reg_base.end());

  auto location_input = get_location(input);
  maybe_copy_input(location_input, input);
  prepare_input_for_reg(location_input, input, ret);

  auto location_output = get_location(output);
  prepare_input_for_reg(location_output, output, ret);
  return ret;
}

std::pair<uint32_t, int> DpuRunnerDdr::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  __TIC__(DPU_RUNNER_COPY_INPUT);
  UNI_LOG_CHECK(my_input_.empty(), VART_SIZE_MISMATCH);
  my_input_ = prepare_input(input, output);
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

static size_t get_reg_id(const xir::Tensor* tensor) {
  auto ret = std::numeric_limits<size_t>::max();
  UNI_LOG_CHECK(tensor->has_attr("reg_id"), VART_TENSOR_INFO_ERROR)
      << "tensor.name" << tensor->get_name();
  ret = (size_t)tensor->get_attr<int>("reg_id");
  return ret;
}

static int get_ddr_addr(const xir::Tensor* tensor) {
  auto ret = -1;
  UNI_LOG_CHECK(tensor->has_attr("ddr_addr"), VART_TENSOR_INFO_ERROR);
  UNI_LOG_CHECK(tensor->has_attr("location"), VART_TENSOR_INFO_ERROR);
  // not on-chip, TODO: rename 1 to a const symbol
  UNI_LOG_CHECK(tensor->get_attr<int>("location") == 1, VART_TENSOR_INFO_ERROR);
  ret = tensor->get_attr<int>("ddr_addr");
  return ret;
}

void DpuRunnerDdr::fill_gen_reg(size_t device_core_id,
                                std::vector<uint64_t>& gen_reg) {
  UNI_LOG_CHECK(my_input_.size() != 0u, VART_SIZE_MISMATCH);
  auto num_of_batch = (int)session_->get_num_of_engines();
  auto num_of_regs =
      const_cast<const xir::DpuController*>(session_->get_dpu_controller())
          ->get_size_of_gen_regs(device_core_id);
  auto& reg_base = my_input_;
  for (auto reg_base_idx = 0u; reg_base_idx < reg_base.size(); ++reg_base_idx) {
    auto& reg = reg_base[reg_base_idx];
    auto dims = reg->get_tensor()->get_shape();
    auto dim_idx = std::vector<int32_t>(dims.size());
    auto reg_id = get_reg_id(reg->get_tensor());
    UNI_LOG_CHECK(reg_id >= 0 && reg_id < MAX_REG_ID_SIZE, VART_DPU_INFO_ERROR)
        << "reg_id = " << reg_id
        << ", tensor_name = " << reg->get_tensor()->get_name();
    int ddr_addr = get_ddr_addr(reg->get_tensor());
    auto base_reg_flag = reg->get_tensor()->get_name().find("__reg__") == 0;
    auto reg_batch = base_reg_flag ? num_of_batch : dims[0];
    for (auto batch_idx = 0; batch_idx < reg_batch; ++batch_idx) {
      auto idx2 = std::min(batch_idx, reg->get_tensor()->get_shape()[0] - 1);
      dim_idx[0] = idx2;
      uint64_t base;
      size_t size;
      std::tie(base, size) = reg->data_phy(dim_idx);
      UNI_LOG_CHECK(size != 0u, VART_DPU_ALLOC_ERROR)
          << "data_phy size is 0, please check!";
      // move get_ddr_addr() out of this loop to improve perf;
      // int ddr_addr = get_ddr_addr(reg->get_tensor());
      base = base - ddr_addr;
      auto reg_idx = batch_idx * num_of_regs + reg_id;
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER) >= 2)
          << "set base reg: " << reg_idx                            //
          << " num_of_regs: " << num_of_regs                        //
          << " reg_id: " << reg_id                                  //
          << " batch_idx: " << batch_idx                            //
          << " num_of_batch " << num_of_batch                       //
          << " / " << (reg->get_tensor()->get_shape()[0] - 1)       //
          << " base = " << std::hex << "0x" << base << std::dec     //
          << " ddr = " << std::hex << "0x" << ddr_addr << std::dec  //
          << " tensor " << reg->get_tensor()->to_string()           //
          ;
      gen_reg[reg_idx] = base;
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
