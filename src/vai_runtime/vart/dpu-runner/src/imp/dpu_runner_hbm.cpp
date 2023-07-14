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
#include "dpu_runner_hbm.hpp"

#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <fstream>
#include <thread>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>
#include <xir/tensor/tensor.hpp>

#include "vart/tensor_buffer.hpp"
DEF_ENV_PARAM(XLNX_ENABLE_DUMP, "0");
DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0");
DEF_ENV_PARAM(XLNX_SHORT_CIRCUIT_DPU_CODE, "0");
DEF_ENV_PARAM(XLNX_ENABLE_DEBUG_MODE, "0");
DEF_ENV_PARAM(DEBUG_DPU_RUNNER_2, "0");

namespace vart {
namespace dpu {

static std::vector<std::shared_ptr<DpuCore>> get_cloud_dpu_cores(
    xir::DpuController* dpu_controller) {
  auto size = dpu_controller->get_num_of_dpus();
  auto ret = std::vector<std::shared_ptr<DpuCore>>();
  ret.reserve(size);
  for (auto device_core_id = 0u; device_core_id < size; ++device_core_id) {
    ret.emplace_back(vitis::ai::WeakStore<size_t, DpuCore>::create(
        device_core_id, dpu_controller->get_core_id(device_core_id)));
  }
  return ret;
}

DpuRunnerHbm::DpuRunnerHbm(const std::vector<const xir::Tensor*> input_tensors,
                           const std::vector<const xir::Tensor*> output_tensors,
                           const size_t device_core_id,
                           DpuSessionBaseImp* session)
    : vart::dpu::DpuRunnerBaseImp(input_tensors, output_tensors, session),
      cores_(get_cloud_dpu_cores(session->get_dpu_controller())),
      device_core_id_{device_core_id},  //
      device_memory_{} {
  auto device_id =
      session->get_dpu_controller()->get_device_id(device_core_id_);
  device_memory_ = vitis::ai::WeakStore<size_t, xir::DeviceMemory>::create(
      device_id, device_id);
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << " create dpu runner @ " << (void*)this  //
      << " device_id= " << device_id             //
      << " device_core_id=" << device_core_id_   //
      ;
}

DpuRunnerHbm::~DpuRunnerHbm() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << " destroy dpu runner @ " << (void*)this  //
      << " device_core_id=" << device_core_id_    //
      ;
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
  CHECK_NE(ret, -1) << "cannot find tensor! name=" << name;
  return ret;
}

std::pair<uint32_t, int> DpuRunnerHbm::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  auto my_input_tensor_buffers = session_->get_inputs();
  auto my_output_tensor_buffers = session_->get_outputs();
  // CHECK_EQ(my_output_tensor_buffers.size(), output.size());
  CHECK(my_input_tensor_buffers.size() == input.size());
  {
    // begin copy input
    auto input_size = my_input_tensor_buffers.size();
    for (auto input_idx = 0u; input_idx < input_size; ++input_idx) {
      if (my_input_tensor_buffers[input_idx] != input[input_idx]) {
        vart::TensorBuffer::copy_tensor_buffer(
            input[input_idx], my_input_tensor_buffers[input_idx]);
      }
    }
    // end copy input
  }
  auto ret = std::make_pair<uint32_t, int>(1u, 0);

  {
    auto device_core_id = device_core_id_;  // scheduler_->next();
    // hold workspace lock
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
        << "trying to obtain lock for workspace for device_cu_core["
        << device_core_id << "]";
    auto w = cores_[device_core_id]->lock_workspace();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
        << "abtained lock for workspace[" << w->get_workspace_id() << "]";
    auto workspace_regs = session_->get_kernel()->get_workspace_regs();
    chunks_ = w->get_workspaces(workspace_regs);
    //
    upload_data(my_input_tensor_buffers, chunks_, device_core_id);
    {
      // auto lock = w->lock_core();

      // start_dpu2() invokes fill_gen_reg(), so we have to prepare
      // chunks_ before that.
      start_dpu2(device_core_id);

      /// device_scheduler_->mark_busy_time(device_core_id, -1 /*release the
      /// core*/);
    }
    download_data(my_output_tensor_buffers, chunks_, device_core_id);
  }
  {
    // begin copy output
    auto output_size = output.size();
    for (auto output_idx = 0u; output_idx < output_size; ++output_idx) {
      // TODO return a pointer
      auto my_output_idx =
          find_tensor_index(my_output_tensor_buffers,
                            output[output_idx]->get_tensor()->get_name());
      if (my_output_tensor_buffers[my_output_idx] != output[output_idx]) {
        vart::TensorBuffer::copy_tensor_buffer(
            my_output_tensor_buffers[my_output_idx], output[output_idx]);
      }
    }
    // end copy output
  }
  return ret;
}

void DpuRunnerHbm::upload_data(
    const std::vector<vart::TensorBuffer*>& input,
    // chunks[engine_id][reg_id].get();
    const std::vector<std::map<std::string, std::unique_ptr<HbmChunk>>>& chunks,
    size_t device_core_id) {
  CHECK_EQ(session_->get_my_input_tensors().size(), input.size());
  // upload data
  for (auto idx = 0u; idx < input.size(); ++idx) {
    auto tensor_buffer = input[idx];
    const auto& my_tensor = session_->get_my_input_tensors()[idx];
    auto reg_idx = my_tensor.get_reg_id();
    auto ddr_addr = my_tensor.get_ddr_addr();
    auto ddr_size = my_tensor.get_size();
    for (auto batch_id = 0u; batch_id < get_batch_size(tensor_buffer);
         ++batch_id) {
      uint64_t data_addr = 0u;
      size_t data_size = 0;
      auto dim =
          std::vector<int>(tensor_buffer->get_tensor()->get_shape().size());
      dim[0] = batch_id;
      std::tie(data_addr, data_size) = tensor_buffer->data(dim);
      CHECK_GT(data_size, 0u);
      CHECK(data_addr != 0u);
      auto engine_id = batch_id;
      CHECK_LT((unsigned)engine_id, chunks.size()) << "batch id out of range.";
      auto reg_id = std::string("REG_") + std::to_string(reg_idx);
      const auto it_chunk = chunks[engine_id].find(reg_id);
      CHECK(it_chunk != chunks[engine_id].end());
      const auto chunk = it_chunk->second.get();
      CHECK(chunk != nullptr) << "cannot find chunk for engine_id=" << engine_id
                              << " reg_id=" << reg_id;
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))                   //
          << "upload data to dpu core " << device_core_id << " "  //
          << " batch_id=" << batch_id                             //
          << " size=" << ddr_size                                 //
          << " reg_id=" << reg_id                                 //
          << " data_addr=" << (void*)data_addr;
      chunk->upload(device_memory_.get(), (void*)data_addr, ddr_addr, ddr_size);
    }
  };
}

void DpuRunnerHbm::download_data(
    const std::vector<vart::TensorBuffer*>& output,
    const std::vector<std::map<std::string, std::unique_ptr<HbmChunk>>>&
        chunks,  // chunks[engine_id][reg_id].get();
    size_t device_core_id) {
  CHECK_EQ(session_->get_my_output_tensors().size(), output.size());
  // download data
  for (auto idx = 0u; idx < output.size(); ++idx) {
    auto* tensor_buffer = output[idx];
    const auto& my_tensor = session_->get_my_output_tensors()[idx];
    auto reg_idx = my_tensor.get_reg_id();  // TODO: change it to string
    auto ddr_addr = my_tensor.get_ddr_addr();
    auto ddr_size = my_tensor.get_size();
    for (auto batch_id = 0u; batch_id < get_batch_size(tensor_buffer);
         ++batch_id) {
      uint64_t data_addr = 0u;
      size_t data_size = 0;
      auto dim =
          std::vector<int>(tensor_buffer->get_tensor()->get_shape().size());
      dim[0] = batch_id;
      std::tie(data_addr, data_size) = tensor_buffer->data(dim);
      CHECK_GT(data_size, 0u);
      CHECK(data_addr != 0u);
      auto engine_id = batch_id;
      CHECK_LT((unsigned)engine_id, chunks.size()) << "batch id out of range.";
      auto reg_id = std::string("REG_") + std::to_string(reg_idx);
      const auto it_chunk = chunks[engine_id].find(reg_id);
      CHECK(it_chunk != chunks[engine_id].end());
      const auto chunk = it_chunk->second.get();
      CHECK(chunk != nullptr) << "cannot find chunk for engine_id=" << engine_id
                              << " reg_id=" << reg_id;
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))                       //
          << "download data from dpu core " << device_core_id << " "  //
          << " batch_id=" << batch_id                                 //
          << " size=" << ddr_size                                     //
          << " reg_id=" << reg_id << " virt_addr=" << (void*)data_addr;
      chunk->download(device_memory_.get(), (void*)data_addr, ddr_addr,
                      ddr_size);
    }
  };
}

int DpuRunnerHbm::wait(int jobid, int timeout) { return 0; }

constexpr size_t NUM_OF_DPU_REGS = 8;

static int get_reg_index(const std::string& reg_id) {
  CHECK(reg_id.size() >= 5 &&  //
        reg_id[0] == 'R' &&    //
        reg_id[1] == 'E' &&    //
        reg_id[2] == 'G' &&    //
        reg_id[3] == '_' &&    //
        reg_id[4] >= '0' && reg_id[4] <= '9')
      << "reg id is not support! reg_id = " << reg_id;
  return reg_id[4] - '0';
}
void DpuRunnerHbm::fill_gen_reg(size_t device_core_id,
                                std::vector<uint64_t>& gen_reg) {
  auto num_of_engines = session_->get_num_of_engines();
  CHECK_LE(num_of_engines, cores_[device_core_id]->get_num_of_engines())
      << ", please check hbm_address_assignment.txt";
  for (auto engine_id = 0u; engine_id < num_of_engines; ++engine_id) {
    for (const auto& reg_workspace : chunks_[engine_id]) {
      const auto& reg_id = reg_workspace.first;
      auto chunk = reg_workspace.second.get();
      CHECK(chunk != nullptr);
      auto reg_idx = get_reg_index(reg_id);
      auto idx = engine_id * NUM_OF_DPU_REGS + reg_idx;
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER) >= 2)
          << "debug "
          << "idx " << idx << " "                        //
          << "reg_idx " << reg_idx << " "                //
          << "num_of_engines " << num_of_engines << " "  //
          << "gen_reg.size() " << gen_reg.size() << " "  //
          << std::endl;
      gen_reg[idx] = chunk->get_offset();
    }
  }
  return;
}

}  // namespace dpu
}  // namespace vart
