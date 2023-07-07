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
#include "./sfm_control_xrt_edge.hpp"

#include <glog/logging.h>

#include <cmath>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>
DEF_ENV_PARAM(DEBUG_SFM_RUNNER, "0");
DEF_ENV_PARAM(XLNX_ENABLE_HW_SMFC, "1");
DEF_ENV_PARAM_2(XLNX_SMFC_BUFFER_SIZE, "5242880", size_t);
#include <ert.h>
namespace {

std::unique_ptr<xir::XrtCu> make_xrt_cu() {
  if (xclProbe() > 0) {
    auto ret = std::make_unique<xir::XrtCu>(std::string{"sfm_xrt_top"});
    if (ret->get_num_of_cu() > 0u) {
      return ret;
    }
  }
  return nullptr;
}

static std::unique_ptr<xir::BufferObject> create_workspace(
    const xir::XrtCu* xrt_cu) {
  if (xrt_cu == nullptr) {
    return nullptr;
  }
  auto idx = 0u;
  const auto fullname = xrt_cu->get_full_name(idx);
  size_t device_id = xrt_cu->get_device_id(idx);
  size_t buffer_size = ENV_PARAM(XLNX_SMFC_BUFFER_SIZE);
  // (sizeof(int8_t) + sizeof(float)) * 1024u * 1024u;
  return xir::BufferObject::create(buffer_size, device_id, fullname);
}

SfmControllerXrtEdge::SfmControllerXrtEdge(size_t core_idx,
                                           std::unique_ptr<xir::XrtCu>&& xrt_cu)
    : xir::SfmController{},
      core_idx_{core_idx},
      page_size_(
          #ifdef _WIN32
          4096
          #else
          getpagesize()
        #endif
      ),
      xrt_cu_{std::move(xrt_cu)},
      workspace_{create_workspace(xrt_cu_.get())},
      mutex_{} {}

SfmControllerXrtEdge::~SfmControllerXrtEdge() {}

static size_t align(size_t a, size_t b) {
  if (a % b == 0) {
    return a;
  }
  return (a / b + 1) * b;
}
void SfmControllerXrtEdge::run(const int8_t* input, float scale,
                               unsigned int cls, unsigned int group,
                               float* output) {
  CHECK(supported(scale, cls, group))
      << "scale=" << scale << " cls=" << cls << " group=" << group
      << " num_of_cus: "
      << (xrt_cu_ == nullptr ? std::string("N/A")
                             : std::to_string(xrt_cu_->get_num_of_cu()))
      << " ENV_PARAM(XLNX_ENABLE_HW_SMFC) = " << ENV_PARAM(XLNX_ENABLE_HW_SMFC);
  std::lock_guard<std::mutex> lock(mutex_);
  auto exCls = align(cls, 4u);
  do {
    auto workspace_size = workspace_->size();
    auto input_size = exCls * group * sizeof(int8_t);
    auto input_aligned_size = align(input_size, page_size_);
    auto output_size = exCls * group * sizeof(float);
    auto output_aligned_size = align(output_size, page_size_);
    auto total_aligned_size = input_aligned_size + output_aligned_size;
    // The hw softmax could only compute 65535 rows data as a group,
    // so we store the amount of groups into n_of_group
    auto n_of_run = total_aligned_size / workspace_size +
                    (total_aligned_size % workspace_size == 0 ? 0 : 1);
    auto n_of_group = group / MAX_GROUP + (group % MAX_GROUP == 0 ? 0 : 1);
    n_of_run = std::max(n_of_run, n_of_group);
    LOG_IF(INFO, ENV_PARAM(DEBUG_SFM_RUNNER)) << "pages: " << n_of_run;

    // The output start address will be set after the input's tail address,
    // the split_offset store the batch input data size(aligned by page_size).
    auto batch_per_run = group / n_of_run;
    auto split_offset =
        align(exCls * batch_per_run * sizeof(int8_t), page_size_);
    auto last_batch = group - (n_of_run - 1) * batch_per_run;
    auto workspace_input = workspace_->get_w<char>(0);
    auto workspace_output = workspace_->get_r<char>(split_offset);
    // get the physical address value, this int64 number will be used soon.
    auto input_phy = workspace_->phy(0);
    auto output_phy = input_phy + split_offset;
    int fix_pos = -(int8_t)log2f(scale);
    uint32_t offset = 0u;

    for (auto r = 0u; r < n_of_run; ++r) {
      // these input and output are the actually points of "workspace_"
      auto r_input = input + (cls * batch_per_run * r);
      auto r_output = output + (cls * batch_per_run * r);
      auto this_batch = r == n_of_run - 1 ? last_batch : batch_per_run;
      auto this_input_size = this_batch * exCls * sizeof(int8_t);
      auto this_output_size = this_batch * exCls * sizeof(float);
      // workspace_input and workspace_output are the actually position on
      // the physical address, they should be set or fetch by the align-rule.
      if (cls == exCls) {
        memcpy(workspace_input, r_input, this_input_size);
      } else {
        for (auto i = 0u; i < this_batch; ++i) {
          memcpy(&workspace_input[exCls * i], &r_input[cls * i],
                 cls * sizeof(int8_t));
        }
      }
      workspace_->sync_for_write(0, this_input_size);
      auto xrt_cls = cls;
      auto xrt_batch = this_batch;
      auto xrt_input = input_phy;
      auto xrt_output = output_phy;
      auto xrt_fixpos = fix_pos;
      auto xrt_offset = offset;
      run_xrt_cu(core_idx_, xrt_input, xrt_cls, xrt_batch, xrt_fixpos,
                 xrt_output, xrt_offset);
      LOG_IF(INFO, ENV_PARAM(DEBUG_SFM_RUNNER))
          << "group " << group << " "                                //
          << "cls " << cls << " "                                    //
          << "exnCls " << exCls << " "                               //
          << "this_batch " << this_batch << " "                      //
          << "last_batch " << last_batch << " "                      //
          << "batch_per_run " << batch_per_run << " "                //
          << "n_of_run " << n_of_run << " "                          //
          << "r " << r << " "                                        //
          << "fix_pos " << fix_pos << " "                            //
          << "offset " << offset << " "                              //
          << "input_phy " << std::hex << "0x" << input_phy << " "    //
          << "output_phy " << std::hex << "0x" << output_phy << " "  //
          << "split_offset " << std::hex << "0x" << split_offset     //
          << std::dec;
      workspace_->sync_for_read(split_offset, this_output_size);
      if (cls == exCls) {
        memcpy(r_output, workspace_output, this_output_size);
      } else {
        for (auto i = 0u; i < this_batch; ++i) {
          memcpy((char*)(&r_output[cls * i]),
                 (char*)(&workspace_output[exCls * i * sizeof(float)]),
                 cls * sizeof(float));
        }
      }
      if (0) {
        CHECK(std::ofstream(std::string{"dump/softmax_"} + std::to_string(r) +
                            ".in")
                  .write((char*)r_input, this_input_size)
                  .good());
        CHECK(std::ofstream(std::string{"dump/softmax_"} + std::to_string(r) +
                            ".out")
                  .write((char*)r_output, this_output_size)
                  .good());
      }
    }
  } while (0);
}

void SfmControllerXrtEdge::run_xrt_cu(size_t core_idx, const uint64_t input,
                                      const unsigned int cls,
                                      const unsigned int group,
                                      const int fixpos, uint64_t output,
                                      uint32_t offset) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_SFM_RUNNER))
      << std::hex                               //
      << "core_idx " << core_idx << " "         //
      << "input: " << input << std::dec << " "  //
      << "cls: " << cls << " "
      << "group: " << group << " "
      << "scale: " << fixpos << " ";
  auto func = [=](ert_start_kernel_cmd* ecmd) -> void {
    auto rsz = 60u;
    ecmd->count = 1 + rsz;
    ecmd->data[XSFM_CONTROL_AP] = 0x0;
    ecmd->data[XSFM_CONTROL_IER / 4] =
        0x1;  // must enable this, otherwise, sfmx can only used once.
    ecmd->data[1] = 0x1;  // GLBL_IRQ_ENA(Global Interrupt Enable Register)
    ecmd->data[2] = 0x1;  // IP_IRQ_ENA(IP Interrupt Enable Register)
    ecmd->data[3] = 0x0;  // IP_IRQ_STS(IP Interrupt Status Register)

    ecmd->data[XSFM_X_LEN / 4] = cls & 0xFFFFFFFF;
    ecmd->data[XSFM_Y_LEN / 4] = group & 0xFFFFFFFF;
    ecmd->data[XSFM_SRC_ADDR_L / 4] = input & 0xFFFFFFFF;
    ecmd->data[XSFM_SRC_ADDR_H / 4] = (input >> 32) & 0xFFFFFFFF;
    ecmd->data[XSFM_SCALE / 4] = fixpos & 0xFFFFFFFF;
    ecmd->data[XSFM_DST_ADDR_L / 4] = output & 0xFFFFFFFF;
    ecmd->data[XSFM_DST_ADDR_H / 4] = (output >> 32) & 0xFFFFFFFF;
    ecmd->data[XSFM_OFFSET / 4] = *((uint32_t*)((void*)(&offset)));
  };
  xrt_cu_->run(
      core_idx, func,
      // on_success
      [core_idx](xclDeviceHandle handle, uint64_t cu_addr) -> void {
        LOG_IF(INFO, ENV_PARAM(DEBUG_SFM_RUNNER))
            << "core_idx = " << core_idx << "\n";
      },
      // on failure
      [core_idx](xclDeviceHandle handle, uint64_t cu_addr) -> void {
        LOG(FATAL) << "sfm timeout! "
                   << "core_idx = " << core_idx << "\n";
      });
}

bool SfmControllerXrtEdge::supported(float scale, unsigned int cls,
                                     unsigned int group) const {
  auto ret = ENV_PARAM(XLNX_ENABLE_HW_SMFC) != 0;
  int fix_pos = -(int8_t)log2f(scale);
  ret = ret && xrt_cu_ != nullptr;
  ret = ret && xrt_cu_->get_num_of_cu() > 0u;
  ret = ret && fix_pos >= CUR_SCALE;
  ret = ret && cls <= MAX_CLS;
  return ret;
}

SfmControllerXrtEdgeWithScheduler::SfmControllerXrtEdgeWithScheduler() {
  auto first_xrt_cu = make_xrt_cu();
  if (first_xrt_cu != nullptr) {
    auto num_of_cus = first_xrt_cu->get_num_of_cu();
    controllers_.emplace_back(
        std::make_unique<SfmControllerXrtEdge>(0u, std::move(first_xrt_cu)));
    for (auto core_idx = 1u; core_idx < num_of_cus; ++core_idx) {
      auto xrt_cu = make_xrt_cu();
      CHECK(xrt_cu != nullptr);
      controllers_.emplace_back(
          std::make_unique<SfmControllerXrtEdge>(core_idx, std::move(xrt_cu)));
    }
  }
  // scheduler_ = vart::dpu::DeviceScheduler::create(controllers_.size());
  return;
}

void SfmControllerXrtEdgeWithScheduler::run(const int8_t* input, float scale,
                                            unsigned int cls,
                                            unsigned int group, float* output) {
  auto core_idx = 0;  // scheduler_->next();
  controllers_[core_idx]->run(input, scale, cls, group, output);
  // scheduler_->mark_busy_time(core_idx, -1);
}

void SfmControllerXrtEdgeWithScheduler::run_xrt_cu(
    size_t core_idx_, const uint64_t input, const unsigned int cls,
    const unsigned int group, const int fixpos, uint64_t output,
    uint32_t offset) {
  auto core_idx = 0;  // scheduler_->next();
  controllers_[core_idx]->run_xrt_cu(core_idx, input, cls, group, fixpos,
                                     output, offset);
  // scheduler_->mark_busy_time(core_idx, -1);
}

bool SfmControllerXrtEdgeWithScheduler::supported(float scale, unsigned int cls,
                                                  unsigned int group) const {
  auto ret = !controllers_.empty();
  for (auto& c : controllers_) {
    ret = ret && c->supported(scale, cls, group);
  }
  return ret;
};

}  // namespace

namespace xir {
std::shared_ptr<SfmController> SfmController::get_instance() {
  return vitis::ai::WeakSingleton<SfmControllerXrtEdgeWithScheduler>::create();
}
}  // namespace xir
