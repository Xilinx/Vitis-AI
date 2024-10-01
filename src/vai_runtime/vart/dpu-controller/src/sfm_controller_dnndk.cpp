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
#include <fcntl.h>
#include <glog/logging.h>
#include <math.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <fstream>
#include <mutex>
#include <vitis/ai/env_config.hpp>
#include <xir/buffer_object.hpp>
DEF_ENV_PARAM(DEBUG_SFM_CONTROLLER, "0");
DEF_ENV_PARAM_2(XLNX_SMFC_BUFFER_SIZE, "5242880", size_t);
#include "../../buffer-object/src/dpu.h"
#include "xir/sfm_controller.hpp"

namespace xir {

static size_t align(size_t a, size_t b) {
  if (a % b == 0) {
    return a;
  }
  return (a / b + 1) * b;
}

class SfmControllerDnndk : public xir::SfmController {
 public:
  SfmControllerDnndk(int fd);
  virtual ~SfmControllerDnndk() = default;
  SfmControllerDnndk(const SfmControllerDnndk& other) = delete;
  SfmControllerDnndk& operator=(const SfmControllerDnndk& rhs) = delete;

 public:
  virtual void run(const int8_t* input, float scale, unsigned int cls,
                   unsigned int group, float* output) override;
  virtual bool supported(float scale, unsigned int cls,
                         unsigned int group) const override;

 public:
  virtual void run_xrt_cu(size_t core_idx, const uint64_t input,
                          const unsigned int cls, const unsigned int group,
                          const int scale, uint64_t output,
                          uint32_t offset) override;

 private:
  int fd_;
  std::unique_ptr<xir::BufferObject> workspace_;
  std::mutex mtx_;
  size_t page_size_;
};

std::shared_ptr<SfmController> SfmController::get_instance() {
  // on x86 cloud environment, it might not open /dev/dpu, so
  // effectively, all HwSmFc is disabled.
  auto fd = open(
#ifdef __QNX__
      "/dev/xdpu/0",
#else
      "/dev/dpu",
#endif
      O_RDWR);
  if (fd < 0) {
    LOG_IF(WARNING, ENV_PARAM(DEBUG_SFM_CONTROLLER))
        << "cannot open /dev/dpu for smfc";
    return nullptr;
  }
  uint32_t flags = 0;
  auto retval = ioctl(fd, DPUIOC_G_INFO, (void*)(&flags));
  close(fd);
  auto sfm_num = SFM_NUM(flags);
  auto dpu_num = DPU_NUM(flags);

  CHECK_EQ(retval, 0) << "read sfm info failed.";
  LOG_IF(INFO, ENV_PARAM(DEBUG_SFM_CONTROLLER))
      << "sfm_num " << sfm_num << " "  //
      << "dpu_num " << dpu_num << " "  //
      ;
  auto softmax_valid = sfm_num > 0;
  if (!softmax_valid) {
    return nullptr;
  }
  // open it again for sfm controller
  fd = open(
#ifdef __QNX__
      "/dev/xdpu/0",
#else
      "/dev/dpu",
#endif
      O_RDWR);
  if (fd < 0) {
    LOG_IF(WARNING, ENV_PARAM(DEBUG_SFM_CONTROLLER))
        << "cannot open /dev/dpu for smfc";
    return nullptr;
  }
  return std::unique_ptr<SfmControllerDnndk>(new SfmControllerDnndk(fd));
}
size_t WORKSPACE_SIZE = (sizeof(int8_t) + sizeof(float)) * 1024u * 1024u;
static std::unique_ptr<xir::BufferObject> create_workspace() {
  size_t device_id = 0u;  // xrt_cu->get_device_id(idx);
  size_t buffer_size = ENV_PARAM(XLNX_SMFC_BUFFER_SIZE);
  const std::string fullname = "not-used";
  return xir::BufferObject::create(buffer_size, device_id, fullname);
}
SfmControllerDnndk::SfmControllerDnndk(int fd)
    : fd_{fd}, workspace_{create_workspace()}, mtx_{}, page_size_{0} {
  page_size_ = (decltype(page_size_))getpagesize();
}

bool SfmControllerDnndk::supported(float scale, unsigned int cls,
                                   unsigned int group) const {
  int fix_pos = -(int8_t)log2f(scale);
  auto ok = fix_pos >= 3;
  LOG_IF(INFO, ENV_PARAM(DEBUG_SFM_CONTROLLER))
      << "ok " << ok << " "            //
      << "fix_pos " << fix_pos << " "  //
      << "group " << group << " "      //
      << "cls " << cls << " "          //
      ;
  return ok;
}

void SfmControllerDnndk::run(const int8_t* input, float scale, unsigned int cls,
                             unsigned int group, float* output) {
  std::lock_guard<std::mutex> lock(mtx_);
  struct ioc_softmax_t reg;
  auto exCls = align(cls, 4u);
  do {
    auto workspace_size = workspace_->size();
    auto input_size = exCls * group * sizeof(int8_t);
    auto input_aligned_size = align(input_size, page_size_);
    auto output_size = exCls * group * sizeof(float);
    auto output_aligned_size = align(output_size, page_size_);
    auto total_aligned_size = input_aligned_size + output_aligned_size;
    auto n_of_run = total_aligned_size / workspace_size +
                    (total_aligned_size % workspace_size == 0 ? 0 : 1);
    auto split_offset = input_aligned_size;
    auto batch_per_run = group / n_of_run;
    auto last_batch = group - (n_of_run - 1) * batch_per_run;
    auto workspace_intput = workspace_->get_w<char>(0);
    auto workspace_output = workspace_->get_r<char>(split_offset);
    auto input_phy = workspace_->phy(0);
    auto output_phy = input_phy + split_offset;
    int fix_pos = -(int8_t)log2f(scale);
    uint32_t offset = 0u;

    for (auto r = 0u; r < n_of_run; ++r) {
      auto r_input = input + (cls * batch_per_run * r);
      auto r_output = output + (exCls * batch_per_run * r);
      auto this_batch = r == n_of_run - 1 ? last_batch : batch_per_run;
      auto this_input_size = this_batch * exCls * sizeof(int8_t);
      auto this_output_size = this_batch * exCls * sizeof(float);
      if (cls == exCls) {
        memcpy(workspace_intput, r_input, this_input_size);
      } else {
        for (auto i = 0u; i < batch_per_run; ++i) {
          memcpy(&workspace_intput[exCls * i], &r_input[cls * i],
                 cls * sizeof(int8_t));
        }
      }
      workspace_->sync_for_write(0, this_input_size);
      reg.width = cls;
      reg.height = this_batch;
      CHECK_EQ(input_phy & 0xFFFFFF0000000000, 0)
          << "invalid input_phy 0x" << std::hex << input_phy        //
          << ", only the low 40bits of physical address are valid"  //
          << ", high 24bits should be 0" << std::dec;
      CHECK_EQ(output_phy & 0xFFFFFF0000000000, 0)
          << "invalid output_phy 0x" << std::hex << output_phy      //
          << ", only the low 40bits of physical address are valid"  //
          << ", high 24bits should be 0" << std::dec;
      reg.input = input_phy;
      reg.output = output_phy;
      reg.scale = fix_pos;
      reg.offset = offset;
#ifdef __QNX__
      auto ioctl_ret = devctl(fd_, REQ_RUN_SOFTMAX, &reg, sizeof(reg), NULL);
      CHECK_EQ(ioctl_ret, EOK);
#else
      auto ioctl_ret = ioctl(fd_, DPUIOC_RUN_SOFTMAX, &reg);
      CHECK_EQ(ioctl_ret, 0);
#endif
      LOG_IF(INFO, ENV_PARAM(DEBUG_SFM_CONTROLLER))
          << "group " << group << " "                                //
          << "cls " << cls << " "                                    //
          << "exnCls " << exCls << " "                               //
          << "this_batch " << this_batch << " "                      //
          << "last_batch " << last_batch << " "                      //
          << "batch_per_run " << batch_per_run << " "                //
          << "n_of_run " << n_of_run << " "                          //
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
        for (auto i = 0u; i < batch_per_run; ++i) {
          memcpy((char*)(&r_output[cls * i]),
                 (char*)(&workspace_output[exCls * i * sizeof(float)]),
                 cls * sizeof(float));
        }
      }
      if (ENV_PARAM(DEBUG_SFM_CONTROLLER) >= 2) {
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

void SfmControllerDnndk::run_xrt_cu(size_t core_idx, const uint64_t input,
                                    const unsigned int cls,
                                    const unsigned int group, const int scale,
                                    uint64_t output, uint32_t offset) {
  // TODO:  implementation
  LOG(FATAL) << "SfmControllerDnndk::run_xrt_cu() not implemented";
}

}  // namespace xir
