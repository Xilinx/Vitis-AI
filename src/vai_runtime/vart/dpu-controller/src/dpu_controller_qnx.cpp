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
#include "./dpu_controller_qnx.hpp"
#include "vitis/ai/xxd.hpp"
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>

DEF_ENV_PARAM(DEBUG_DPU_CONTROLLER, "0")

namespace {

unsigned long DpuControllerQnx::allocate_task_id() { //
  // it seems that task id is no use.
  return 1024;
}

DpuControllerQnx::DpuControllerQnx()
    : xir::DpuController{},                     //
      dpu_{vitis::ai::QnxDpu::get_instance()}, //
      task_id_{0} {
  task_id_ = allocate_task_id(); // task id is not so useful.
}

DpuControllerQnx::~DpuControllerQnx() { //
}

void DpuControllerQnx::run(const uint64_t code,
                           const std::vector<uint64_t> &gen_reg,
                           int device_id /*not used*/) {
  struct ioc_kernel_run2_t ioc_kernel_run2;
  auto parameter = gen_reg[0];
  auto workspace = gen_reg[1];
  ioc_kernel_run2.handle_id = 0;
  ioc_kernel_run2.time_start = 0;
  ioc_kernel_run2.time_end = 0;
  ioc_kernel_run2.core_id = 0;
  // TODO: it support 64 bits register?
  ioc_kernel_run2.addr_code = code;
  ioc_kernel_run2.addr0 = parameter;
  ioc_kernel_run2.addr1 = workspace;
  ioc_kernel_run2.addr2 = 0x0; // ((uint32_t)code) & 0xFFFFFFFF;
  ioc_kernel_run2.addr3 = 0x0;
  ioc_kernel_run2.addr4 = 0x0;
  ioc_kernel_run2.addr5 = 0x0;
  ioc_kernel_run2.addr6 = 0x0;
  ioc_kernel_run2.addr7 = 0x0;

  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
      << "code " << std::hex << "0x" << code << " "           //
      << "parameter " << std::hex << "0x" << parameter << " " //
      << "workspace " << std::hex << "0x" << workspace << " " //
      << std::dec << "dpu_->get_handle() " << dpu_->get_handle() << " "
      << vitis::ai::xxd((const unsigned char *)&ioc_kernel_run2,
                         sizeof(ioc_kernel_run2), 16, 4);
  CHECK_EQ(0, qnx_ioctl(dpu_->get_handle(), DPU_IOCTL_RUN2, &ioc_kernel_run2,
                        sizeof(ioc_kernel_run2)))
      << "dpu timeout?";
  return;
}

static struct Registar {
  Registar() {
    auto fd = open("/dev/xdpu/0", O_RDWR);
    auto disabled = fd < 0;
    if (!disabled) {
      xir::DpuController::registar("01_qnx_dpu", []() {
        return std::shared_ptr<xir::DpuController>(
            vitis ::ai::WeakSingleton<DpuControllerQnx>::create());
      });
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
          << "register the qnx dpu controller";
      close(fd);
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
          << "cancel register the qnx dpu controller, because "
             "/dev/xdpu/0 is not opened";
    }
  }
} g_registar;

} // namespace
