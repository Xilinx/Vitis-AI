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
#include "./dpu_control_dnndk.hpp"
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>

DEF_ENV_PARAM(DEBUG_DPU_CONTROLLER, "0")

#define DPU_IOCTL_MAGIC 'D'
#define REQ_DPU_RUN _IOWR(DPU_IOCTL_MAGIC, 4, struct req_session_submit_t *)
#define REQ_DPU_CREATE_TASK _IOWR(DPU_IOCTL_MAGIC, 9, struct req_task_manu_t *)
typedef unsigned long task_handle_t;
struct req_task_manu_t {
  task_handle_t task_id; /* the handle of DPU task (RETURNED) */
};

struct req_kernel_run_t {
  task_handle_t handle_id; /* the handle of DPU task */
  uint32_t addr_code;      /* the address for DPU code */
  uint32_t addr0;          /* address reg0 */
  uint32_t addr1;          /* address reg1 */
  uint32_t addr2;          /* address reg2 */
  uint32_t addr3;          /* address reg3 */
  uint32_t addr4;          /* address reg4 */
  uint32_t addr5;          /* address reg5 */
  uint32_t addr6;          /* address reg6 */
  uint32_t addr7;          /* address reg7 */
  long long time_start;    /* the start timestamp before running (RETURNED) */
  long long time_end;      /* the end timestamp after running (RETURNED) */
  int core_id;             /* the core id of the task*/
};

extern "C" long dpu_run_request(unsigned int cmd_id, void *arg);

namespace {
task_handle_t DpuControllerDnndk::allocate_task_id() { //
  struct req_task_manu_t req_task_manu;
  memset(&req_task_manu, 0, sizeof(req_task_manu));
  auto retval = ioctl(fd_, REQ_DPU_CREATE_TASK, (void *)(&req_task_manu));
  CHECK_EQ(retval, 0) << "dpu allocate_task_id failed";
  LOG_IF(INFO, false) << "create dpu task ok."
                      << "req_task_manu.task_id; " << req_task_manu.task_id
                      << " ";
  return req_task_manu.task_id;
}

DpuControllerDnndk::DpuControllerDnndk()
    : xir::DpuController{}, fd_{open("/dev/dpu", O_RDWR)}, task_id_{0} {
  CHECK_GT(fd_, 0) << "cannot open /dev/dpu";
  task_id_ = allocate_task_id(); // task id is not so useful.
}

DpuControllerDnndk::~DpuControllerDnndk() { //
  // free_task_id(task_id_);
  //
  // it is wired that the task_id does not need to be freed.
}

void DpuControllerDnndk::run(const uint64_t code,
                             const std::vector<uint64_t> &gen_reg,
                             int device_id /* not used */) {
  struct req_kernel_run_t t2;
  CHECK_EQ(gen_reg.size(), 2u);
  auto parameter = gen_reg[0];
  auto workspace = gen_reg[1];
  memset(&t2, 0, sizeof(t2));
  t2.handle_id = task_id_;
  t2.addr_code = (uint32_t)code;
  t2.addr0 = (uint32_t)parameter;
  t2.addr1 = (uint32_t)workspace;
  t2.addr2 = (uint32_t)code;
  t2.addr3 = 0;
  t2.addr4 = 0;
  t2.addr5 = 0;
  t2.addr6 = 0;
  t2.addr7 = 0;
  auto retval = ioctl(fd_, REQ_DPU_RUN, (void *)(&t2));
  CHECK_EQ(retval, 0) << "run dpu failed.";
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
      << std::hex                           //
      << "workspace 0x" << workspace << " " //
      << "code 0x" << code << " "           //
      << "parameter 0x" << parameter << " " //
      << std::dec                           //
      ;
  return;
}

static struct Registar {
  Registar() {
    auto fd = open("/dev/dpu", O_RDWR);
    auto disabled = fd < 0;
    if (!disabled) {
      xir::DpuController::registar("01_dnndk", []() {
        return std::shared_ptr<xir::DpuController>(
            vitis ::ai::WeakSingleton<DpuControllerDnndk>::create());
      });
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
          << "register the dnndk dpu controller";
      close(fd);
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
          << "cancel register the dnndk dpu controller, because "
             "/dev/dpu is not opened";
    }
  }
} g_registar;

} // namespace
