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
#include "./dpu_controller_dnndk.hpp"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstddef>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>
#include <vitis/ai/xxd.hpp>
DEF_ENV_PARAM(DEBUG_DPU_CONTROLLER, "0")
DEF_ENV_PARAM(XLNX_ENABLE_FINGERPRINT_CHECK, "1")
#define DPU_IOCTL_MAGIC 'D'
#define REQ_DPU_RUN _IOWR(DPU_IOCTL_MAGIC, 4, struct req_session_submit_t*)
#define REQ_DPU_CREATE_TASK _IOWR(DPU_IOCTL_MAGIC, 9, struct req_task_manu_t*)
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

namespace {

static void read(void* data, int fd, uint64_t offset_addr, size_t size) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
      << "data " << data << " "           //
      << "offset " << offset_addr << " "  //
      << "size " << size << " "           //
      ;
  long page_size = sysconf(_SC_PAGE_SIZE);
  unsigned long offset = offset_addr % page_size;
  unsigned long base =
      (offset_addr / page_size) * page_size;  // page size alignment;
  unsigned long extra_size = size + offset;
  unsigned long map_size =
      (extra_size / page_size) * page_size +
      (extra_size % page_size == 0 ? 0 : page_size);  // page size alignment;
  auto offset_data =
      mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
  CHECK_NE(offset_data, MAP_FAILED);
  memcpy(data, reinterpret_cast<const char*>(offset_data) + offset, size);
  munmap(offset_data, map_size);
}

static uint64_t get_dpu_fingerprint() {
  if (ENV_PARAM(XLNX_ENABLE_FINGERPRINT_CHECK)) {
#ifdef __QNX__
    return 0u;
#else
    auto phy_l = 0x8f0001f0;
    auto phy_h = 0x8f0001f4;
    auto mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    CHECK_GT(mem_fd, 0) << "cannot open /dev/mem";
    uint64_t ret = 0u;
    uint64_t value_l = 0;
    uint64_t value_h = 0;
    read(&value_l, mem_fd, phy_l, 4);
    read(&value_h, mem_fd, phy_h, 4);
    ret = value_h;
    ret = (ret << 32) + value_l;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
        << " fingerprint: 0x" << std::hex << ret << std::dec << std::hex
        << " 0x" << value_h << " 0x" << value_l << std::dec;
    close(mem_fd);
    return ret;
#endif
  }
  return 0u;
}

DpuControllerDnndk::DpuControllerDnndk()
    : xir::DpuController{},
      fd_{open("/dev/dpu", O_RDWR)},
      fingerprint_{get_dpu_fingerprint()} {
  CHECK_GT(fd_, 0) << "cannot open /dev/dpu";
}

DpuControllerDnndk::~DpuControllerDnndk() {  //
}

static std::string dump_gen_reg(const std::vector<uint64_t>& gen_reg) {
  std::ostringstream str;
  str << std::hex;
  for (const auto& v : gen_reg) {
    str << " 0x" << v;
  }
  return str.str();
}

void DpuControllerDnndk::run(size_t core_idx, const uint64_t code,
                             const std::vector<uint64_t>& gen_reg) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
      << std::hex                                          //
      << "code 0x" << code << " "                          //
      << "core_idx " << core_idx << " "                    //
      << "gen_reg: " << dump_gen_reg(gen_reg) << std::dec  //
      ;
  CHECK_EQ(code % 4096u, 0u) << "code  = " << code;
  struct req_kernel_run_t t2;
  // CHECK_EQ(gen_reg.size(), 2u);
  auto parameter = gen_reg[0];
  auto workspace = gen_reg[1];
  memset(&t2, 0, sizeof(t2));
  t2.handle_id = 0u;
  t2.addr_code = (uint32_t)code;
  t2.addr0 = (uint32_t)parameter;
  t2.addr1 = (uint32_t)workspace;
  t2.addr2 = (uint32_t)code;
  t2.addr3 = 0;
  t2.addr4 = 0;
  t2.addr5 = 0;
  t2.addr6 = 0;
  t2.addr7 = 0;
  auto retval = ioctl(fd_, REQ_DPU_RUN, (void*)(&t2));

  CHECK_EQ(retval, 0) << "run dpu failed.";
  return;
}

size_t DpuControllerDnndk::get_num_of_dpus() const {
  // because scheduler is done by dpu.ko, no need to get
  // num_of_dpuds.
  return 1u;
}
size_t DpuControllerDnndk::get_device_id(size_t device_core_id) const {
  // only single device is supported.
  return 0u;
}
size_t DpuControllerDnndk::get_core_id(size_t device_core_id) const {
  // only single device is supported.
  return device_core_id;
}

uint64_t DpuControllerDnndk::get_fingerprint(size_t device_core_id) const {
  // TODO: return a magic number that match all target, i.e. disable
  // fingerprint checking.
  return fingerprint_;
}

static struct Registar {
  Registar() {
    auto fd = open("/dev/dpu", O_RDWR);
    auto disabled = fd < 0;
    if (!disabled) {
      xir::DpuController::registar("00_dnndk", []() {
        return std::shared_ptr<xir::DpuController>(
            vitis::ai::WeakSingleton<DpuControllerDnndk>::create());
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

}  // namespace
