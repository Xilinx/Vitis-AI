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
#include <stddef.h> /* offsetof */
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstddef>
#include <vitis/ai/env_config.hpp>
#ifndef _WIN32
#include <vitis/ai/trace.hpp>
#endif
#include <vitis/ai/weak.hpp>
#include <vitis/ai/xxd.hpp>
// (TODO) change the default value to 0 after bugfix dpu.ko.
DEF_ENV_PARAM(DEBUG_DPU_KO_LOCK, "1")
DEF_ENV_PARAM(DEBUG_DPU_CONTROLLER, "0")
DEF_ENV_PARAM(XLNX_ENABLE_FINGERPRINT_CHECK, "1")
DEF_ENV_PARAM(XLNX_SHOW_DPU_COUNTER, "0");

#include "../../buffer-object/src/dpu.h"

namespace {

// static void read(void* data, int fd, uint64_t offset_addr, size_t size) {
//   LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
//       << "data " << data << " "           //
//       << "offset " << offset_addr << " "  //
//       << "size " << size << " "           //
//       ;
//   long page_size = sysconf(_SC_PAGE_SIZE);
//   unsigned long offset = offset_addr % page_size;
//   unsigned long base =
//       (offset_addr / page_size) * page_size;  // page size alignment;
//   unsigned long extra_size = size + offset;
//   unsigned long map_size =
//       (extra_size / page_size) * page_size +
//       (extra_size % page_size == 0 ? 0 : page_size);  // page size alignment;
//   auto offset_data =
//       mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
//   CHECK_NE(offset_data, MAP_FAILED);
//   memcpy(data, reinterpret_cast<const char*>(offset_data) + offset, size);
//   munmap(offset_data, map_size);
// }

static uint64_t get_dpu_fingerprint() {
  if (ENV_PARAM(XLNX_ENABLE_FINGERPRINT_CHECK)) {
#ifdef __QNX__
    return 0u;
#else
    auto fd = open("/dev/dpu", O_RDWR | O_SYNC);
    CHECK_GE(fd, 0) << "cannot open /dev/dpu";
    uint64_t ret = 0;
    auto retval = ioctl(fd, DPUIOC_G_TGTID, (void*)(&ret));
    close(fd);
    CHECK_EQ(retval, 0) << "read dpu fingerprint failed.";
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
        << " fingerprint: 0x" << std::hex << ret << std::dec << std::hex
        << " 0x" << ret << std::dec;
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

  auto cu_device_id = 0;
  auto cu_core_id = 0;
  auto cu_addr = 0x0;
  auto cu_name = "DPU";
  auto cu_fingerprint = fingerprint_;
#ifndef _WIN32
  for (size_t i = 0; i < get_num_of_dpus(); i++) { 
    std::ostringstream name;
    name << "DPU_" << cu_core_id << std::endl;
    auto cu_full_name = name.str();
    vitis::ai::trace::add_info("dpu-controller", TRACE_VAR(cu_device_id),
                               TRACE_VAR(cu_core_id), TRACE_VAR(cu_addr),
                               TRACE_VAR(cu_name), TRACE_VAR(cu_full_name),
                               TRACE_VAR(cu_fingerprint));
    cu_core_id ++;
  }
#endif
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

static std::string xdpu_get_counter(const ioc_kernel_run_t& t) {
  std::ostringstream str;
  struct {
    char name[64];
    uint32_t addr;
    int is_u64;
  } regs[] = {
      {"LSTART", offsetof(ioc_kernel_run_t, lstart_cnt), 0},
      {"LEND", offsetof(ioc_kernel_run_t, lend_cnt), 0},  //
      {"CSTART", offsetof(ioc_kernel_run_t, cstart_cnt), 0},
      {"CEND", offsetof(ioc_kernel_run_t, cend_cnt), 0},  //
      {"SSTART", offsetof(ioc_kernel_run_t, sstart_cnt), 0},
      {"SEND", offsetof(ioc_kernel_run_t, send_cnt), 0},  //
      {"MSTART", offsetof(ioc_kernel_run_t, pstart_cnt), 0},
      {"MEND", offsetof(ioc_kernel_run_t, pend_cnt), 0},  //
      {"CYCLE_L", offsetof(ioc_kernel_run_t, counter), 0},
      {"CYCLE_H", offsetof(ioc_kernel_run_t, counter) + sizeof(uint32_t),
       0},  //
      {"TIMER", offsetof(ioc_kernel_run_t, time_start), 1},
  };
  int cnt = 0;
  const char* base = (const char*)&t;
  for (const auto& reg : regs) {
    if (reg.is_u64 == 0) {
      const uint32_t* value = (const uint32_t*)(base + reg.addr);
      str << " " << reg.name << " "  //
          << *value << " "           //
          ;
    } else {
      const uint64_t* value = (const uint64_t*)(base + reg.addr);
      str << " " << reg.name << " "  //
          << *value << " "           //
          ;
    }
    cnt++;
  }
  return str.str();
}

uint64_t get_device_hwcounter(const ioc_kernel_run_t& t) {
	char* base = ( char*)&t;
	uint32_t value_l = *(uint32_t*)(base + offsetof(ioc_kernel_run_t, counter));
	uint32_t value_h = *(uint32_t*)(base + offsetof(ioc_kernel_run_t, counter) + sizeof(uint32_t));
	uint64_t value = ((uint64_t)value_h << 32) | value_l;
	return value;
}

void DpuControllerDnndk::run(size_t core_idx, const uint64_t code,
                             const std::vector<uint64_t>& gen_reg) {
  static std::vector<std::mutex> mtxs(16);
  std::unique_ptr<std::lock_guard<std::mutex>> lock;
  if (ENV_PARAM(DEBUG_DPU_KO_LOCK)) {
    lock = std::make_unique<std::lock_guard<std::mutex>>(mtxs[core_idx]);
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
      << std::hex                                          //
      << "code 0x" << code << " "                          //
      << "core_idx " << core_idx << " "                    //
      << "gen_reg: " << dump_gen_reg(gen_reg) << std::dec  //
      ;
  CHECK_EQ(code % 4096u, 0u) << "code  = " << code;
  struct ioc_kernel_run_t t2;
  auto size = gen_reg.size();
  memset(&t2, 0, sizeof(t2));
  t2.core_id = (int)core_idx;
  t2.addr_code = code;
  t2.addr0 = size >= 1 ? gen_reg[0] : 0;
  t2.addr1 = size >= 2 ? gen_reg[1] : 0;
  t2.addr2 = size >= 3 ? gen_reg[2] : 0;
  t2.addr3 = size >= 4 ? gen_reg[3] : 0;
  t2.addr4 = size >= 5 ? gen_reg[4] : 0;
  t2.addr5 = size >= 6 ? gen_reg[5] : 0;
  t2.addr6 = size >= 7 ? gen_reg[6] : 0;
  t2.addr7 = size >= 8 ? gen_reg[7] : 0;
#ifndef _WIN32
  vitis::ai::trace::add_trace("dpu-controller", vitis::ai::trace::func_start,
                              core_idx, 0);
#endif
  auto retval = ioctl(fd_, DPUIOC_RUN, (void*)(&t2));
  auto hwcounter = get_device_hwcounter(t2);
#ifndef _WIN32
  vitis::ai::trace::add_trace("dpu-controller", vitis::ai::trace::func_end,
                              core_idx, hwcounter);
#endif
  if (ENV_PARAM(XLNX_SHOW_DPU_COUNTER)) {
    auto core_idx = t2.core_id;
    std::cout << "core_idx = " << core_idx << " " << xdpu_get_counter(t2)
              << std::endl;
  }

  CHECK_EQ(retval, 0) << "run dpu failed.";
  return;
}

size_t DpuControllerDnndk::get_num_of_dpus() const {
  // because scheduler is done by dpu.ko, no need to get
  // num_of_dpuds.

  // it is important to get number of core id, otherwiese, DPU
  // workspace is shared among DPUs.

  // on x86 cloud environment, it might not open /dev/dpu, so
  // effectively, all HwSmFc is disabled.
  auto get_num_of_dpus = []() -> size_t {
    auto fd = open(
#ifdef __QNX__
        "/dev/xdpu/0",
#else
        "/dev/dpu",
#endif
        O_RDWR);
    if (fd < 0) {
      LOG_IF(WARNING, ENV_PARAM(DEBUG_DPU_CONTROLLER))
          << "cannot open /dev/dpu for smfc";
      return 0u;
    }
    uint32_t flags = 0;
    auto retval = ioctl(fd, DPUIOC_G_INFO, (void*)(&flags));
    close(fd);
    auto sfm_num = SFM_NUM(flags);
    size_t dpu_num = DPU_NUM(flags);

    CHECK_EQ(retval, 0) << "read sfm info failed.";
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
        << "sfm_num " << sfm_num << " "  //
        << "dpu_num " << dpu_num << " "  //
        ;
    return dpu_num;
  };
  static size_t num_of_dpus = get_num_of_dpus();
  return num_of_dpus;
}
size_t DpuControllerDnndk::get_device_id(size_t device_core_id) const {
  // only single device is supported.
  return 0u;
}
size_t DpuControllerDnndk::get_core_id(size_t device_core_id) const {
  // only single device is supported.
  CHECK_LT(device_core_id, get_num_of_dpus());
  return device_core_id;
}

uint64_t DpuControllerDnndk::get_fingerprint(size_t device_core_id) const {
  // TODO: return a magic number that match all target, i.e. disable
  // fingerprint checking.
  return fingerprint_;
}

static struct Registar {
  Registar() {
    if (!access("/dev/dpu", F_OK)) {
      xir::DpuController::registar("00_dnndk", []() {
        return std::shared_ptr<xir::DpuController>(
            vitis::ai::WeakSingleton<DpuControllerDnndk>::create());
      });
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
          << "register the dnndk dpu controller";
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
          << "cancel register the dnndk dpu controller, because "
             "/dev/dpu is not opened";
    }
  }
} g_registar;

}  // namespace
