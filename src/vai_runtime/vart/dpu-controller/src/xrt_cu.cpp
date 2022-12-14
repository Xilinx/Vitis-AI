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
#include "xrt_cu.hpp"

#include <UniLog/UniLog.hpp>
#include <ert.h>
#include <glog/logging.h>

#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/xxd.hpp>

extern "C" XCL_DRIVER_DLLESPEC int xclRegRead(xclDeviceHandle handle,
                                              uint32_t ipIndex, uint32_t offset,
                                              uint32_t* datap);
#include "../../xrt-device-handle/src/xrt_xcl_read.hpp"
DEF_ENV_PARAM(DEBUG_XRT_CU, "0");
DEF_ENV_PARAM(XLNX_DPU_TIMEOUT, "10000");
DEF_ENV_PARAM(XLNX_XRT_CU_DRY_RUN, "0");

#define DOMAIN xclBOKind(0)
namespace xir {
XrtCu::XrtCu(const std::string& cu_name)
    : cu_name_{cu_name}, handle_{xir::XrtDeviceHandle::get_instance()} {
  auto num_of_cus = handle_->get_num_of_cus(cu_name_);
  bo_handles_.reserve(num_of_cus);
  for (auto idx = 0u; idx < num_of_cus; ++idx) {
    auto xcl_handle =
        xclOpen(handle_->get_device_id(cu_name_, idx), NULL, XCL_INFO);
    auto bo_handle = xclAllocBO(xcl_handle, 4096, DOMAIN, XCL_BO_FLAGS_EXECBUF);
    auto bo_addr = xclMapBO(xcl_handle, bo_handle, true);
    auto cu_index = handle_->get_cu_index(cu_name_, idx);
    auto ip_index = handle_->get_ip_index(cu_name_, idx);
    auto cu_mask = handle_->get_cu_mask(cu_name_, idx);
    auto cu_addr = handle_->get_cu_addr(cu_name_, idx);
    auto cu_device_id = handle_->get_device_id(cu_name_, idx);
    auto cu_core_id = handle_->get_core_id(cu_name_, idx);
    auto cu_fingerprint = handle_->get_fingerprint(cu_name_, idx);
    auto cu_name = handle_->get_instance_name(cu_name_, idx);
    auto cu_kernel_name = handle_->get_cu_kernel_name(cu_name_, idx);
    auto cu_full_name = handle_->get_cu_full_name(cu_name_, idx);
    auto cu_uuid = handle_->get_uuid(cu_name_, idx);
    UNI_LOG_CHECK(bo_addr != nullptr, VART_XRT_NULL_PTR);
    auto r = xclOpenContext(xcl_handle, &cu_uuid[0], cu_index, true);
    PCHECK(r == 0) << "cannot open context! "
                   << "cu_index " << cu_index << " "      //
                   << "cu_addr " << cu_addr << " "        //
                   << "fingerprint " << std::hex << "0x"  //
                   << cu_fingerprint << std::dec << " "   //
        ;
    // dpu read-only register range [0x10,0x200)
#ifdef HAS_xclIPSetReadRange
    r = xclIPSetReadRange(xcl_handle, cu_index, 0x10, 0x1F0);
    PCHECK(r == 0) << "cannot set read range! "
                   << "cu_index " << cu_index << " "      //
                   << "cu_addr " << cu_addr << " "        //
                   << "fingerprint " << std::hex << "0x"  //
                   << cu_fingerprint << std::dec << " "   //
        ;
#endif
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_CU))
        << "idx " << idx << " "              //
        << "handle " << xcl_handle << " "    //
        << "bo_handle " << bo_handle << " "  //
        << "bo_addr " << bo_addr << " "      //
        << "cu_index " << cu_index << " "    //
        << "ip_index " << ip_index << " "    //
        << "cu_mask " << cu_mask << " "      //
        << "cu_addr " << std::hex << "0x" << cu_addr << std::dec
        << "cu_device "                     //
        << cu_device_id << " "              //
        << "cu_core " << cu_core_id << " "  //
        << "cu_fingerprint 0x" << std::hex << cu_fingerprint << std::dec
        << " "                                         //
        << "cu_full_name " << cu_full_name << " "      //
        << "cu_kernel_name " << cu_kernel_name << " "  //
        << "cu_name " << cu_name << " "                //
        ;

    bo_handles_.emplace_back(
        my_bo_handle{xcl_handle, bo_handle, bo_addr, cu_index, ip_index,
                     cu_mask, cu_addr, cu_device_id, cu_core_id, cu_fingerprint,
                     cu_name, cu_kernel_name, cu_full_name, cu_uuid});
    init_cmd(idx);
  }
}

struct timestamps {
  uint64_t total;
  uint64_t to_driver;
  uint64_t to_cu;
  uint64_t cu_complete;
  uint64_t done;
};
static inline uint64_t tp2ns(struct timespec* tp) {
  return (uint64_t)tp->tv_sec * 1000000000UL + tp->tv_nsec;
}
static void print_one_timestamp(const timestamps& ts) {
  LOG(INFO) << "Total: " << ts.total / 1000 << "us\t"
            << "ToDriver: " << ts.to_driver / 1000 << "us\t"
            << "ToCU: " << ts.to_cu / 1000 << "us\t"
            << "Complete: " << ts.cu_complete / 1000 << "us\t"
            << "Done: " << ts.done / 1000 << "us" << std::endl;
}
static void print_timestamp(const uint64_t start, const uint64_t end,
                            cu_cmd_state_timestamps* c) {
  struct timestamps ts;
  ts.total = end - start;
  ts.to_driver = c->skc_timestamps[ERT_CMD_STATE_NEW] - start;
  ts.to_cu = c->skc_timestamps[ERT_CMD_STATE_RUNNING] -
             c->skc_timestamps[ERT_CMD_STATE_NEW];
  ts.cu_complete = c->skc_timestamps[ERT_CMD_STATE_COMPLETED] -
                   c->skc_timestamps[ERT_CMD_STATE_RUNNING];
  ts.done = end - c->skc_timestamps[ERT_CMD_STATE_COMPLETED];
  print_one_timestamp(ts);
}

void XrtCu::run(size_t device_core_idx, XrtCu::prepare_ecmd_t prepare,
                callback_t on_success, callback_t on_failure) {
  UNI_LOG_CHECK(bo_handles_.size() > 0u, VART_XRT_DEVICE_BUSY)
    << "no cu availabe. cu_name=" << cu_name_;
  struct timespec tp;
#ifdef _WIN32
  uint64_t start = 0;  // TODO; implemented it on windows.
#else
  clock_gettime(CLOCK_MONOTONIC, &tp);
  uint64_t start = tp2ns(&tp);
#endif

  device_core_idx = device_core_idx % bo_handles_.size();
  auto ecmd = bo_handles_[device_core_idx].get();
  auto cu_mask = bo_handles_[device_core_idx].cu_mask;
  auto cu_addr = bo_handles_[device_core_idx].cu_addr;
  auto handle = bo_handles_[device_core_idx].handle;
  auto bo_handle = bo_handles_[device_core_idx].bo_handle;
  ecmd->cu_mask = cu_mask;
  ecmd->stat_enabled = 1;
  prepare(ecmd);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_CU))
      << "sizeof(ecmd) " << sizeof(*ecmd) << " "                 //
      << "ecmd->state " << ecmd->state << " "                    //
      << "ecmd->cu_mask " << ecmd->cu_mask << " "                //
      << "ecmd->extra_cu_masks " << ecmd->extra_cu_masks << " "  //
      << "ecmd->count " << ecmd->count << " "                    //
      << "ecmd->opcode " << ecmd->opcode << " "                  //
      << "ecmd->type " << ecmd->type << " "                      //
      << ((ENV_PARAM(DEBUG_XRT_CU) >= 2)
              ? vitis::ai::xxd((unsigned char*)ecmd,
                               (sizeof *ecmd) + ecmd->count * 4, 8, 1)
              : std::string(""));
  ;
  __TIC__(XRT_RUN)
  bool is_done = false;
  auto start_from_0 = std::chrono::steady_clock::now();
  auto start_from = start_from_0;
  auto state = 0;
  if (ENV_PARAM(XLNX_XRT_CU_DRY_RUN)) {
    is_done = true;
    state = 4;
  } else {
    auto exec_buf_result = xclExecBuf(handle, bo_handle);
    UNI_LOG_CHECK(exec_buf_result == 0, VART_XRT_FUNC_FAULT)
      << " cannot execute buffer";
    start_from = std::chrono::steady_clock::now();
    while (!is_done) {
      auto wait_value = xclExecWait(handle, 1000);
      UNI_LOG_CHECK(wait_value >= 0, VART_XRT_FUNC_FAULT)
      << " cannot xclExecWait";
      state = ecmd->state;
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_CU) >= 2)
          << "wait_value " << wait_value << " "  //
          << "state " << state << " "            //
          ;
      if (state >= ERT_CMD_STATE_COMPLETED) {
        is_done = true;
      }
      if (!is_done) {
        auto now = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now - start_from)
                      .count();
        if (ms > ENV_PARAM(XLNX_DPU_TIMEOUT)) {
          break;
        }
      }
    }
  };
  if (!is_done) {
#ifdef _WIN32
    uint64_t end = 0;  // todo , implemented it on windows
#else
    clock_gettime(CLOCK_MONOTONIC, &tp);
    uint64_t end = tp2ns(&tp);
#endif
    auto now = std::chrono::steady_clock::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - start_from)
            .count();
    LOG_IF(WARNING, !is_done)
        << "cu timeout! "                                //
        << "device_core_idx " << device_core_idx << " "  //
        << " handle=" << handle                          //
        << " ENV_PARAM(XLNX_DPU_TIMEOUT) " << ENV_PARAM(XLNX_DPU_TIMEOUT)
        << " "                                                           //
        << "state " << state << " "                                      //
        << "ERT_CMD_STATE_COMPLETED " << ERT_CMD_STATE_COMPLETED << " "  //
        << "ms " << ms << " "                                            //
        << " bo=" << bo_handle                                           //
        << " is_done " << is_done << " "                                 //
        ;
#ifndef _WIN32
    print_timestamp(start, end, ert_start_kernel_timestamps(ecmd));
#endif

  } else if (ENV_PARAM(DEBUG_XRT_CU)) {
#ifdef _WIN32
    uint64_t end = 0;  // todo: implemented on windows.
#else
    clock_gettime(CLOCK_MONOTONIC, &tp);
    uint64_t end = tp2ns(&tp);
#endif
    auto now = std::chrono::steady_clock::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::microseconds>(now - start_from)
            .count();
    auto ms0 = std::chrono::duration_cast<std::chrono::microseconds>(
                   now - start_from_0)
                   .count();
#ifndef _WIN32
    auto c = ert_start_kernel_timestamps(ecmd);
    LOG(INFO) << "device_core_idx =" << device_core_idx << " "  //
              << "handle =" << handle << " "                    //
              << "time = " << ms << " "                         //
              << "time0 = " << ms0 << " "                       //
              << "ts0 = " << c->skc_timestamps[0] << " "        //
              << "ts1 = " << c->skc_timestamps[1] << " "        //
        ;
    print_timestamp(start, end, c);
#endif
  }
  __TOC__(XRT_RUN)
  if (is_done) {
    on_success(handle, cu_addr);
  } else {
    on_failure(handle, cu_addr);
  }
}

XrtCu::~XrtCu() {
  int idx = 0;
  for (const auto& x : bo_handles_) {
    xclUnmapBO(x.handle, x.bo_handle, x.bo_addr);
    xclFreeBO(x.handle, x.bo_handle);
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_CU))
        << "idx " << idx << " "  //
        << "handle " << x.handle << " "
        << "bo_handle " << x.bo_handle << " "  //
        << "bo_addr " << x.bo_addr << " "      //
        << "cu_mask " << x.cu_mask << " "      //
        << "cu_addr " << std::hex << "0x" << x.cu_addr << std::dec
        << "device_id " << x.device_id << " "                                //
        << "core_id " << x.core_id << " "                                    //
        << "fingerprint 0x" << std::hex << x.fingerprint << std::dec << " "  //
        ;
    auto uuid = x.uuid;
    auto r = xclCloseContext(x.handle, &uuid[0], x.cu_index);
    PCHECK(r == 0) << "cannot close context! "
                   << " cu_mask " << x.cu_mask    //
                   << " cu_index " << x.cu_index  //
                   << " cu_addr " << std::hex << "0x" << x.cu_addr
                   << std::dec  //
        ;
    xclClose(x.handle);
    idx = idx + 1;
  }
}
size_t XrtCu::get_num_of_cu() const { return bo_handles_.size(); }

std::string XrtCu::get_full_name(size_t device_core_idx) const {
  return bo_handles_[device_core_idx].full_name;
}

std::string XrtCu::get_kernel_name(size_t device_core_idx) const {
  return bo_handles_[device_core_idx].kernel_name;
}

std::string XrtCu::get_instance_name(size_t device_core_idx) const {
  return bo_handles_[device_core_idx].name;
}

size_t XrtCu::get_device_id(size_t device_core_idx) const {
  return bo_handles_[device_core_idx].device_id;
}

size_t XrtCu::get_core_id(size_t device_core_idx) const {
  return bo_handles_[device_core_idx].core_id;
}

uint64_t XrtCu::get_fingerprint(size_t device_core_idx) const {
  return bo_handles_[device_core_idx].fingerprint;
}

uint32_t XrtCu::read_register(size_t device_core_idx, uint32_t offset) const {
  auto xcl_handle = bo_handles_[device_core_idx].handle;
  auto cu_index = bo_handles_[device_core_idx].cu_index;
  auto cu_addr = bo_handles_[device_core_idx].cu_addr;
  uint32_t value = 0;
  auto read_result = xrtXclRead(xcl_handle, cu_index, offset, cu_addr, &value);
  UNI_LOG_CHECK(read_result == 0, VART_XRT_READ_ERROR)
                           << "xclRegRead has error!"                       //
                           << "read_result " << read_result << " "          //
                           << "device_core_idx " << device_core_idx << " "  //
                           << "cu_addr " << std::hex << "0x" << cu_index
                           << " "  //
      ;
  return value;
}
ert_start_kernel_cmd* XrtCu::get_cmd(size_t device_core_id) {
  auto ecmd = bo_handles_[device_core_id].get();
  return ecmd;
}

void XrtCu::init_cmd(size_t device_core_id) {
  auto ecmd = get_cmd(device_core_id);
  for (size_t i = 4; i < 128; ++i) {
    ecmd->data[i] = read_register(device_core_id, i * sizeof(uint32_t));
  }
}

}  // namespace xir
