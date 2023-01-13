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

#pragma once
#include <xrt/experimental/xrt-next.h>
#include <xir/xrt_device_handle.hpp>

static int g_batchnum = 0;
static std::mutex mtx_init;

template <typename T>
uint64_t data_slice(T source, int begin, int end) {
  return (source >> begin) & ((T(1) << int(end - begin)) - 1);
}

std::vector<std::pair<std::string, uint32_t>> vm_plat{
    std::pair<std::string, uint32_t>("DPUCVDX8G", 0x134),  // vck190
    std::pair<std::string, uint32_t>("DPUCV2DX8G", 0x134)  // vek280
};

void get_batch() {
  mtx_init.lock();
  if (g_batchnum == 0) {
    std::shared_ptr<xir::XrtDeviceHandle> hp =
        xir::XrtDeviceHandle::get_instance();
    xir::XrtDeviceHandle* h = hp.get();
    std::string cu_name = h->get_cu_full_name("", 0);
    uint32_t read_res;
    for (int j = 0; j < (int)vm_plat.size(); j++) {
      if (cu_name.find(vm_plat[j].first) != std::string::npos) {
        auto cu_handle = h->get_handle("", 0);
        auto cu_idx = h->get_cu_index("", 0);
        xclRegRead(cu_handle, cu_idx, vm_plat[j].second, &read_res);
        g_batchnum = data_slice(read_res, 0, 4);
        // std::cout <<"batch : " << g_batchnum <<"\n";
        break;
      }
    }
  }
  if (g_batchnum == 0) g_batchnum = 1;
  mtx_init.unlock();
}

