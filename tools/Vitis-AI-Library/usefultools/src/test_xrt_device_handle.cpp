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
#include <glog/logging.h>
#include <iostream>
using namespace std;
#include <xir/xrt_device_handle.hpp>

static void show(xir::XrtDeviceHandle* h, const std::string& cu_name) {
  LOG(INFO) << " cu size= " << h->get_num_of_cus(cu_name);
  for (auto i = 0u; i < h->get_num_of_cus(cu_name); ++i) {
    LOG(INFO) << "cu[" << i << "]: "                                          //
              << "device_id " << h->get_device_id(cu_name, i) << " "          //
              << "full_name " << h->get_cu_full_name(cu_name, i) << " "       //
              << "kernel_name " << h->get_cu_kernel_name(cu_name, i) << " "   //
              << "instance_name " << h->get_instance_name(cu_name, i) << " "  //
              << "fingerprint " << std::hex << "0x"                           //
              << h->get_fingerprint(cu_name, i) << std::dec << " "            //
              << "cu_handle " << h->get_handle(cu_name, i) << " "             //
              << "cu_mask " << h->get_cu_mask(cu_name, i) << " "              //
              << "cu_addr " << std::hex << "0x" << h->get_cu_addr(cu_name, i)
              << " "  //
        ;
  }
}
int main(int argc, char* argv[]) {
  // auto cu_name = std::string{argv[1]};
  auto cu_name = std::string("");
  {
    auto h1 = xir::XrtDeviceHandle::get_instance();
    show(h1.get(), cu_name);
  }

  return 0;
}
