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
#include <xrt.h>

#include <iostream>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(VERBOSE, "0");
using namespace std;
#include "xir/xrt_device_handle.hpp"

static void show(xir::XrtDeviceHandle* h, const std::string& cu_name) {
  LOG(INFO) << "cu_name = " << cu_name
            << " size= " << h->get_num_of_cus(cu_name);
  for (auto i = 0u; i < h->get_num_of_cus(cu_name); ++i) {
    LOG(INFO) << "cu[" << i << "]: "                                          //
              << "device id : " << h->get_device_id(cu_name, i) << " "        //
              << "full name : " << h->get_cu_full_name(cu_name, i) << " "     //
              << "kernel name: " << h->get_cu_kernel_name(cu_name, i) << " "  //
              << "instance name " << h->get_instance_name(cu_name, i) << " "  //
              << "fingerprint " << std::hex << "0x"                           //
              << h->get_fingerprint(cu_name, i) << std::dec << " "            //
              << "cu_handle " << h->get_handle(cu_name, i) << " "             //
              << "cu_mask " << h->get_cu_mask(cu_name, i) << " "              //
              << "cu_addr " << std::hex << "0x" << h->get_cu_addr(cu_name, i)
              << " "  //
        ;
    auto handle = h->get_handle(cu_name, i);
    xclDeviceInfo2 deviceInfo;
    CHECK_EQ(xclGetDeviceInfo2(handle, &deviceInfo), 0)
        << "Unable to obtain device information";
    LOG_IF(INFO, ENV_PARAM(VERBOSE))
        << "DSA = " << deviceInfo.mName << "\n"                         //
        << "PCIe = GEN" << deviceInfo.mPCIeLinkSpeed << "x"             //
        << deviceInfo.mPCIeLinkWidth << "\n"                            //
        << "OCL Frequency = " << deviceInfo.mOCLFrequency[0] << " MHz"  //
        << "\n"                                                         //
        << "DDR Bank = " << deviceInfo.mDDRBankCount << "\n"            //
        << "Device Temp = " << deviceInfo.mOnChipTemp << " C\n"         //
        << "DeviceVersion = " << std::hex << "0x" << deviceInfo.mDeviceVersion
        << std::dec << "\n"                                                 //
        << "MIG Calibration = " << std::boolalpha << deviceInfo.mMigCalib   //
        << "\n"                                                             //
        << "mMagic = " << deviceInfo.mMagic                                 //
        << "\n"                                                             //
        << "mHALMajorVersion = " << deviceInfo.mHALMajorVersion << "\n"     //
        << "mHALMinorVersion = " << deviceInfo.mHALMinorVersion << "\n"     //
        << "mVendorId = " << deviceInfo.mVendorId << " "                    //
        << "\n"                                                             //
        << "mDeviceId = " << deviceInfo.mDeviceId << " "                    //
        << "\n"                                                             //
        << "mSubsystemId = " << deviceInfo.mSubsystemId << " "              //
        << "\n"                                                             //
        << "mSubsystemVendorId = " << deviceInfo.mSubsystemVendorId << " "  //
        << "\n"                                                             //
        << "mDeviceVersion = " << deviceInfo.mDeviceVersion << " "          //
        << "\n"                                                             //

        << "\n"  //
        ;
  }
}
int main(int argc, char* argv[]) {
  auto cu_name = std::string{argv[1]};
  {
    auto h1 = xir::XrtDeviceHandle::get_instance();
    show(h1.get(), cu_name);
    cout << "press any key to continue..." << endl;
    char c;
    cin >> c;
  }
  /* auto h2 = xir::XrtDeviceHandle::get_instance();
  show(h2.get(), cu_name);
  auto h3 = xir::XrtDeviceHandle::create();
  show(h3.get(), cu_name);
  auto h4 = xir::XrtDeviceHandle::create();
  show(h4.get(), cu_name);*/

  return 0;
}
