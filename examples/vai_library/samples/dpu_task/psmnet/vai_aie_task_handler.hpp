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
/*
 * Filename: facedetect.hpp
 *
 * Description:
 * This network is used to getting position and score of faces in the input
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once

#include <glog/logging.h>
#include <xrt/experimental/xrt_aie.h>
#include <xrt/experimental/xrt_bo.h>
#include <xrt/experimental/xrt_device.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/xrt.h>
#include <thread>

#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEBUG_AIE_DEVICE, "0");
DEF_ENV_PARAM(DEBUG_AIE_DEVICE_SLEEP, "500");
DEF_ENV_PARAM(DEBUG_AIE_DEVICE_RETRIES, "10");

DEF_ENV_PARAM_2(DPU_XCLBIN,
                "/run/media/mmcblk0p1/dpu.xclbin",
                std::string);
//const std::string dpuxclbin = "/run/media/mmcblk0p1/dpu.xclbin";

/**
 *  * xrtDeviceOpenFromXcl() - Open a device from a shim xclDeviceHandle
 *  *
 *  * @xhdl:         Shim xclDeviceHandle
 *  * Return:        Handle representing the opened device, or nullptr on error
 *  *
 *  * The returned XRT device handle must be explicitly closed when
 *  * nolonger needed.
 *  */

class vai_aie_task_handler {
 public:
  vai_aie_task_handler(const char* xclbin_path = ENV_PARAM(DPU_XCLBIN).c_str(), int dev_id = 0) {
#if !defined(__AIESIM__)
    dhdl = nullptr;
    int value = 0;
    // Create XRT device handle for XRT API
    xclbinFilename = xclbin_path;
    auto counter = 0;
    do {
      counter++;
      dhdl = xrtDeviceOpen(dev_id);  // device index=0
      CHECK(dhdl != nullptr) << "cannot open device";
      value = xrtDeviceLoadXclbinFile(dhdl, xclbinFilename);
      LOG_IF(INFO, ENV_PARAM(DEBUG_AIE_DEVICE))
          << "xrtDeviceLoadXclbinFile error: "
          << "value " << value << " "                    //
          << "xclbinFilename " << xclbinFilename << " "  //
          << "dhdl " << dhdl << " "                      //
          << "counter = " << counter << " "
          ;
      if (value != 0) {
        value = xrtDeviceClose(dhdl);
        LOG_IF(INFO, ENV_PARAM(DEBUG_AIE_DEVICE))
            << "close board failure when retry. value=" << value;
        std::this_thread::sleep_for(
            std::chrono::milliseconds(ENV_PARAM(DEBUG_AIE_DEVICE_SLEEP)));
        continue;
      }
      auto value1 = xrtDeviceGetXclbinUUID(dhdl, uuid);
      CHECK_EQ(value1, 0) << "xrtDeviceGetXclbinUUID error: "
                         << "value " << value1 << " "                    //
                         << "xclbinFilename " << xclbinFilename << " "  //
                         << "dhdl " << dhdl << " "                      //
			 << "counter = " << counter << " "
          ;
      LOG_IF(INFO, ENV_PARAM(DEBUG_AIE_DEVICE))
          << "xrtDeviceOpen/xrtDeviceLoadXclbinFile/xrtDeviceGetXclbinUUID OK";
      break;
    } while (counter < ENV_PARAM(DEBUG_AIE_DEVICE_RETRIES));
    CHECK_EQ(value, 0) << "cannot init board";
#endif
  };
  ~vai_aie_task_handler() {
#if !defined(__AIESIM__)
    auto value = xrtDeviceClose(dhdl);
    CHECK_EQ(value, 0) << "xrtDeviceClose error: "
                       << "value " << value << " "                    //
                       << "xclbinFilename " << xclbinFilename << " "  //
                       << "dhdl " << dhdl << " "                      //
        ;
    //LOG(INFO) << "xrtDeviceClose OK";
#endif
  };

#if !defined(__AIESIM__)
  xrtDeviceHandle dhdl;
  xuid_t uuid;
#endif
 private:
#if !defined(__AIESIM__)
  const char* xclbinFilename;
#endif
};
