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
#define DEF_XCLBIN "/media/sd-mmcblk0p1/dpu.xclbin"
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
  vai_aie_task_handler(const char* xclbin_path = DEF_XCLBIN, int dev_id = 0) {
#if !defined(__AIESIM__)
    dhdl = nullptr;
    // Create XRT device handle for XRT API
    xclbinFilename = xclbin_path;
    dhdl = xrtDeviceOpen(dev_id);  // device index=0
    CHECK(dhdl != nullptr) << "cannot open device";
    xrtDeviceLoadXclbinFile(dhdl, xclbinFilename);
    xrtDeviceGetXclbinUUID(dhdl, uuid);
#endif
  };
  ~vai_aie_task_handler() {
#if !defined(__AIESIM__)
    xrtDeviceClose(dhdl);
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
