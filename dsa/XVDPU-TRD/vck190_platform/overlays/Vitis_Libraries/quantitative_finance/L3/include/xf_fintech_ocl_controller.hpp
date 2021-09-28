/*
 * Copyright 2019 Xilinx, Inc.
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

#ifndef _XF_FINTECH_OCL_CONTROLLER_H_
#define _XF_FINTECH_OCL_CONTROLLER_H_

#include <mutex>

#include "xf_fintech_device.hpp"

namespace xf {
namespace fintech {

/**
 * @class OCLController
 *
 * A class handling interaction with OpenCL.
 */

class OCLController {
   public:
    OCLController();
    virtual ~OCLController() = 0;

    /**
     * Claim the device
     */
    int claimDevice(Device* device);

    /**
     * Release the device
     */
    int releaseDevice(void);

    /**
     * Check the device is ready
     */
    bool deviceIsPrepared(void);

   private:
    virtual int createOCLObjects(Device* device) = 0;
    virtual int releaseOCLObjects(void) = 0;

   protected:
    void setCLError(cl_int clError);
    cl_int getCLError(void);

    std::mutex m_mutex;

    Device* m_pDevice;
    bool m_bDeviceIsPrepared;

    cl_int m_clError;
};

} // end namespace fintech
} // end namespace xf

#endif //_XF_FINTECH_OCL_CONTROLLER_H_
