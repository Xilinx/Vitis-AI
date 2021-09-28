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

#ifndef _XF_FINTECH_DEVICE_H_
#define _XF_FINTECH_DEVICE_H_

#include <mutex>
#include <string>

#include "xcl2.hpp"

namespace xf {
namespace fintech {

class OCLController;

/**
 * @class Device
 *
 * A class representing an individual accelerator card.
 */
class Device {
   public:
    Device();
    Device(cl::Device clDevice);
    virtual ~Device();

    /**
     * Recognised Xilinx device types.
     */
    typedef enum {
        U50,
        U200,
        U250,
        U280,

        UNKNOWN

    } DeviceType;

    /**
     * Retrieves the enclosed OpenCL device object.
     * This allows the user to then invoke any standard OpenCL functions that
     * require a cl::Device object
     */
    cl::Device getCLDevice(void);

    /**
     * Retrieves the name of this device object.
     */
    std::string getName(void);

    /**
     * Retrieves the device type of this object (or DeviceType::UNKNOWN if is not a
     * supported device)
     */
    DeviceType getDeviceType(void);

    /**
     * Converts a string representation of the device type
     */
    std::string getDeviceTypeString(void);

    /**
     * Claim the device
     */
    int claim(OCLController* owner);

    /**
     * Release the device
     */
    int release(OCLController* owner);

   private:
    void setupDeviceType(void);

    cl::Device m_clDevice;
    std::mutex m_mutex;

    std::string m_deviceName;
    DeviceType m_deviceType;

    OCLController* m_pOwner;
};

} // end namespace fintech
} // end namespace xf

#endif
