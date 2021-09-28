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

#ifndef _XF_FINTECH_DEVICE_MANAGER_H_
#define _XF_FINTECH_DEVICE_MANAGER_H_

#include <cstring>
#include <vector>

#include "xcl2.hpp"

#include "xf_fintech_device.hpp"

namespace xf {
namespace fintech {

/**
 * @class DeviceManager
 * @brief Used to enumerate available Xilinx devices.
 */
class DeviceManager {
   public:
    static DeviceManager* getInstance(void);

    /**
     * Returns a vector of ALL available Xilinx devices
     */
    static std::vector<Device*> getDeviceList(void);

    /**
     * Allows a user to get a list of devices whose name contains a specified
     * substring.
     * Passing "u250" would find any devices whose name contained "u250" e.g.
     * "xilinx_u250_xdma_201830_2"
     * This can be useful in a system with different types of cards installed.
     *
     * @param[in] deviceSubString Substring to look for in the device name
     *
     */
    static std::vector<Device*> getDeviceList(std::string deviceSubString);

    /**
     * Allows a user to list of devices whose type matches the specifed
     * enumeration
     * This can be useful in a system with different types of cards installed.
     *
     * @param[in] deviceType Device type to look for.
     *
     */
    static std::vector<Device*> getDeviceList(Device::DeviceType deviceType);

   private:
    DeviceManager();
    virtual ~DeviceManager();

    void createDeviceList(void);
    void deleteDeviceList(void);

    static DeviceManager* m_instance; // SINGLETON

    std::vector<Device*> m_deviceList;
};

} // end namespace fintech
} // end namespace xf

#endif // end __XF_FINTECH_DEVICE_MANAGER_H_
