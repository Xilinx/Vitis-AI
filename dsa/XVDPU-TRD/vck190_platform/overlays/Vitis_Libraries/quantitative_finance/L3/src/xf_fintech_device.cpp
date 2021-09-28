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

#include "xf_fintech_device.hpp"
#include "xf_fintech_error_codes.hpp"

using namespace xf::fintech;

typedef struct _DeviceTypeStringLookupElement {
    Device::DeviceType deviceType;
    std::string deviceTypeString;

} DeviceTypeStringLookupElement;

static std::string UNKNOWN_DEVICE_STRING = "UNKNOWN_DEVICE";

static DeviceTypeStringLookupElement DEVICE_TYPE_STRING_LOOKUP_TABLE[]{
    // Device Type					String
    //------------------------------------------------------
    {Device::DeviceType::U50, "u50"},
    {Device::DeviceType::U200, "u200"},
    {Device::DeviceType::U250, "u250"},
    {Device::DeviceType::U280, "u280"},

    {Device::DeviceType::UNKNOWN, UNKNOWN_DEVICE_STRING}};

static const uint32_t DEVICE_TYPE_STRING_LOOKUP_TABLE_SIZE =
    sizeof(DEVICE_TYPE_STRING_LOOKUP_TABLE) / sizeof(DEVICE_TYPE_STRING_LOOKUP_TABLE[0]);

Device::Device() {
    m_pOwner = nullptr;
}

Device::Device(cl::Device clDevice) {
    m_clDevice = clDevice;

    m_clDevice.getInfo(CL_DEVICE_NAME, &m_deviceName);

    setupDeviceType();

    m_pOwner = nullptr;
}

Device::~Device() {}

std::string Device::getName(void) {
    return m_deviceName;
}

Device::DeviceType Device::getDeviceType(void) {
    return m_deviceType;
}

void Device::setupDeviceType(void) {
    uint32_t i;
    bool bFound = false;
    DeviceTypeStringLookupElement* pLookupElement;

    for (i = 0; i < DEVICE_TYPE_STRING_LOOKUP_TABLE_SIZE; i++) {
        pLookupElement = &DEVICE_TYPE_STRING_LOOKUP_TABLE[i];

        if (pLookupElement->deviceType != Device::DeviceType::UNKNOWN) {
            if (m_deviceName.find(pLookupElement->deviceTypeString) != std::string::npos) {
                m_deviceType = pLookupElement->deviceType;
                bFound = true;
                break;
            }
        }
    }

    if (bFound == false) {
        m_deviceType = DeviceType::UNKNOWN;
    }
}

std::string Device::getDeviceTypeString(void) {
    uint32_t i;
    DeviceTypeStringLookupElement* pLookupElement;
    std::string s = UNKNOWN_DEVICE_STRING;

    for (i = 0; i < DEVICE_TYPE_STRING_LOOKUP_TABLE_SIZE; i++) {
        pLookupElement = &DEVICE_TYPE_STRING_LOOKUP_TABLE[i];

        if (pLookupElement->deviceType == m_deviceType) {
            s = pLookupElement->deviceTypeString;
            break;
        }
    }

    return s;
}

cl::Device Device::getCLDevice(void) {
    return m_clDevice;
}

int Device::claim(OCLController* pOwner) {
    int retval = XLNX_OK;

    m_mutex.lock();

    if (pOwner == nullptr) {
        retval = XLNX_ERROR_DEVICE_INVALID_OCL_CONTROLLER;
    }

    if (retval == XLNX_OK) {
        if (m_pOwner == pOwner) {
            retval = XLNX_ERROR_DEVICE_ALREADY_OWNED_BY_THIS_OCL_CONTROLLER;
        }
    }

    if (retval == XLNX_OK) {
        if (m_pOwner != nullptr) {
            if (m_pOwner != pOwner) {
                // Another OCL controller is attempting to claim the device when it is
                // already claimed by a different OCL Controller
                retval = XLNX_ERROR_DEVICE_OWNED_BY_ANOTHER_OCL_CONTROLLER;
            }
        }
    }

    if (retval == XLNX_OK) {
        m_pOwner = pOwner;
    }

    m_mutex.unlock();

    return retval;
}

int Device::release(OCLController* pOwner) {
    int retval = XLNX_OK;

    m_mutex.lock();

    if (pOwner == nullptr) {
        retval = XLNX_ERROR_DEVICE_INVALID_OCL_CONTROLLER;
    }

    if (retval == XLNX_OK) {
        if (m_pOwner != pOwner) {
            retval = XLNX_ERROR_DEVICE_NOT_OWNED_BY_SPECIFIED_OCL_CONTROLLER;
        }
    }

    if (retval == XLNX_OK) {
        m_pOwner = nullptr;
    }

    m_mutex.unlock();

    return retval;
}
