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

#include <cstring>
#include <iostream>
#include <string>

#include "xf_fintech_device_manager.hpp"

using namespace xf::fintech;

DeviceManager* DeviceManager::m_instance = nullptr;

DeviceManager::DeviceManager() {
    createDeviceList();
}

DeviceManager::~DeviceManager() {
    deleteDeviceList();
}

DeviceManager* DeviceManager::getInstance(void) {
    if (m_instance == nullptr) {
        m_instance = new DeviceManager();
    }

    return m_instance;
}

void DeviceManager::createDeviceList(void) {
    std::vector<cl::Device> xclDeviceList;

    xclDeviceList = xcl::get_xil_devices();

    for (unsigned int i = 0; i < xclDeviceList.size(); i++) {
        Device* device = new Device(xclDeviceList[i]);

        m_deviceList.push_back(device);
    }
}

void DeviceManager::deleteDeviceList(void) {
    unsigned int i;
    for (i = 0; i < m_deviceList.size(); i++) {
        Device* device = m_deviceList[i];
        delete (device);
    }

    m_deviceList.clear();
}

std::vector<Device*> DeviceManager::getDeviceList(void) {
    DeviceManager* pInstance;

    pInstance = DeviceManager::getInstance();

    return pInstance->m_deviceList;
}

std::vector<Device*> DeviceManager::getDeviceList(std::string deviceSubString) {
    std::vector<Device*> fullList;
    std::vector<Device*> matchingList;
    std::string s;

    fullList = DeviceManager::getDeviceList();

    for (unsigned int i = 0; i < fullList.size(); i++) {
        Device* device = fullList[i];

        s = device->getName();

        // check for path
        std::string deviceName = deviceSubString;
        const size_t last_slash_idx = deviceName.find_last_of("\\/");
        if (std::string::npos != last_slash_idx) {
            deviceName.erase(0, last_slash_idx + 1);
        }

        // check for extension
        const size_t period_idx = deviceName.rfind('.');
        if (std::string::npos != period_idx) {
            deviceName.erase(period_idx);
        }

        if (s.find(deviceName) != std::string::npos) {
            matchingList.push_back(device);
        }
    }

    return matchingList;
}

std::vector<Device*> DeviceManager::getDeviceList(Device::DeviceType deviceType) {
    std::vector<Device*> fullList;
    std::vector<Device*> matchingList;
    std::string s;

    fullList = DeviceManager::getDeviceList();

    for (unsigned int i = 0; i < fullList.size(); i++) {
        Device* device = fullList[i];

        if (device->getDeviceType() == deviceType) {
            matchingList.push_back(device);
        }
    }

    return matchingList;
}
