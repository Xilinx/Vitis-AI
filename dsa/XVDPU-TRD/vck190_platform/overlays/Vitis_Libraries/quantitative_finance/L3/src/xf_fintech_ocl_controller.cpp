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

#include <stdio.h>

#include "xf_fintech_error_codes.hpp"
#include "xf_fintech_ocl_controller.hpp"

using namespace xf::fintech;

OCLController::OCLController() {
    m_pDevice = nullptr;
    m_bDeviceIsPrepared = false;
}

OCLController::~OCLController() {}

int OCLController::claimDevice(Device* device) {
    int retval = XLNX_OK;
    bool bClaimedDevice = false;

    m_mutex.lock();

    if (m_pDevice != nullptr) {
        // we need to limit an OCL controller to only owning one device...
        // if the user wishes to switch device, they must first release the one they
        // already own.
        if (m_pDevice != device) {
            retval = XLNX_ERROR_OCL_CONTROLLER_ALREADY_OWNS_ANOTHER_DEVICE;
        }
    }

    if (retval == XLNX_OK) {
        // try to claim the device...
        // the device knows which OCLcontroller (if any) owns it and will reject
        // this call
        // if another already OCLController owns it.
        retval = device->claim(this);
    }

    if (retval == XLNX_OK) {
        // SUCCESS - this OCL controller now owns the device!
        bClaimedDevice = true;

        // Stash the device object...we will need it later when we come to release
        // it...
        m_pDevice = device;
    }

    if (retval == XLNX_OK) {
        // create our OCL Objects.  This will suck in the XCLBIN file, program the
        // FPGA, create buffers, etc...
        retval = createOCLObjects(m_pDevice);
    }

    if (retval == XLNX_OK) {
        m_bDeviceIsPrepared = true;
    } else {
        // uh-oh...something failed in the creation of our OCL objects...
        // if we claimed the device earlier, we should release it...
        if (bClaimedDevice) {
            device->release(this);
            m_pDevice = nullptr;
        }
    }

    m_mutex.unlock();

    return retval;
}

int OCLController::releaseDevice(void) {
    int retval = XLNX_OK;

    m_mutex.lock();

    if (m_pDevice == nullptr) {
        retval = XLNX_ERROR_OCL_CONTROLLER_DOES_NOT_OWN_ANY_DEVICE;
    }

    if (retval == XLNX_OK) {
        m_bDeviceIsPrepared = false;

        releaseOCLObjects();

        m_pDevice->release(this);
        m_pDevice = nullptr;
    }

    m_mutex.unlock();

    return retval;
}

bool OCLController::deviceIsPrepared(void) {
    return m_bDeviceIsPrepared;
}

void OCLController::setCLError(cl_int clError) {
    m_clError = clError;
}

cl_int OCLController::getCLError(void) {
    return m_clError;
}
