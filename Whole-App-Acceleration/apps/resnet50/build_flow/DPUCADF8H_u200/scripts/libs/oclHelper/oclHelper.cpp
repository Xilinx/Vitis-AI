/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
#include "oclHelper.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

//
// Load file to memory
//
static int loadFile2Memory(const char *filename, char **result) {
    int size = 0;

    std::ifstream stream(filename, std::ifstream::binary);
    if (!stream) {
        return -1;
    }

    stream.seekg(0, stream.end);
    size = stream.tellg();
    stream.seekg(0, stream.beg);

    *result = new char[size + 1];
    stream.read(*result, size);
    if (!stream) {
        return -2;
    }
    stream.close();
    (*result)[size] = 0;
    return size;
}

//
// Get device version
//
static void getDeviceVersion(oclHardware &hardware) {
    char versionString[512];
    size_t size = 0;
    cl_int err = clGetDeviceInfo(
        hardware.mDevice, CL_DEVICE_VERSION, 511, versionString, &size);
    if (err != CL_SUCCESS) {
        std::cout << oclErrorCode(err) << "\n";
        return;
    }
    unsigned major = 0;
    unsigned minor = 0;
    unsigned state = 0;
    for (size_t i = 0; i < size; i++) {
        if (!versionString[i]) {
            break;
        }
        if (versionString[i] == ' ') {
            state++;
            continue;
        }
        if (versionString[i] == '.') {
            state++;
            continue;
        }
        if (state == 0) {
            continue;
        }
        if (state == 1) {
            major *= 10;
            major += (versionString[i] - '0');
            continue;
        }
        if (state == 2) {
            minor *= 10;
            minor += (versionString[i] - '0');
            continue;
        }
        break;
    }
    hardware.mMajorVersion = major;
    hardware.mMinorVersion = minor;
}

//
// Get OCL hardware
//
oclHardware getOclHardware(cl_device_type type) {
    oclHardware hardware = {0, 0, 0, 0, 0, 0};
    cl_platform_id platforms[16] = {0};
    cl_device_id devices[16];
    char platformName[256];
    char deviceName[256];
    cl_uint platformCount = 0;
    cl_int err = clGetPlatformIDs(0, 0, &platformCount);
    err = clGetPlatformIDs(16, platforms, &platformCount);
    if (err != CL_SUCCESS) {
        std::cout << oclErrorCode(err) << "\n";
        return hardware;
    }

    for (cl_uint i = 0; i < platformCount; i++) {
        err = clGetPlatformInfo(
            platforms[i], CL_PLATFORM_NAME, 256, platformName, 0);
        if (err != CL_SUCCESS) {
            std::cout << oclErrorCode(err) << "\n";
            return hardware;
        }
        cl_uint deviceCount = 0;
        err = clGetDeviceIDs(platforms[i], type, 16, devices, &deviceCount);
        if ((err != CL_SUCCESS) || (deviceCount == 0)) {
            continue;
        }

        err = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 256, deviceName, 0);
        if (err != CL_SUCCESS) {
            std::cout << oclErrorCode(err) << "\n";
            return hardware;
        }

        cl_context_properties contextData[3] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0};
        cl_context context =
            clCreateContextFromType(contextData, type, 0, 0, &err);
        if (err != CL_SUCCESS) {
            continue;
        }
        cl_command_queue queue =
            clCreateCommandQueue(context, devices[0], 0, &err);
        if (err != CL_SUCCESS) {
            std::cout << oclErrorCode(err) << "\n";
            return hardware;
        }

        hardware.mPlatform = platforms[i];
        hardware.mContext = context;
        hardware.mDevice = devices[0];
        hardware.mQueue = queue;
        getDeviceVersion(hardware);
        std::cout << "Platform = " << platformName << "\n";
        std::cout << "Device = " << deviceName << "\n";
        std::cout << "OpenCL Version = " << hardware.mMajorVersion << '.'
                  << hardware.mMinorVersion << "\n";
        return hardware;
    }
    return hardware;
}

//
// Get OCL software
//
int getOclSoftware(oclSoftware &software, const oclHardware &hardware) {
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
    cl_int err = clGetDeviceInfo(
        hardware.mDevice, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, 0);
    if (err != CL_SUCCESS) {
        std::cout << oclErrorCode(err) << "\n";
        return -1;
    }

    unsigned char *kernelCode = 0;
    std::cout << "Loading " << software.mFileName << "\n";

    int size = loadFile2Memory(software.mFileName, (char **)&kernelCode);
    if (size < 0) {
        std::cout << "Failed to load kernel\n";
        return -2;
    }

    if (deviceType == CL_DEVICE_TYPE_ACCELERATOR) {
        size_t n = size;
        software.mProgram =
            clCreateProgramWithBinary(hardware.mContext,
                                      1,
                                      &hardware.mDevice,
                                      &n,
                                      (const unsigned char **)&kernelCode,
                                      0,
                                      &err);
    } else {
        software.mProgram = clCreateProgramWithSource(
            hardware.mContext, 1, (const char **)&kernelCode, 0, &err);
    }
    if (!software.mProgram || (err != CL_SUCCESS)) {
        std::cout << oclErrorCode(err) << "\n";
        return -3;
    }

    software.mKernel =
        clCreateKernel(software.mProgram, software.mKernelName, NULL);
    if (software.mKernel == 0) {
        std::cout << oclErrorCode(err) << "\n";
        return -4;
    }

    delete[] kernelCode;
    return 0;
}

//
// Release software and hardware
//
void release(oclSoftware &software) {
    clReleaseKernel(software.mKernel);
    clReleaseProgram(software.mProgram);
}

void release(oclHardware &hardware) {
    clReleaseCommandQueue(hardware.mQueue);
    clReleaseContext(hardware.mContext);
    if ((hardware.mMajorVersion >= 1) && (hardware.mMinorVersion > 1)) {
        // Only available in OpenCL >= 1.2
        clReleaseDevice(hardware.mDevice);
    }
}
