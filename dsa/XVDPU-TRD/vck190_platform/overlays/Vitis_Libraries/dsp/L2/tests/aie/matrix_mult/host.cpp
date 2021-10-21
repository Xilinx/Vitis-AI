/*
 * Copyright 2021 Xilinx, Inc.
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

#include "test.cpp"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <sstream>

//#include "input.h"
//#include "input1.h"
//#include "golden.h"

// This is used for the PL Kernels
#include "xrt/xrt.h"
#include "xrt/experimental/xrt_kernel.h"

// Using the Cardano API that call XRT API
#include "adf/adf_api/XRTConfig.h"
extern "C" {
#include <xaiengine.h>
}

void readTxt(std::string file, int16_t* buffer, int size) {
    int index = 0;
    char line[1024] = {0};

    std::fstream fhdl(file.c_str(), std::ios::in);
    if (!fhdl) {
        std::cout << "ERROR: " << file << " file could not open !" << std::endl;
        exit(1);
    }
    while (fhdl.getline(line, sizeof(line))) {
        std::string str(line);
        std::replace(str.begin(), str.end(), ',', ' ');
        std::stringstream data(str.c_str());
        data >> buffer[index++];
        data >> buffer[index++];
        if (index >= size) {
            break;
        }
    }
    std::cout << "File: " << file << "; Total input data is " << index << std::endl;
}

static std::vector<char> load_xclbin(xrtDeviceHandle device, const std::string& fnm) {
    if (fnm.empty()) throw std::runtime_error("No xclbin specified");

    // load bit stream
    std::ifstream stream(fnm);
    stream.seekg(0, stream.end);
    size_t size = stream.tellg();
    stream.seekg(0, stream.beg);

    std::vector<char> header(size);
    stream.read(header.data(), size);

    auto top = reinterpret_cast<const axlf*>(header.data());
    if (xrtDeviceLoadXclbin(device, top)) throw std::runtime_error("Xclbin loading failed");

    return header;
}

int main(int argc, char** argv) {
    //////////////////////////////////////////
    // Open xclbin
    //////////////////////////////////////////
    auto dhdl = xrtDeviceOpen(0); // Open Device the local device
    if (dhdl == nullptr) throw std::runtime_error("No valid device handle found. Make sure using right xclOpen index.");
    auto xclbin = load_xclbin(dhdl, "kernel.xclbin");
    auto top = reinterpret_cast<const axlf*>(xclbin.data());
    adf::registerXRT(dhdl, top->m_header.uuid);

    xrtKernelHandle mm2s_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "mm2s");
    xrtKernelHandle mm2s_khdl1 = xrtPLKernelOpen(dhdl, top->m_header.uuid, "bmm2s");
    xrtKernelHandle s2mm_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm");

    int sizeIn = P_INPUT_SAMPLES_A;  // 256;
    int sizeIn1 = P_INPUT_SAMPLES_B; // 256;
    int sizeOut = P_OUTPUT_SAMPLES;  // 256;

    void* cint16Input_ptr = NULL;
    void* cint16Input1_ptr = NULL;
    void* cint16golden_ptr = NULL;
    posix_memalign(&cint16Input_ptr, 4096, (sizeIn * 2 * sizeof(int16_t)));
    posix_memalign(&cint16Input1_ptr, 4096, (sizeIn1 * 2 * sizeof(int16_t)));
    posix_memalign(&cint16golden_ptr, 4096, (sizeOut * 2 * sizeof(int16_t)));
    int16_t* cint16Input = reinterpret_cast<int16_t*>(cint16Input_ptr);
    int16_t* cint16Input1 = reinterpret_cast<int16_t*>(cint16Input1_ptr);
    int16_t* cint16golden = reinterpret_cast<int16_t*>(cint16golden_ptr);

    readTxt(INPUT_FILE_A, cint16Input, sizeIn * 2);
    readTxt(INPUT_FILE_B, cint16Input1, sizeIn1 * 2);
    readTxt(REF_OUTPUT_FILE, cint16golden, sizeOut * 2);

    //////////////////////////////////////////
    // input memory
    // Allocating the input size of sizeIn to MM2S
    // This is using low-level XRT call xclAllocBO to allocate the memory
    //////////////////////////////////////////

    xrtBufferHandle in_bohdl = xrtBOAlloc(dhdl, sizeIn * sizeof(int16_t) * 2, 0, xrtKernelArgGroupId(mm2s_khdl, 0));
    auto in_bomapped = reinterpret_cast<int16_t*>(xrtBOMap(in_bohdl));
    memcpy(in_bomapped, cint16Input, sizeIn * sizeof(int16_t) * 2);
    printf("Input memory virtual addr 0x%llx\n", in_bomapped);

    xrtBufferHandle in_bohdl1 = xrtBOAlloc(dhdl, sizeIn1 * sizeof(int16_t) * 2, 0, xrtKernelArgGroupId(mm2s_khdl1, 0));
    auto in_bomapped1 = reinterpret_cast<int16_t*>(xrtBOMap(in_bohdl1));
    memcpy(in_bomapped1, cint16Input1, sizeIn1 * sizeof(int16_t) * 2);
    printf("Input memory virtual addr 0x%llx\n", in_bomapped1);

    //////////////////////////////////////////
    // output memory
    // Allocating the output size of sizeOut to S2MM
    // This is using low-level XRT call xclAllocBO to allocate the memory
    //////////////////////////////////////////

    xrtBufferHandle out_bohdl = xrtBOAlloc(dhdl, sizeOut * sizeof(int16_t) * 2, 0, xrtKernelArgGroupId(s2mm_khdl, 1));
    auto out_bomapped = reinterpret_cast<int16_t*>(xrtBOMap(out_bohdl));
    memset(out_bomapped, 0x0, sizeOut * sizeof(int16_t) * 2);
    printf("Output memory virtual addr 0x%llx\n", out_bomapped);

    //////////////////////////////////////////
    // mm2s ip
    // Using the xrtPLKernelOpen function to manually control the PL Kernel
    // that is outside of the AI Engine graph
    //////////////////////////////////////////

    // Need to provide the kernel handle, and the argument order of the kernel arguments
    // Here the in_bohdl is the input buffer, the nullptr is the streaming interface and must be null,
    // lastly, the size of the data. This info can be found in the kernel definition.
    xrtRunHandle mm2s_rhdl = xrtKernelRun(mm2s_khdl, in_bohdl, nullptr, sizeIn * sizeof(int16_t) * 2);
    printf("run mm2s\n");

    xrtRunHandle mm2s_rhdl1 = xrtKernelRun(mm2s_khdl1, in_bohdl1, nullptr, sizeIn1 * sizeof(int16_t) * 2);
    printf("run bmm2s\n");

    //////////////////////////////////////////
    // s2mm ip
    // Using the xrtPLKernelOpen function to manually control the PL Kernel
    // that is outside of the AI Engine graph
    //////////////////////////////////////////

    // Need to provide the kernel handle, and the argument order of the kernel arguments
    // Here the out_bohdl is the output buffer, the nullptr is the streaming interface and must be null,
    // lastly, the size of the data. This info can be found in the kernel definition.
    xrtRunHandle s2mm_rhdl = xrtKernelRun(s2mm_khdl, nullptr, out_bohdl, sizeOut * sizeof(int16_t) * 2);
    printf("run s2mm\n");

    //////////////////////////////////////////
    // graph execution for AIE
    //////////////////////////////////////////

    printf("graph init. This does nothing because CDO in boot PDI already configures AIE.\n");
    matMult.init();

    printf("graph run\n");
    matMult.run(1);

    matMult.end();
    printf("graph end\n");

    //////////////////////////////////////////
    // wait for mm2s done
    //////////////////////////////////////////

    auto state = xrtRunWait(mm2s_rhdl);
    std::cout << "mm2s completed with status(" << state << ")\n";
    xrtRunClose(mm2s_rhdl);
    xrtKernelClose(mm2s_khdl);

    auto state1 = xrtRunWait(mm2s_rhdl1);
    std::cout << "bmm2s completed with status(" << state1 << ")\n";
    xrtRunClose(mm2s_rhdl1);
    xrtKernelClose(mm2s_khdl1);

    //////////////////////////////////////////
    // wait for s2mm done
    //////////////////////////////////////////

    state = xrtRunWait(s2mm_rhdl);
    std::cout << "s2mm completed with status(" << state << ")\n";
    xrtRunClose(s2mm_rhdl);
    xrtKernelClose(s2mm_khdl);

    //////////////////////////////////////////
    // Comparing the execution data to the golden data
    //////////////////////////////////////////
    int ret = xrtBOSync(out_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, sizeOut * sizeof(int16_t) * 2, 0);

    int errorCount = 0;
    {
        for (int i = 0; i < sizeOut * 2; i++) {
            if (out_bomapped[i] != cint16golden[i]) {
                printf("Error found @ %d, %d != %d\n", i, out_bomapped[i], cint16golden[i]);
                errorCount++;
            }
        }

        if (errorCount)
            printf("Test failed with %d errors\n", errorCount);
        else
            printf("Test passed\n");
    }

    //////////////////////////////////////////
    // clean up XRT
    //////////////////////////////////////////

    std::cout << "Releasing remaining XRT objects...\n";
    // xrtBOUnmap(dhdl, in_bohdl, in_bomapped);
    // xrtBOUnmap(dhdl, out_bohdl, out_bomapped);
    xrtBOFree(in_bohdl);
    xrtBOFree(in_bohdl1);
    xrtBOFree(out_bohdl);
    xrtDeviceClose(dhdl);

    return errorCount;
}
