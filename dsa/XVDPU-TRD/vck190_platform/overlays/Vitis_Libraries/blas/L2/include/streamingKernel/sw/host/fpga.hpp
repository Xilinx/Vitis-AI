/**********
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
 * **********/
/**
 *  @brief FPGA utilities
 *
 *  $DateTime: 2018/01/30 15:02:37 $
 */

#ifndef XF_BLAS_FPGA_HPP
#define XF_BLAS_FPGA_HPP

#include "assert.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iterator>
#include "xcl2.hpp"
#include "host_types.hpp"

namespace xf {

namespace blas {

class Fpga {
   protected:
    std::string m_XclbinFile;

    cl::Context m_Context;
    cl::CommandQueue m_CommandQueue;
    cl::Program m_Program;
    cl::Kernel m_Kernels[BLAS_numKernels];
    std::vector<cl::Memory> m_Buffers[BLAS_numKernels];
    std::vector<cl::Event> m_Mem2FpgaEvents[BLAS_numKernels];
    std::vector<cl::Event> m_ExeKernelEvents[BLAS_numKernels];

    unsigned int m_deviceId;

   public:
    Fpga(unsigned int p_deviceId = 0) { m_deviceId = p_deviceId; }

    bool loadXclbin(std::string p_xclbinFile) {
        bool ok = false;
        std::vector<cl::Device> l_devices = xcl::get_xil_devices();
        cl::Device l_device = l_devices[m_deviceId];
        std::string l_deviceName = l_device.getInfo<CL_DEVICE_NAME>();
        std::cout << "INFO: device name is: " << l_deviceName << std::endl;
        // Create the OpenCL context, cmmandQueue and program
        cl::Context l_context(l_device);
        m_Context = l_context;
        cl::CommandQueue l_cmdQueue(m_Context, l_device,
                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
        m_CommandQueue = l_cmdQueue;
        cl::Program::Binaries l_bins = xcl::import_binary_file(p_xclbinFile);
        l_devices.resize(1);
        cl::Program l_program(m_Context, {l_device}, l_bins);
        m_Program = l_program;
        ok = true;
        return (ok);
    }

    bool createKernel(unsigned int p_kernelId, std::string p_kernelName) {
        bool ok = false;
        assert(p_kernelId < BLAS_numKernels);
        std::string l_name = p_kernelName + ":{" + p_kernelName + "_" + std::to_string(p_kernelId) + "}";
        cl::Kernel l_kernel(m_Program, l_name.c_str());
        m_Kernels[p_kernelId] = l_kernel;
        ok = true;
        return (ok);
    }

    bool createBufferForKernel(unsigned int p_kernelId, MemDesc p_memDesc) {
        bool ok = false;

        assert(p_kernelId < BLAS_numKernels);
        cl::Buffer l_buffer(m_Context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, p_memDesc.sizeBytes(),
                            p_memDesc.data());
        m_Buffers[p_kernelId].push_back(l_buffer);
        m_Kernels[p_kernelId].setArg(0, l_buffer);
        m_Kernels[p_kernelId].setArg(1, l_buffer);
        ok = true;
        return (ok);
    }

    bool copyToKernel(unsigned int p_kernelId) {
        bool ok = false;
        assert(p_kernelId < BLAS_numKernels);
        cl::Event l_event;
        // Send the input data to the accelerator
        m_CommandQueue.enqueueMigrateMemObjects(m_Buffers[p_kernelId], 0 /* 0 means from host*/, NULL, &l_event);
        m_Mem2FpgaEvents[p_kernelId].push_back(l_event);
        ok = true;
        return (ok);
    }

    bool callKernel(unsigned int p_kernelId) {
        bool ok = false;
        assert(p_kernelId < BLAS_numKernels);
        cl::Event l_event;
        m_CommandQueue.enqueueTask(m_Kernels[p_kernelId], &(m_Mem2FpgaEvents[p_kernelId]), &l_event);
        m_ExeKernelEvents[p_kernelId].push_back(l_event);
        m_Mem2FpgaEvents[p_kernelId].clear();
        ok = true;
        return (ok);
    }

    bool copyFromKernel(unsigned int p_kernelId) {
        bool ok = false;
        assert(p_kernelId < BLAS_numKernels);
        cl::Event l_event;
        m_CommandQueue.enqueueMigrateMemObjects(m_Buffers[p_kernelId], CL_MIGRATE_MEM_OBJECT_HOST,
                                                &(m_ExeKernelEvents[p_kernelId]));
        m_ExeKernelEvents[p_kernelId].clear();
        ok = true;
        return (ok);
    }

    bool finish() {
        bool ok = false;
        m_CommandQueue.finish();
        ok = true;
        return (ok);
    }
};
}
} // namespace
#endif
