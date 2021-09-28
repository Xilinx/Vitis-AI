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

#ifndef XF_BLAS_GEMM_STREAM_HPP
#define XF_BLAS_GEMM_STREAM_HPP

#include "assert.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iterator>
#include "xcl2/xcl2.hpp"
#include "host_types.hpp"
#include "fpga.hpp"

namespace xf {

namespace blas {

template <int t_NumKernels>
class GemmStream : public Fpga {
   protected:
    cl::Kernel m_LdKernels[t_NumKernels];
    cl::Kernel m_StKernels[t_NumKernels];

   public:
    GemmStream(unsigned int p_deviceId = 0) : Fpga(p_deviceId) {}

    bool createKernel(unsigned int p_kernelId) {
        bool ok = false;
        assert(p_kernelId < t_NumKernels);

        std::stringstream l_ldName, l_stName;
        l_ldName << "gemmLoadKernel"
                 << ":{"
                 << "gemmLoadKernel"
                 << "_" << std::to_string(p_kernelId) << "}";
        m_LdKernels[p_kernelId] = cl::Kernel(m_Program, l_ldName.str().c_str());

        l_stName << "gemmStoreKernel"
                 << ":{"
                 << "gemmStoreKernel"
                 << "_" << std::to_string(p_kernelId) << "}";
        m_StKernels[p_kernelId] = cl::Kernel(m_Program, l_stName.str().c_str());
        ok = true;
        return (ok);
    }

    bool createBufferForKernel(unsigned int p_kernelId, MemDesc p_memDesc) {
        bool ok = false;

        assert(p_kernelId < t_NumKernels);
        cl::Buffer l_buffer(m_Context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, p_memDesc.sizeBytes(),
                            p_memDesc.data());
        m_Buffers[p_kernelId].push_back(l_buffer);
        m_LdKernels[p_kernelId].setArg(0, l_buffer);
        m_StKernels[p_kernelId].setArg(0, l_buffer);
        ok = true;
        return (ok);
    }

    bool callKernel(unsigned int p_kernelId) {
        bool ok = false;
        assert(p_kernelId < t_NumKernels);
        cl::Event l_event[2];
        m_CommandQueue.enqueueTask(m_LdKernels[p_kernelId], &(m_Mem2FpgaEvents[p_kernelId]), &l_event[0]);
        m_CommandQueue.enqueueTask(m_StKernels[p_kernelId], &(m_Mem2FpgaEvents[p_kernelId]), &l_event[1]);
        m_ExeKernelEvents[p_kernelId].insert(m_ExeKernelEvents[p_kernelId].end(), l_event, l_event + 2);
        m_Mem2FpgaEvents[p_kernelId].clear();
        ok = true;
        return (ok);
    }
};
}
} // namespace
#endif
