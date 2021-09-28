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
#include "hpc/L2/include/sw/fpga.hpp"
template <unsigned int t_NumChannels>
class KernelLoadNnz : public Kernel {
   public:
    KernelLoadNnz(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<uint8_t> p_sigBuf[t_NumChannels]) {
        cl_int err;
        vector<cl::Memory> l_buffers;
        for (unsigned int i = 0; i < t_NumChannels; ++i) {
            m_buffer[i] = createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, p_sigBuf[i]);
            OCL_CHECK(err, err = m_kernel.setArg(i, m_buffer[i]));
            l_buffers.push_back(m_buffer[i]);
        }
        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(l_buffers, 0)); /* 0 means from host*/
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer[t_NumChannels];
};

class KernelLoadIdx : public Kernel {
   public:
    KernelLoadIdx(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<uint8_t>& p_buf) {
        cl_int err;
        m_buffer = createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, p_buf);
        OCL_CHECK(err, err = m_kernel.setArg(0, m_buffer));
        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer}, 0)); /* 0 means from host*/
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer;
};

class KernelLoadCol : public Kernel {
   public:
    KernelLoadCol(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<uint8_t>& p_paramBuf, host_buffer_t<uint8_t>& p_xBuf) {
        cl_int err;
        m_buffer[0] = createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, p_paramBuf);
        OCL_CHECK(err, err = m_kernel.setArg(0, m_buffer[0]));
        m_buffer[1] = createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, p_xBuf);
        OCL_CHECK(err, err = m_kernel.setArg(1, m_buffer[1]));
        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer[0], m_buffer[1]},
                                                                                0)); /* 0 means from host*/
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer[2];
};

class KernelLoadRbParam : public Kernel {
   public:
    KernelLoadRbParam(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<uint8_t>& p_buf) {
        cl_int err;
        m_buffer = createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, p_buf);
        OCL_CHECK(err, err = m_kernel.setArg(0, m_buffer));
        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer}, 0)); /* 0 means from host*/
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer;
};

class KernelStoreY : public Kernel {
   public:
    KernelStoreY(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setArgs(unsigned int p_rows, host_buffer_t<uint8_t>& p_buf) {
        cl_int err;
        OCL_CHECK(err, err = m_kernel.setArg(0, p_rows));
        m_buffer = createDeviceBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, p_buf);
        OCL_CHECK(err, err = m_kernel.setArg(1, m_buffer));
    }
    void getMem() {
        // Copy input data to device global memory
        cl_int err;
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(
                           {m_buffer}, CL_MIGRATE_MEM_OBJECT_HOST)); /* 0 means from host*/
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer;
};
