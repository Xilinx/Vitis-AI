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
#ifndef XF_HPC_FPGA_XRT_HPP
#define XF_HPC_FPGA_XRT_HPP

#include <map>

#include "xcl2.hpp"

#include "experimental/xrt_kernel.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_bo.h"

using namespace std;

template <typename T>
using host_buffer_t = std::vector<T, aligned_allocator<T> >;

/**
 * @brief class FPGA is used to manage FPGA info
 */
class FPGA {
   public:
    xrt::device m_device;
    xrt::uuid m_uuid;
    map<const void*, xrt::bo> m_bufferMaps;

    FPGA() = delete;
    FPGA(const char* p_xclbin, unsigned int deviceIndex = 0) {
        m_device = xrt::device(deviceIndex);
        m_uuid = m_device.load_xclbin(p_xclbin);
    }

    template <typename T>
    xrt::bo createDeviceBuffer(const host_buffer_t<T>& p_buffer, unsigned int p_mem) {
        const void* l_ptr = (const void*)p_buffer.data();
        if (exists(l_ptr)) {
            return m_bufferMaps[l_ptr];
        } else {
            size_t l_bufferSize = sizeof(T) * p_buffer.size();
            m_bufferMaps.insert({l_ptr, xrt::bo(m_device, (void*)p_buffer.data(), l_bufferSize, p_mem)});
            return m_bufferMaps[l_ptr];
        }
    }

    bool copyToFpga(xrt::bo& p_bufHandle) {
        p_bufHandle.sync(XCL_BO_SYNC_BO_TO_DEVICE, p_bufHandle.size(), 0);
        return true;
    }

    bool copyFromFpga(xrt::bo& p_bufHandle) {
        p_bufHandle.sync(XCL_BO_SYNC_BO_FROM_DEVICE, p_bufHandle.size(), 0);
        return true;
    }

   protected:
    bool exists(const void* p_ptr) const {
        auto it = m_bufferMaps.find(p_ptr);
        return it != m_bufferMaps.end();
    }
};

class XKernel {
   public:
    FPGA* m_fpga;
    xrt::kernel m_kernel;
    xrt::run m_run;

    XKernel(FPGA* p_fpga = nullptr) : m_fpga(p_fpga) {}

    void init(FPGA* p_fpga) { m_fpga = p_fpga; }
    void getCU(const string& p_name) { m_kernel = xrt::kernel(m_fpga->m_device, m_fpga->m_uuid.get(), p_name); }
    void run() { m_run.wait(); }

    static void runAll(const vector<XKernel>& p_kernels) {
        auto start = chrono::high_resolution_clock::now();
        vector<future<void> > runall;
        for (auto ker : p_kernels) ker.m_run.wait();

        auto finish = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = finish - start;
        double t_sec = elapsed.count();
        cout << "Execution time " << t_sec << "s." << endl;
    }

    template <typename T>
    xrt::bo createDeviceBuffer(const host_buffer_t<T>& p_buffer, unsigned int p_mem) const {
        return m_fpga->createDeviceBuffer(p_buffer, p_mem);
    }
};

#endif
