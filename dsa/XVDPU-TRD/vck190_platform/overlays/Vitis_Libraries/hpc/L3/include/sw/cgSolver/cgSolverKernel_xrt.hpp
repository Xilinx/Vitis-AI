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
//#include "fpga.hpp"

#include <stdlib.h>
#include <cstdlib>
#include <future>

#include "fpga_xrt.hpp"

//#define DEBUG

using namespace std;

class CGKernelControl : public XKernel {
   public:
    CGKernelControl(FPGA* p_fpga = nullptr) : XKernel(p_fpga) {}
    void setMem(host_buffer_t<uint8_t>& h_instr) {
        m_buffer_instr = createDeviceBuffer(h_instr, m_kernel.group_id(0));

        m_fpga->copyToFpga(m_buffer_instr);
        m_run = m_kernel(m_buffer_instr);
    }

    void run() { auto state = m_run.wait(); }

    void getMem() { m_fpga->copyFromFpga(m_buffer_instr); }

   private:
    xrt::bo m_buffer_instr;
};

template <typename T, int t_ParEntries, int t_NumChannels = 1>
class CGKernelGemv : public XKernel {
   public:
    CGKernelGemv(FPGA* p_fpga = nullptr) : XKernel(p_fpga) {}
    void setMem(host_buffer_t<T>& h_A, host_buffer_t<T>& h_pk, host_buffer_t<T>& h_Apk) {
        vector<host_buffer_t<T> > h_As(t_NumChannels);
        uint32_t l_vecSize = h_pk.size();
        // Running the kernel

        if (t_NumChannels == 1) {
            m_buffer_A.push_back(createDeviceBuffer(h_A, m_kernel.group_id(0)));
        } else {
            for (int i = 0; i < t_NumChannels; i++) h_As[i].resize(h_A.size() / t_NumChannels);

            for (int i = 0; i < l_vecSize; i++)
                copy(h_A.begin() + i * l_vecSize, h_A.begin() + (i + 1) * l_vecSize,
                     h_As[i % t_NumChannels].begin() + (i / t_NumChannels) * l_vecSize);

            for (int i = 0; i < t_NumChannels; i++) {
                m_buffer_A.push_back(createDeviceBuffer(h_As[i], m_kernel.group_id(i)));
#if DEBUG
                cout << "gemv cu here is " << m_kernel.group_id(i) << "\n";
#endif
            }
        }

        m_buffer_pk = createDeviceBuffer(h_pk, m_kernel.group_id(t_NumChannels));
        m_buffer_Apk = createDeviceBuffer(h_Apk, m_kernel.group_id(t_NumChannels + 2));

        // Copy input data to device global memory
        for (int i = 0; i < t_NumChannels; i++) {
            m_fpga->copyToFpga(m_buffer_A[i]);
        }
        m_fpga->copyToFpga(m_buffer_pk);
        m_fpga->copyToFpga(m_buffer_Apk);

        m_run = m_kernel(m_buffer_A[0], m_buffer_A[1], m_buffer_A[2], m_buffer_A[3], m_buffer_A[4], m_buffer_A[5],
                         m_buffer_A[6], m_buffer_A[7], m_buffer_A[8], m_buffer_A[9], m_buffer_A[10], m_buffer_A[11],
                         m_buffer_A[12], m_buffer_A[13], m_buffer_A[14], m_buffer_A[15], m_buffer_pk, m_buffer_pk,
                         m_buffer_Apk);
    }
    void run() { auto state = m_run.wait(); }
    void getMem() { m_fpga->copyFromFpga(m_buffer_Apk); }

   private:
    xrt::bo m_buffer_pk, m_buffer_Apk;
    vector<xrt::bo> m_buffer_A;
};

template <typename T>
class CGKernelUpdatePk : public XKernel {
   public:
    CGKernelUpdatePk(FPGA* p_fpga = nullptr) : XKernel(p_fpga) {}
    void setMem(host_buffer_t<T>& h_pk, host_buffer_t<T>& h_zk) {
        // Running the kernel
        m_buffer_pk = createDeviceBuffer(h_pk, m_kernel.group_id(0));
        m_buffer_zk = createDeviceBuffer(h_zk, m_kernel.group_id(2));

        m_fpga->copyToFpga(m_buffer_pk);
        m_fpga->copyToFpga(m_buffer_zk);

        m_run = m_kernel(m_buffer_pk, m_buffer_pk, m_buffer_zk);
    }
    void run() { auto state = m_run.wait(); }
    void getMem() { m_fpga->copyFromFpga(m_buffer_pk); }

   private:
    xrt::bo m_buffer_pk, m_buffer_zk;
};
template <typename T>
class CGKernelUpdateRkJacobi : public XKernel {
   public:
    CGKernelUpdateRkJacobi(FPGA* p_fpga = nullptr) : XKernel(p_fpga) {}
    void setMem(host_buffer_t<T>& h_rk, host_buffer_t<T>& h_zk, host_buffer_t<T>& h_jacobi, host_buffer_t<T>& h_Apk) {
        // Running the kernel
        m_buffer_rk = createDeviceBuffer(h_rk, m_kernel.group_id(0));
        m_buffer_zk = createDeviceBuffer(h_zk, m_kernel.group_id(2));
        m_buffer_jacobi = createDeviceBuffer(h_jacobi, m_kernel.group_id(3));
        m_buffer_Apk = createDeviceBuffer(h_Apk, m_kernel.group_id(4));

        m_fpga->copyToFpga(m_buffer_rk);
        m_fpga->copyToFpga(m_buffer_zk);
        m_fpga->copyToFpga(m_buffer_jacobi);
        m_fpga->copyToFpga(m_buffer_Apk);

        m_run = m_kernel(m_buffer_rk, m_buffer_rk, m_buffer_zk, m_buffer_jacobi, m_buffer_Apk);
    }

    void run() { auto state = m_run.wait(); }
    void getMem() {
        m_fpga->copyFromFpga(m_buffer_rk);
        m_fpga->copyFromFpga(m_buffer_zk);
    }

   private:
    xrt::bo m_buffer_Apk, m_buffer_rk, m_buffer_zk, m_buffer_jacobi;
};

template <typename T>
class CGKernelUpdateXk : public XKernel {
   public:
    CGKernelUpdateXk(FPGA* p_fpga = nullptr) : XKernel(p_fpga) {}
    void setMem(host_buffer_t<T>& h_xk, host_buffer_t<T>& h_pk) {
        // Running the kernel
        m_buffer_xk = createDeviceBuffer(h_xk, m_kernel.group_id(0));
        m_buffer_pk = createDeviceBuffer(h_pk, m_kernel.group_id(2));

        // Copy input data to device global memory
        m_fpga->copyToFpga(m_buffer_xk);
        m_fpga->copyToFpga(m_buffer_pk);
        // Setting Kernel Arguments
        m_run = m_kernel(m_buffer_xk, m_buffer_xk, m_buffer_pk);
    }
    void run() { auto state = m_run.wait(); }
    void getMem() { m_fpga->copyFromFpga(m_buffer_xk); }

   private:
    xrt::bo m_buffer_xk, m_buffer_pk;
};
