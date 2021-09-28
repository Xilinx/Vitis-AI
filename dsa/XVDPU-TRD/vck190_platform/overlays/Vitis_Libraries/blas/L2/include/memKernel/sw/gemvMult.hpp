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

#ifndef XF_BLAS_GEMVMULT_HPP
#define XF_BLAS_GEMVMULT_HPP
#include "sw/fpga.hpp"

template <typename t_DataType, unsigned int t_NumChannels>
class GemvKernel : public Kernel {
   public:
    GemvKernel(uint32_t p_m, uint32_t p_n, FPGA* fpga) : Kernel(fpga) {
        m_m = p_m;
        m_n = p_n;
    }

    void setMem(host_buffer_t<t_DataType>& h_A, host_buffer_t<t_DataType>& h_x, host_buffer_t<t_DataType>& h_r) {
        cl_int err;
        vector<cl::Memory> l_buffers;
        vector<host_buffer_t<t_DataType> > h_As(t_NumChannels);
        setMemA(h_A, h_As);

        for (auto x : m_buffer_A) l_buffers.push_back(x);

        m_buffer_x = createDeviceBuffer(CL_MEM_READ_ONLY, h_x);
        m_buffer_r = createDeviceBuffer(CL_MEM_WRITE_ONLY, h_r);
        l_buffers.push_back(m_buffer_x);
        l_buffers.push_back(m_buffer_r);

        // Setting Kernel Arguments
        uint32_t n_arg = 0;
        OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_m));
        OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_n));

        for (unsigned int i = 0; i < t_NumChannels; i++) {
            OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_buffer_A[i]));
        }
        OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_buffer_x));
        OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_buffer_r));

        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(l_buffers, 0)); /* 0 means from host*/
        finish();
    }

    void run() {
        auto start = chrono::high_resolution_clock::now();
        enqueueTask();
        finish();
        auto coldend = chrono::high_resolution_clock::now();
        enqueueTask();
        finish();
        auto finish = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed0 = coldend - start;
        chrono::duration<double> elapsed1 = finish - coldend;
        double t_sec = elapsed1.count();
        cout << "Software-measured cold start execution time " << elapsed0.count() << "s." << endl;
        cout << "Software-measured hot start execution time " << elapsed1.count() << "s." << endl;
        double tTheory = m_n * m_m / BLAS_parEntries / BLAS_numChannels * 3e-9;
        cout << "Software-measured HW efficiency " << tTheory / t_sec * 100 << "%." << endl;
    }

    void getMem() {
        cl_int err;
        OCL_CHECK(err,
                  err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer_r}, CL_MIGRATE_MEM_OBJECT_HOST));
        finish();
    }

   private:
    void setMemA(host_buffer_t<t_DataType>& h_A, vector<host_buffer_t<t_DataType> >& h_As) {
        if (t_NumChannels == 1) {
            m_buffer_A.push_back(createDeviceBuffer(CL_MEM_READ_ONLY, h_A));
        } else {
            for (unsigned int i = 0; i < t_NumChannels; i++) h_As[i].resize(h_A.size() / t_NumChannels);

            for (unsigned int i = 0; i < m_m; i++)
                copy(h_A.begin() + i * m_n, h_A.begin() + (i + 1) * m_n,
                     h_As[i % t_NumChannels].begin() + (i / t_NumChannels) * m_n);
            for (unsigned int i = 0; i < t_NumChannels; i++) {
                m_buffer_A.push_back(createDeviceBuffer(CL_MEM_READ_ONLY, h_As[i]));
            }
        }
    }
    uint32_t m_m, m_n;
    vector<cl::Buffer> m_buffer_A;
    cl::Buffer m_buffer_x;
    cl::Buffer m_buffer_r;
};

#endif
