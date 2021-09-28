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
#include "fpga.hpp"

class CGKernelControl : public Kernel {
   public:
    CGKernelControl(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<uint8_t>& h_instr) {
        cl_int err;
        // Running the kernel
        m_buffer_instr = createDeviceBuffer(CL_MEM_READ_WRITE, h_instr);
        // Setting Kernel Arguments
        OCL_CHECK(err, err = m_kernel.setArg(0, m_buffer_instr));
        // Copy input data to device global memory
        OCL_CHECK(err,
                  err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer_instr}, 0)); /* 0 means from host*/
    }

    void getMem() {
        cl_int err;
        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer_instr},
                                                                                CL_MIGRATE_MEM_OBJECT_HOST));
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer_instr;
};

template <typename T, int t_ParEntries, int t_NumChannels = 1>
class CGKernelGemv : public Kernel {
   public:
    CGKernelGemv(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<T>& h_A, host_buffer_t<T>& h_pk, host_buffer_t<T>& h_Apk) {
        cl_int err;
        vector<cl::Memory> l_buffers;
        vector<host_buffer_t<T> > h_As(t_NumChannels);
        uint32_t l_vecSize = h_pk.size();
        // Running the kernel

        if (t_NumChannels == 1) {
            m_buffer_A.push_back(createDeviceBuffer(CL_MEM_READ_ONLY, h_A));
            l_buffers.push_back(m_buffer_A[0]);
        } else {
            for (int i = 0; i < t_NumChannels; i++) h_As[i].resize(h_A.size() / t_NumChannels);

            for (int i = 0; i < l_vecSize; i++)
                copy(h_A.begin() + i * l_vecSize, h_A.begin() + (i + 1) * l_vecSize,
                     h_As[i % t_NumChannels].begin() + (i / t_NumChannels) * l_vecSize);

            for (int i = 0; i < t_NumChannels; i++) {
                m_buffer_A.push_back(createDeviceBuffer(CL_MEM_READ_ONLY, h_As[i]));
                l_buffers.push_back(m_buffer_A[i]);
            }
        }
        m_buffer_pk = createDeviceBuffer(CL_MEM_READ_WRITE, h_pk);
        m_buffer_Apk = createDeviceBuffer(CL_MEM_READ_WRITE, h_Apk);

        l_buffers.push_back(m_buffer_pk);
        l_buffers.push_back(m_buffer_Apk);

        // Setting Kernel Arguments
        uint32_t n_arg = 0;
        for (int i = 0; i < t_NumChannels; i++) {
            OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_buffer_A[i]));
        }
        OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_buffer_Apk));

        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(l_buffers, 0)); /* 0 means from host*/
        m_fpga->finish();
    }
    void getMem() {
        cl_int err;
        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err,
                  err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer_Apk}, CL_MIGRATE_MEM_OBJECT_HOST));
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer_pk, m_buffer_Apk;
    vector<cl::Buffer> m_buffer_A;
};

template <typename T>
class CGKernelStoreApk : public Kernel {
   public:
    CGKernelStoreApk(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<T>& h_pk, host_buffer_t<T>& h_Apk) {
        cl_int err;
        // Running the kernel
        m_buffer_pk = createDeviceBuffer(CL_MEM_READ_WRITE, h_pk);
        m_buffer_Apk = createDeviceBuffer(CL_MEM_READ_WRITE, h_Apk);

        // Setting Kernel Arguments
        OCL_CHECK(err, err = m_kernel.setArg(1, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(2, m_buffer_Apk));

        // Copy input data to device global memory
        OCL_CHECK(err,
                  err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer_pk}, 0)); /* 0 means from host*/
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer_pk, m_buffer_Apk;
};

template <typename T>
class CGKernelUpdatePk : public Kernel {
   public:
    CGKernelUpdatePk(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<T>& h_pk, host_buffer_t<T>& h_zk) {
        cl_int err;
        // Running the kernel
        m_buffer_pk = createDeviceBuffer(CL_MEM_READ_WRITE, h_pk);
        m_buffer_zk = createDeviceBuffer(CL_MEM_READ_WRITE, h_zk);

        // Setting Kernel Arguments
        int n_arg = 0;
        OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(n_arg++, m_buffer_zk));

        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer_zk, m_buffer_pk},
                                                                                0)); /* 0 means from host*/
        m_fpga->finish();
    }
    void getMem() {
        cl_int err;
        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err,
                  err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer_pk}, CL_MIGRATE_MEM_OBJECT_HOST));
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer_pk, m_buffer_zk;
};
template <typename T>
class CGKernelUpdateRkJacobi : public Kernel {
   public:
    CGKernelUpdateRkJacobi(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<T>& h_rk, host_buffer_t<T>& h_zk, host_buffer_t<T>& h_jacobi, host_buffer_t<T>& h_Apk) {
        cl_int err;
        vector<cl::Memory> l_buffers;
        // Running the kernel
        m_buffer_rk = createDeviceBuffer(CL_MEM_READ_WRITE, h_rk);
        m_buffer_zk = createDeviceBuffer(CL_MEM_READ_WRITE, h_zk);
        m_buffer_jacobi = createDeviceBuffer(CL_MEM_READ_ONLY, h_jacobi);
        m_buffer_Apk = createDeviceBuffer(CL_MEM_READ_WRITE, h_Apk);

        l_buffers.push_back(m_buffer_rk);
        l_buffers.push_back(m_buffer_zk);
        l_buffers.push_back(m_buffer_Apk);
        l_buffers.push_back(m_buffer_jacobi);

        // Setting Kernel Arguments
        int l_index = 0;
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_rk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_rk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_zk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_jacobi));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_Apk));

        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(l_buffers, 0)); /* 0 means from host*/
        m_fpga->finish();
    }
    void getMem() {
        cl_int err;
        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer_rk, m_buffer_zk},
                                                                                CL_MIGRATE_MEM_OBJECT_HOST));
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer_Apk, m_buffer_rk, m_buffer_zk, m_buffer_jacobi;
};
template <typename T>
class CGKernelUpdateRk : public Kernel {
   public:
    CGKernelUpdateRk(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<T>& h_rk, host_buffer_t<T>& h_Apk) {
        cl_int err;
        vector<cl::Memory> l_buffers;
        // Running the kernel
        m_buffer_rk = createDeviceBuffer(CL_MEM_READ_WRITE, h_rk);
        m_buffer_Apk = createDeviceBuffer(CL_MEM_READ_WRITE, h_Apk);

        l_buffers.push_back(m_buffer_rk);
        l_buffers.push_back(m_buffer_Apk);

        // Setting Kernel Arguments
        int l_index = 0;
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_rk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_rk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_Apk));

        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(l_buffers, 0)); /* 0 means from host*/
        m_fpga->finish();
    }
    void getMem() {
        cl_int err;
        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err,
                  err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer_rk}, CL_MIGRATE_MEM_OBJECT_HOST));
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer_Apk, m_buffer_rk;
};

template <typename T>
class CGKernelUpdateXk : public Kernel {
   public:
    CGKernelUpdateXk(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<T>& h_xk, host_buffer_t<T>& h_pk) {
        cl_int err;
        vector<cl::Memory> l_buffers;
        // Running the kernel
        m_buffer_xk = createDeviceBuffer(CL_MEM_READ_WRITE, h_xk);
        m_buffer_pk = createDeviceBuffer(CL_MEM_READ_WRITE, h_pk);

        l_buffers.push_back(m_buffer_xk);
        l_buffers.push_back(m_buffer_pk);

        // Setting Kernel Arguments
        int l_index = 0;
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_xk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_xk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_pk));

        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(l_buffers, 0)); /* 0 means from host*/
        m_fpga->finish();
    }
    void getMem() {
        cl_int err;
        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err,
                  err = m_fpga->getCommandQueue().enqueueMigrateMemObjects({m_buffer_xk}, CL_MIGRATE_MEM_OBJECT_HOST));
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer_xk, m_buffer_pk;
};

template <typename T>
class CGKernelGemvSeq : public Kernel {
   public:
    CGKernelGemvSeq(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<uint8_t>& h_instr,
                host_buffer_t<T>& h_A,
                host_buffer_t<T>& h_Apk,
                host_buffer_t<T>& h_xk,
                host_buffer_t<T>& h_rk,
                host_buffer_t<T>& h_pk) {
        cl_int err;
        vector<cl::Memory> l_buffers;
        // Running the kernel
        m_buffer_instr = createDeviceBuffer(CL_MEM_READ_WRITE, h_instr);
        m_buffer_xk = createDeviceBuffer(CL_MEM_READ_WRITE, h_xk);
        m_buffer_rk = createDeviceBuffer(CL_MEM_READ_WRITE, h_rk);
        m_buffer_Apk = createDeviceBuffer(CL_MEM_READ_WRITE, h_Apk);
        m_buffer_A = createDeviceBuffer(CL_MEM_READ_ONLY, h_A);
        m_buffer_pk = createDeviceBuffer(CL_MEM_READ_WRITE, h_pk);

        l_buffers.push_back(m_buffer_xk);
        l_buffers.push_back(m_buffer_rk);
        l_buffers.push_back(m_buffer_A);
        l_buffers.push_back(m_buffer_instr);
        l_buffers.push_back(m_buffer_Apk);
        l_buffers.push_back(m_buffer_pk);

        // Setting Kernel Arguments
        int l_index = 0;
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_instr));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_instr));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_A));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_Apk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_Apk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_xk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_xk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_rk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_rk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_rk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_pk));

        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(l_buffers, 0)); /* 0 means from host*/
        m_fpga->finish();
    }
    void getMem() {
        cl_int err;
        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(
                           {m_buffer_xk, m_buffer_rk, m_buffer_pk, m_buffer_Apk, m_buffer_instr},
                           CL_MIGRATE_MEM_OBJECT_HOST));
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer_instr, m_buffer_A, m_buffer_Apk, m_buffer_xk, m_buffer_pk, m_buffer_rk;
};
template <typename T>
class CGKernelGemvJacobiSeq : public Kernel {
   public:
    CGKernelGemvJacobiSeq(FPGA* p_fpga = nullptr) : Kernel(p_fpga) {}
    void setMem(host_buffer_t<uint8_t>& h_instr,
                host_buffer_t<T>& h_A,
                host_buffer_t<T>& h_Apk,
                host_buffer_t<T>& h_xk,
                host_buffer_t<T>& h_rk,
                host_buffer_t<T>& h_zk,
                host_buffer_t<T>& h_jacobi,
                host_buffer_t<T>& h_pk) {
        cl_int err;
        vector<cl::Memory> l_buffers;
        // Running the kernel
        m_buffer_instr = createDeviceBuffer(CL_MEM_READ_WRITE, h_instr);
        m_buffer_xk = createDeviceBuffer(CL_MEM_READ_WRITE, h_xk);
        m_buffer_rk = createDeviceBuffer(CL_MEM_READ_WRITE, h_rk);
        m_buffer_zk = createDeviceBuffer(CL_MEM_READ_WRITE, h_zk);
        m_buffer_jacobi = createDeviceBuffer(CL_MEM_READ_ONLY, h_jacobi);
        m_buffer_Apk = createDeviceBuffer(CL_MEM_READ_WRITE, h_Apk);
        m_buffer_A = createDeviceBuffer(CL_MEM_READ_ONLY, h_A);
        m_buffer_pk = createDeviceBuffer(CL_MEM_READ_WRITE, h_pk);

        l_buffers.push_back(m_buffer_instr);
        l_buffers.push_back(m_buffer_xk);
        l_buffers.push_back(m_buffer_rk);
        l_buffers.push_back(m_buffer_zk);
        l_buffers.push_back(m_buffer_A);
        l_buffers.push_back(m_buffer_jacobi);
        l_buffers.push_back(m_buffer_Apk);
        l_buffers.push_back(m_buffer_pk);

        // Setting Kernel Arguments
        int l_index = 0;
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_instr));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_instr));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_A));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_Apk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_Apk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_xk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_xk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_rk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_rk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_rk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_zk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_jacobi));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_pk));
        OCL_CHECK(err, err = m_kernel.setArg(l_index++, m_buffer_pk));

        // Copy input data to device global memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(l_buffers, 0)); /* 0 means from host*/
        m_fpga->finish();
    }
    void getMem() {
        cl_int err;
        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(
                           {m_buffer_xk, m_buffer_rk, m_buffer_pk, m_buffer_Apk, m_buffer_instr},
                           CL_MIGRATE_MEM_OBJECT_HOST));
        m_fpga->finish();
    }

   private:
    cl::Buffer m_buffer_instr, m_buffer_A, m_buffer_Apk, m_buffer_zk, m_buffer_jacobi, m_buffer_xk, m_buffer_pk,
        m_buffer_rk;
};
