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

#include "cgSolverKernel.hpp"
#include "utils.hpp"
#include "binFiles.hpp"

using namespace std;

template <typename t_DataType,
          unsigned int t_InstrBytes,
          unsigned int t_ParEntries,
          unsigned int t_NumChannels,
          typename t_MemInstr,
          typename t_CGSolverInstr>
class CgSolverGemv {
   public:
    CgSolverGemv(FPGA* fpga, unsigned int p_maxIter, t_DataType p_tol) {
        m_fpga = fpga;
        m_maxIter = p_maxIter;
        m_tol = p_tol;
        m_instrSize = t_InstrBytes * (1 + p_maxIter);
        h_instr.resize(m_instrSize);
        m_kernelControl.fpga(m_fpga);
        m_kernelControl.getCU("krnl_control");
        m_kernelGemv.fpga(m_fpga);
        m_kernelGemv.getCU("krnl_gemv");
        m_kernelUpdatePk.fpga(m_fpga);
        m_kernelUpdatePk.getCU("krnl_update_pk");
        m_kernelUpdateRkJacobi.fpga(m_fpga);
        m_kernelUpdateRkJacobi.getCU("krnl_update_rk_jacobi");
        m_kernelUpdateXk.fpga(m_fpga);
        m_kernelUpdateXk.getCU("krnl_update_xk");
    }

    void setA(const string filePath, unsigned int p_size) {
        m_matrixSize = p_size * p_size;
        m_vecSize = p_size;

        h_A.resize(m_matrixSize);
        h_jacobi.resize(m_vecSize);

        readBin(filePath, h_A.size() * sizeof(t_DataType), h_A);
        for (unsigned int i = 0; i < m_vecSize; i++) {
            h_jacobi[i] = 1.0 / h_A[i * m_vecSize + i];
        }
    }

    void setB(const string filePath) {
        h_b.resize(m_vecSize);
        h_pk.resize(m_vecSize);
        h_Apk.resize(m_vecSize);
        h_xk.resize(m_vecSize);
        h_rk.resize(m_vecSize);
        h_zk.resize(m_vecSize);

        readBin(filePath, h_b.size() * sizeof(t_DataType), h_b);

        t_DataType l_dot = 0, l_rz = 0;

        for (unsigned int i = 0; i < m_vecSize; i++) {
            h_xk[i] = 0;
            h_Apk[i] = 0;
            h_rk[i] = h_b[i];
            h_zk[i] = h_jacobi[i] * h_rk[i];
            l_dot += h_b[i] * h_b[i];
            l_rz += h_rk[i] * h_zk[i];
            h_pk[i] = h_zk[i];
        }

        m_cgInstr.setMaxIter(m_maxIter);
        m_cgInstr.setTols(l_dot * m_tol * m_tol);
        m_cgInstr.setRes(l_dot);
        m_cgInstr.setRZ(l_rz);
        m_cgInstr.setVecSize(m_vecSize);
        m_cgInstr.store(h_instr.data(), m_memInstr);

        m_kernelControl.setMem(h_instr);
        m_kernelGemv.setMem(h_A, h_pk, h_Apk);
        m_kernelUpdatePk.setMem(h_pk, h_zk);
        m_kernelUpdateRkJacobi.setMem(h_rk, h_zk, h_jacobi, h_Apk);
        m_kernelUpdateXk.setMem(h_xk, h_pk);

        m_kernels.push_back(m_kernelControl);
        m_kernels.push_back(m_kernelGemv);
        m_kernels.push_back(m_kernelUpdatePk);
        m_kernels.push_back(m_kernelUpdateXk);
        m_kernels.push_back(m_kernelUpdateRkJacobi);
    }

    void solve(int& lastIter, uint64_t& finalClock) {
        lastIter = 0;
        finalClock = 0;
        Kernel::run(m_kernels);
        m_kernelControl.getMem();
        m_kernelUpdateRkJacobi.getMem();
        m_kernelUpdateXk.getMem();
        for (unsigned int i = 0; i < m_maxIter; i++) {
            lastIter = i;
            m_cgInstr.load(h_instr.data() + (i + 1) * t_InstrBytes, m_memInstr);
            if (m_cgInstr.getMaxIter() == 0) {
                break;
            }
            finalClock = m_cgInstr.getClock();
        }
    }

    int verify(const string filePath) {
        host_buffer_t<CG_dataType> h_x(m_vecSize);
        readBin(filePath, h_x.size() * sizeof(t_DataType), h_x);
        int err = 0;
        compare(h_x.size(), h_x.data(), h_xk.data(), err);
        return err;
    }

   private:
    FPGA* m_fpga;
    vector<Kernel> m_kernels;

    unsigned int m_maxIter, m_instrSize;
    t_DataType m_tol;
    unsigned int m_matrixSize, m_vecSize;
    t_MemInstr m_memInstr;
    t_CGSolverInstr m_cgInstr;
    host_buffer_t<uint8_t> h_instr;
    host_buffer_t<t_DataType> h_A, h_x, h_b;
    host_buffer_t<t_DataType> h_pk, h_Apk, h_xk, h_rk, h_zk, h_jacobi;
    CGKernelControl m_kernelControl;
    CGKernelGemv<t_DataType, t_ParEntries, t_NumChannels> m_kernelGemv;
    CGKernelUpdatePk<t_DataType> m_kernelUpdatePk;
    CGKernelUpdateRkJacobi<t_DataType> m_kernelUpdateRkJacobi;
    CGKernelUpdateXk<t_DataType> m_kernelUpdateXk;
};
