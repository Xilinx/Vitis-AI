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
 *  @file instr.hpp
 *  @brief HPC_CG Level 1 template function implementation.
 *
 */

#ifndef XF_HPC_CG_ISA_HPP
#define XF_HPC_CG_ISA_HPP

#include "assert.h"
#include "memInstr.hpp"

namespace xf {
namespace hpc {
namespace cg {

/** Class CGSolverInstr defines a set of parameters used in between host and the kernel of the CG solver for linear
 * system Ax=b
 * @param m_ID the id of the task
 * @param m_VecSize vector size
 * @param m_Iter the current iteration
 * @param m_MaxIter the maximum iteration for the solver
 * @param m_Tols the compond tolerence, which is tol * tol * norm(b) * norm(b)
 * @param m_Res the residual, square of the norm(r)
 */
template <typename t_DataType>
class CGSolverInstr {
   public:
    CGSolverInstr() {
        m_ID = 0;
        m_VecSize = 0;
        m_Iter = 0;
        m_MaxIter = 0;
        m_Tols = 0;
        m_Res = 0;
        m_Alpha = 0;
        m_Beta = 0;
        m_Clock = 0;
    }

    void setClock(uint64_t p_Clock) { m_Clock = p_Clock; }
    uint64_t getClock() const { return m_Clock; }

    void setID(uint32_t p_ID) { m_ID = p_ID; }
    uint32_t getID() const { return m_ID; }

    void setVecSize(uint32_t p_VecSize) { m_VecSize = p_VecSize; }
    uint32_t getVecSize() const { return m_VecSize; }

    void setMaxIter(uint32_t p_MaxIter) { m_MaxIter = p_MaxIter; }
    uint32_t getMaxIter() const { return m_MaxIter; }

    void setRZ(t_DataType p_RZ) { m_RZ = p_RZ; }
    t_DataType getRZ() const { return m_RZ; }

    void setRes(t_DataType p_Res) { m_Res = p_Res; }
    t_DataType getRes() const { return m_Res; }

    void setBeta(t_DataType p_Beta) { m_Beta = p_Beta; }
    t_DataType getBeta() const { return m_Beta; }

    void setAlpha(t_DataType p_Alpha) { m_Alpha = p_Alpha; }
    t_DataType getAlpha() const { return m_Alpha; }

    void setTols(t_DataType p_Tols) { m_Tols = p_Tols; }
    t_DataType getTols() const { return m_Tols; }

    uint32_t getIter() const { return m_Iter; }
    void setIter(uint32_t p_iter) { m_Iter = p_iter; }

    template <typename T, typename t_MemInstr>
    void store(T* p_mem, t_MemInstr& p_memInstr) {
        encode(p_memInstr);
        p_memInstr.template store<T>(p_mem);
    }

    template <typename T, typename t_MemInstr>
    void load(T* p_mem, t_MemInstr& p_memInstr) {
        p_memInstr.template load<T>(p_mem);
        decode(p_memInstr);
    }

#ifndef __SYNTHESIS__
    friend std::ostream& operator<<(std::ostream& os, CGSolverInstr& cgInstr) {
        os << "m_ID: " << cgInstr.m_ID << ", ";
        os << "m_Iter: " << cgInstr.m_Iter << ", ";
        os << "m_Res: " << cgInstr.m_Res << ", ";
        os << "m_RZ: " << cgInstr.m_RZ << ", ";
        os << "m_Alpha: " << cgInstr.m_Alpha << ", ";
        os << "m_Beta: " << cgInstr.m_Beta << ", ";
        os << "m_VecSize: " << cgInstr.m_VecSize << ", ";
        os << "m_Tols: " << cgInstr.m_Tols << ", ";
        os << "m_MaxIter: " << cgInstr.m_MaxIter << ", ";
        os << "m_Clock: " << cgInstr.m_Clock << ", ";
        return os;
    }
#endif

   protected:
    template <typename t_MemInstr>
    void encode(t_MemInstr& p_memInstr) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        unsigned int l_loc = 0;
        p_memInstr.template encode<uint32_t>(l_loc, m_ID);
        p_memInstr.template encode<uint32_t>(l_loc, m_Iter);
        p_memInstr.template encode<uint32_t>(l_loc, m_VecSize);
        p_memInstr.template encode<uint32_t>(l_loc, m_MaxIter);
        p_memInstr.template encode<t_DataType>(l_loc, m_Tols);
        p_memInstr.template encode<t_DataType>(l_loc, m_Res);
        p_memInstr.template encode<t_DataType>(l_loc, m_RZ);
        p_memInstr.template encode<t_DataType>(l_loc, m_Alpha);
        p_memInstr.template encode<t_DataType>(l_loc, m_Beta);
        p_memInstr.template encode<uint64_t>(l_loc, m_Clock);
    }
    template <typename t_MemInstr>
    void decode(t_MemInstr& p_memInstr) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        unsigned int l_loc = 0;
        p_memInstr.template decode<uint32_t>(l_loc, m_ID);
        p_memInstr.template decode<uint32_t>(l_loc, m_Iter);
        p_memInstr.template decode<uint32_t>(l_loc, m_VecSize);
        p_memInstr.template decode<uint32_t>(l_loc, m_MaxIter);
        p_memInstr.template decode<t_DataType>(l_loc, m_Tols);
        p_memInstr.template decode<t_DataType>(l_loc, m_Res);
        p_memInstr.template decode<t_DataType>(l_loc, m_RZ);
        p_memInstr.template decode<t_DataType>(l_loc, m_Alpha);
        p_memInstr.template decode<t_DataType>(l_loc, m_Beta);
        p_memInstr.template decode<uint64_t>(l_loc, m_Clock);
    }

   private:
    uint32_t m_ID, m_VecSize, m_Iter, m_MaxIter;
    t_DataType m_Tols, m_Res, m_RZ;
    t_DataType m_Alpha, m_Beta;
    uint64_t m_Clock;
};

} // namespace
}
}

#endif
