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

#ifndef XF_HPC_CG_CONTROL_HPP
#define XF_HPC_CG_CONTROL_HPP

#include "xf_blas.hpp"
#include "tasks.hpp"
#include "token.hpp"
#include "cgInstr.hpp"
#include "timer.hpp"
#include "ap_utils.h"

namespace xf {
namespace hpc {
namespace cg {

template <typename t_DataType,
          uint32_t t_ParEntries,
          uint32_t t_InstrBytes,
          uint32_t t_NumTasks,
          uint32_t t_TkWidth = 8>
void control(typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_instr,
             hls::stream<uint8_t>& p_signal,
             hls::stream<uint64_t>& p_clock,
             hls::stream<ap_uint<t_TkWidth> >& p_tokenIn,
             hls::stream<ap_uint<t_TkWidth> >& p_tokenOut) {
    uint32_t l_numFinished = 0;
    xf::hpc::cg::Task<t_DataType> l_tasks[t_NumTasks];

    xf::hpc::cg::Token<t_DataType> l_token;
    xf::hpc::StreamInstr<sizeof(l_token)> l_cs;
    xf::hpc::MemInstr<t_InstrBytes> l_memInstr;
    xf::hpc::cg::CGSolverInstr<t_DataType> l_cgSolverInstr;

    constexpr uint32_t l_numLine = t_InstrBytes / sizeof(t_DataType) / t_ParEntries;
    uint32_t l_instrIndex = t_NumTasks;

    p_signal.write(START);
    for (int i = 0; i < t_NumTasks; i++) {
        l_cgSolverInstr.load(p_instr + i * l_numLine, l_memInstr);
        l_tasks[i].setMaxIter(l_cgSolverInstr.getMaxIter());
        l_tasks[i].setTols(l_cgSolverInstr.getTols());

        l_token.setID(i);
        l_token.setVecSize(l_cgSolverInstr.getVecSize());
        l_token.setRZ(l_cgSolverInstr.getRZ());
        l_token.encode_write(p_tokenOut, l_cs);
    }

    while (l_numFinished < t_NumTasks) {
        l_token.read_decode(p_tokenIn, l_cs);
        p_signal.write(STAMP);
        ap_wait();
        uint64_t l_clock = p_clock.read();

        uint32_t l_id = l_token.getID();
        t_DataType l_res = l_token.getRes();
        l_tasks[l_id].setRes(l_res);

        l_cgSolverInstr.setID(l_id);
        l_cgSolverInstr.setIter(l_tasks[l_id].getIter());
        l_cgSolverInstr.setRes(l_tasks[l_id].getRes());
        l_cgSolverInstr.setRZ(l_token.getRZ());
        l_cgSolverInstr.setAlpha(l_token.getAlpha());
        l_cgSolverInstr.setBeta(l_token.getBeta());
        l_cgSolverInstr.setClock(l_clock);

        if ((!l_tasks[l_id].meetTol()) && l_tasks[l_id].increase()) {
            l_token.increase();
        } else {
            l_numFinished++;
            l_token.setExit();
        }
        l_token.encode_write(p_tokenOut, l_cs);
        l_cgSolverInstr.store(p_instr + l_instrIndex * l_numLine, l_memInstr);
        l_instrIndex++;
    }
    p_signal.write(STOP);
}
}
}
}
#endif
