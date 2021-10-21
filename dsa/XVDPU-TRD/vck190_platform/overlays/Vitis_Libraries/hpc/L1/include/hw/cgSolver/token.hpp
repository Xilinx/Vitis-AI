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

#ifndef XF_HPC_CG_TOKEN_HPP
#define XF_HPC_CG_TOKEN_HPP

#include <cstdint>
#include "ap_int.h"
#include "hls_stream.h"
#include "streamInstr.hpp"

namespace xf {
namespace hpc {
namespace cg {

template <typename t_DataType>
class Token {
   public:
    Token() {
        m_Exit = 0;
        m_ID = 0;
        m_VecSize = 0;
        m_rz = 0;
        m_Res = 0;
        m_Alpha = 0;
        m_Beta = 0;
        m_Iter = 0;
    }
    Token(uint32_t p_id, uint32_t p_vecSize) : m_ID(p_id), m_VecSize(p_vecSize) {
        m_Exit = 0;
        m_rz = 0;
        m_Res = 0;
        m_Alpha = 0;
        m_Beta = 0;
        m_Iter = 0;
    }

#ifndef __SYNTHESIS__
    friend std::ostream& operator<<(std::ostream& os, Token& token) {
        os << "m_ID: " << token.m_ID << '\t';
        os << "m_VecSize: " << token.m_VecSize << '\t';
        os << "m_Alpha: " << token.m_Alpha << '\t';
        os << "m_Beta: " << token.m_Beta << '\t';
        os << "m_rz: " << token.m_rz << '\t';
        os << "m_Res: " << token.m_Res << '\t';
        os << "m_Iter: " << token.m_Iter << '\t';
        os << "m_Exit: " << token.m_Exit << '\t';
        return os;
    }
#endif

    uint32_t getID() const { return m_ID; }
    void setID(uint32_t p_ID) { m_ID = p_ID; }

    uint32_t getVecSize() const { return m_VecSize; }
    void setVecSize(uint32_t p_vecSize) { m_VecSize = p_vecSize; }

    t_DataType getAlpha() const { return m_Alpha; }
    void setAlpha(t_DataType p_alpha) { m_Alpha = p_alpha; }

    t_DataType getBeta() const { return m_Beta; }
    void setBeta(t_DataType p_beta) { m_Beta = p_beta; }

    t_DataType getRes() const { return m_Res; }
    void setRes(t_DataType p_res) { m_Res = p_res; }

    t_DataType getRZ() const { return m_rz; }
    void setRZ(t_DataType p_rz) { m_rz = p_rz; }

    void setExit(bool p_exit = true) { m_Exit = p_exit ? 1 : 0; }
    bool getExit() const { return m_Exit == 1 ? true : false; }

    void increase() { m_Iter++; }
    uint32_t getIter() const { return m_Iter; }

    template <typename t_StreamInstr, int t_StreamWidth>
    void encode_write(hls::stream<ap_uint<t_StreamWidth> >& p_tkStr, t_StreamInstr& p_streamInstr) {
        encode(p_streamInstr);
        p_streamInstr.write(p_tkStr);
    }

    template <typename t_StreamInstr, int t_StreamWidth>
    void read_decode(hls::stream<ap_uint<t_StreamWidth> >& p_tkStr, t_StreamInstr& p_streamInstr) {
        p_streamInstr.read(p_tkStr);
        decode(p_streamInstr);
    }

   private:
    template <typename t_StreamInstr>
    void decode(t_StreamInstr& p_streamInstr) {
        uint32_t l_ind = 0;
        p_streamInstr.template decode<uint32_t>(l_ind, m_ID);
        p_streamInstr.template decode<uint32_t>(l_ind, m_VecSize);
        p_streamInstr.template decode<uint32_t>(l_ind, m_Iter);
        p_streamInstr.template decode<t_DataType>(l_ind, m_Alpha);
        p_streamInstr.template decode<t_DataType>(l_ind, m_Beta);
        p_streamInstr.template decode<t_DataType>(l_ind, m_rz);
        p_streamInstr.template decode<t_DataType>(l_ind, m_Res);
        p_streamInstr.template decode<uint32_t>(l_ind, m_Exit);
    }

    template <typename t_StreamInstr>
    void encode(t_StreamInstr& p_streamInstr) {
        uint32_t l_ind = 0;
        p_streamInstr.template encode<uint32_t>(l_ind, m_ID);
        p_streamInstr.template encode<uint32_t>(l_ind, m_VecSize);
        p_streamInstr.template encode<uint32_t>(l_ind, m_Iter);
        p_streamInstr.template encode<t_DataType>(l_ind, m_Alpha);
        p_streamInstr.template encode<t_DataType>(l_ind, m_Beta);
        p_streamInstr.template encode<t_DataType>(l_ind, m_rz);
        p_streamInstr.template encode<t_DataType>(l_ind, m_Res);
        p_streamInstr.template encode<uint32_t>(l_ind, m_Exit);
    }

    t_DataType m_Alpha, m_Beta, m_Res, m_rz;
    uint32_t m_VecSize;
    uint32_t m_ID, m_Iter;
    uint32_t m_Exit;
};
}
}
}
#endif
