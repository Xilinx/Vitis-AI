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

#ifndef XF_HPC_STREAM_INSTR_HPP
#define XF_HPC_STREAM_INSTR_HPP

#include <cstdint>
#include "memInstr.hpp"
#include "ap_int.h"
#include "hls_stream.h"

namespace xf {
namespace hpc {

template <unsigned int t_InstrBytes>
class StreamInstr : public MemInstr<t_InstrBytes> {
   public:
    void read(hls::stream<ap_uint<8> >& p_tkStr) {
        for (int i = 0; i < t_InstrBytes; i++) {
#pragma HLS PIPELINE
            this->m_Instr[i] = p_tkStr.read();
        }
    }

    void write(hls::stream<ap_uint<8> >& p_tkStr) {
        for (int i = 0; i < t_InstrBytes; i++)
#pragma HLS PIPELINE
            p_tkStr.write(this->m_Instr[i]);
    }

    void read(hls::stream<uint8_t>& p_tkStr) {
        for (int i = 0; i < t_InstrBytes; i++) {
#pragma HLS PIPELINE
            this->m_Instr[i] = p_tkStr.read();
        }
    }

    void write(hls::stream<uint8_t>& p_tkStr) {
        for (int i = 0; i < t_InstrBytes; i++)
#pragma HLS PIPELINE
            p_tkStr.write(this->m_Instr[i]);
    }
};
}
}
#endif
