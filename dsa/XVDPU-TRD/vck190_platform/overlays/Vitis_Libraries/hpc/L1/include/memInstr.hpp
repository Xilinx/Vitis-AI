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
 *  @file codes.hpp
 *  @brief HPC_CG Level 1 template function implementation.
 *
 */

#ifndef XF_HPC_MEMINSTR_HPP
#define XF_HPC_MEMINSTR_HPP

#include <cstdint>
#ifndef __SYNTHESIS__
#include <iostream>
#include <iomanip>
#endif

namespace xf {

namespace hpc {

namespace {

template <typename T>
void v2b(const T p_val, uint8_t* p_ch) {
    constexpr uint32_t t_Mult = sizeof(T);
    for (unsigned int i = 0; i < t_Mult; ++i) {
        p_ch[i] = (p_val >> (i * 8)) & 0xff;
    }
}
template <typename T>
void b2v(T& p_val, const uint8_t* p_ch) {
    constexpr uint32_t t_Mult = sizeof(T);
    p_val = 0;
    for (unsigned int i = 0; i < t_Mult; ++i) {
        T l_val = p_ch[i];
        p_val += l_val << (i * 8);
    }
}

template <>
void v2b<bool>(const bool p_val, uint8_t* p_ch) {
    p_ch[0] = p_val ? 1 : 0;
}
template <>
void b2v<bool>(bool& p_val, const uint8_t* p_ch) {
    uint32_t l_val = p_ch[0];
    p_val = l_val == 0 ? false : true;
}

template <>
void v2b<float>(const float p_val, uint8_t* p_ch) {
    constexpr uint32_t t_Mult = sizeof(float);
    uint32_t l_val = *reinterpret_cast<const uint32_t*>(&p_val);
    for (unsigned int i = 0; i < t_Mult; ++i) {
        p_ch[i] = l_val >> (i * 8);
    }
}
template <>
void b2v<float>(float& p_val, const uint8_t* p_ch) {
    constexpr uint32_t t_Mult = sizeof(float);
    uint32_t l_val = 0;
    for (unsigned int i = 0; i < t_Mult; ++i) {
        l_val = l_val << 8;
        l_val += p_ch[t_Mult - 1 - i];
    }
    p_val = *reinterpret_cast<float*>(&l_val);
}

template <>
void v2b<double>(const double p_val, uint8_t* p_ch) {
    constexpr uint64_t t_Mult = sizeof(double);
    uint64_t l_val = *reinterpret_cast<const uint64_t*>(&p_val);
    for (unsigned int i = 0; i < t_Mult; ++i) {
        p_ch[i] = l_val >> (i * 8);
    }
}
template <>
void b2v<double>(double& p_val, const uint8_t* p_ch) {
    uint64_t l_val = 0;
    constexpr uint64_t t_Mult = sizeof(double);
    for (unsigned int i = 0; i < t_Mult; ++i) {
        l_val = l_val << 8;
        l_val += p_ch[t_Mult - 1 - i];
    }
    p_val = *reinterpret_cast<double*>(&l_val);
}
}

template <unsigned int t_InstrBytes>
class MemInstr {
   public:
    MemInstr() {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = m_Instr complete dim = 1
#endif
        init();
    }

    void init() {
        for (int i = 0; i < t_InstrBytes; i++)
#pragma HLS PIPELINE
            m_Instr[i] = 0;
    }
    uint8_t& operator[](int i) { return (m_Instr[i]); }

    void store(uint8_t* p_memPtr) const { memcpy(p_memPtr, m_Instr, t_InstrBytes); }
    void load(const uint8_t* p_memPtr) { memcpy(m_Instr, p_memPtr, t_InstrBytes); }

    template <typename T>
    void store(T* p_memPtr) const {
        constexpr int t_MemBytes = sizeof(T);
        for (unsigned int i = 0; i < t_InstrBytes / t_MemBytes; i++) {
#pragma HLS PIPELINE
            T l_val;
            b2v<T>(l_val, m_Instr + i * t_MemBytes);
            p_memPtr[i] = l_val;
        }
    }

    template <typename T>
    void load(const T* p_memPtr) {
        constexpr int t_MemBytes = sizeof(T);
        for (unsigned int i = 0; i < t_InstrBytes / t_MemBytes; i++) {
#pragma HLS PIPELINE
            T l_val = p_memPtr[i];
            v2b<T>(l_val, m_Instr + i * t_MemBytes);
        }
    }

    template <typename T>
    void encode(unsigned int& p_loc, T p_val) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#else
        assert(p_loc < t_InstrBytes);
#endif
        v2b<T>(p_val, m_Instr + p_loc);
        p_loc += sizeof(T);
    }

    template <typename T>
    void decode(unsigned int& p_loc, T& p_val) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#else
        assert(p_loc < t_InstrBytes);
#endif
        b2v<T>(p_val, m_Instr + p_loc);
        p_loc += sizeof(T);
    }

#ifndef __SYNTHESIS__
    friend std::ostream& operator<<(std::ostream& os, MemInstr& memInstr) {
        static constexpr int t_NumInstr = 4;
        os << "m_Instr: ";
        for (int i = t_InstrBytes - 1; i >= 0; i--) {
            os << std::hex << std::setfill('0') << std::setw(2) << static_cast<uint16_t>(memInstr.m_Instr[i])
               << std::dec;
            if (i % t_NumInstr == 0) os << ' ';
        }
        return os;
    }
#endif

   protected:
    uint8_t m_Instr[t_InstrBytes];
};
}
}
#endif
