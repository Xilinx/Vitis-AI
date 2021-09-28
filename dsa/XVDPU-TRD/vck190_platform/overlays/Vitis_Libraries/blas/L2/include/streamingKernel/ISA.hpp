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
 *  @file ISA.hpp
 *  @brief BLAS Level 2 template function implementation.
 *
 */

#ifndef XF_BLAS_ISA_HPP
#define XF_BLAS_ISA_HPP

#include "assert.h"

namespace xf {

namespace blas {

template <unsigned int t_InstrBytes>
class MemInstr {
   public:
    MemInstr() {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = m_instr complete dim = 1
#endif
    }
    uint8_t& operator[](int i) { return (m_instr[i]); }
    void storeOpCode(uint16_t p_opCode) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        m_instr[0] = p_opCode;
        m_instr[1] = p_opCode >> 8;
    }
    void loadOpCode(uint16_t& p_opCode) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        p_opCode = (m_instr[1] << 8) | m_instr[0];
    }
    void store(const unsigned int p_loc, bool p_val) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        m_instr[p_loc] = p_val;
    }
    void load(const unsigned int p_loc, bool& p_val) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        p_val = m_instr[p_loc];
    }
    void store(const unsigned int p_loc, uint32_t p_val) {
        for (unsigned int i = 0; i < 4; ++i) {
            m_instr[i + p_loc] = p_val >> (i * 8);
        }
    }
    void load(const unsigned int p_loc, uint32_t& p_val) {
        p_val = (m_instr[p_loc + 3] << 24) | (m_instr[p_loc + 2] << 16) | (m_instr[p_loc + 1] << 8) | m_instr[p_loc];
    }
    void storeMem(uint8_t* p_memPtr) { memcpy(p_memPtr, m_instr, t_InstrBytes); }
    void loadMem(uint8_t* p_memPtr) { memcpy(m_instr, p_memPtr, t_InstrBytes); }

#ifdef __AP_INT_H__
    template <int t_MemBytes>
    void storeMem(ap_uint<t_MemBytes * 8>* p_memPtr) {
        for (unsigned int i = 0; i < t_InstrBytes / t_MemBytes; i++)
#pragma HLS PIPELINE
            for (unsigned int j = 0; j < t_MemBytes; ++j) {
                p_memPtr[i].range(8 * (j + 1) - 1, j * 8) = m_instr[i * t_MemBytes + j];
            }
    }
    template <int t_MemBytes>
    void loadMem(ap_uint<t_MemBytes * 8>* p_memPtr) {
        for (unsigned int i = 0; i < t_InstrBytes / t_MemBytes; i++)
#pragma HLS PIPELINE
            for (unsigned int j = 0; j < t_MemBytes; ++j) {
                m_instr[i * t_MemBytes + j] = p_memPtr[i].range(8 * (j + 1) - 1, j * 8);
            }
    }
#endif

   private:
    uint8_t m_instr[t_InstrBytes];
};

//////////////////////////// Instrruction Architecture /////////////////////////
typedef enum {
    OpControl = 0x00,
    OpGemv = 0x01,
    OpGemm = 0x02,
    OpTransp = 0x03,
    OpResult = 0x04,
    OpFail = 0x05,
    OpGemmLdSt = 0x06
} OpCodeType;

template <unsigned int t_InstrBytes>
class ResInstr {
   public:
    uint32_t m_startTime, m_endTime;

   public:
    ResInstr() {}
    ResInstr(uint32_t p_startTime, uint32_t p_endTime) : m_startTime(p_startTime), m_endTime(p_endTime) {
#ifndef __SYNTHESIS__
        assert(m_startTime <= m_endTime);
#endif
    }
    uint32_t getDuration() { return m_endTime - m_startTime; }

    void store(MemInstr<t_InstrBytes>& p_instr) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        uint16_t l_opCode = OpCodeType::OpResult;
        p_instr.storeOpCode(l_opCode);
        p_instr.store(2, m_startTime);
        p_instr.store(6, m_endTime);
    }
    void load(MemInstr<t_InstrBytes>& p_instr) {
#pragma HLS INLINE
        p_instr.load(2, m_startTime);
        p_instr.load(6, m_endTime);
    }
};

//////////////////////////// CONTROL ////////////////////////////
template <unsigned int t_InstrBytes>
class ControlInstr {
   public:
    bool m_isLastOp, m_noop;

   public:
    ControlInstr() {}
    ControlInstr(bool p_IsLastOp, bool p_noop) : m_isLastOp(p_IsLastOp), m_noop(p_noop) {
        assert(m_isLastOp || m_noop);
    }
    bool getIsLastOp() { return m_isLastOp; }
    bool getNoop() { return m_noop; }
    void store(MemInstr<t_InstrBytes>& p_instr) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        uint16_t l_opCode = OpCodeType::OpControl;
        p_instr.storeOpCode(l_opCode);
        p_instr.store(2, m_isLastOp);
        p_instr.store(3, m_noop);
    }
    void load(MemInstr<t_InstrBytes>& p_instr) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        p_instr.load(2, m_isLastOp);
        p_instr.load(3, m_noop);
    }
};

template <unsigned int t_InstrBytes>
class GemmInstr {
   public:
    uint32_t m_aOffset, m_bOffset, m_cOffset, m_xOffset, m_m, m_k, m_n;

   public:
    GemmInstr() {}
    GemmInstr(uint32_t p_aOffset,
              uint32_t p_bOffset,
              uint32_t p_xOffset,
              uint32_t p_cOffset,
              uint32_t p_m,
              uint32_t p_k,
              uint32_t p_n)
        : m_aOffset(p_aOffset),
          m_bOffset(p_bOffset),
          m_cOffset(p_cOffset),
          m_xOffset(p_xOffset),
          m_m(p_m),
          m_k(p_k),
          m_n(p_n) {}
    void init(uint32_t p_aOffset,
              uint32_t p_bOffset,
              uint32_t p_xOffset,
              uint32_t p_cOffset,
              uint32_t p_m,
              uint32_t p_k,
              uint32_t p_n) {
        m_aOffset = p_aOffset;
        m_bOffset = p_bOffset;
        m_xOffset = p_xOffset;
        m_cOffset = p_cOffset;
        m_m = p_m;
        m_k = p_k;
        m_n = p_n;
    }
    void store(MemInstr<t_InstrBytes>& p_instr) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        uint16_t l_opCode = OpCodeType::OpGemm;
        p_instr.storeOpCode(l_opCode);
        p_instr.store(2, m_aOffset);
        p_instr.store(6, m_bOffset);
        p_instr.store(10, m_xOffset);
        p_instr.store(14, m_cOffset);
        p_instr.store(18, m_m);
        p_instr.store(22, m_k);
        p_instr.store(26, m_n);
    }
    void load(MemInstr<t_InstrBytes>& p_instr) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        p_instr.load(2, m_aOffset);
        p_instr.load(6, m_bOffset);
        p_instr.load(10, m_xOffset);
        p_instr.load(14, m_cOffset);
        p_instr.load(18, m_m);
        p_instr.load(22, m_k);
        p_instr.load(26, m_n);
    }
};

template <unsigned int t_InstrBytes>
class GemmLdStInstr {
   public:
    uint32_t m_aOffset, m_bOffset, m_xOffset, m_aWrOffset, m_bWrOffset, m_xWrOffset, m_m, m_k, m_n;

   public:
    GemmLdStInstr() {}
    GemmLdStInstr(uint32_t p_aOffset,
                  uint32_t p_bOffset,
                  uint32_t p_xOffset,
                  uint32_t p_aWrOffset,
                  uint32_t p_bWrOffset,
                  uint32_t p_xWrOffset,
                  uint32_t p_m,
                  uint32_t p_k,
                  uint32_t p_n)
        : m_aOffset(p_aOffset),
          m_bOffset(p_bOffset),
          m_xOffset(p_xOffset),
          m_aWrOffset(p_aWrOffset),
          m_bWrOffset(p_bWrOffset),
          m_xWrOffset(p_xWrOffset),
          m_m(p_m),
          m_k(p_k),
          m_n(p_n) {}
    void init(uint32_t p_aOffset,
              uint32_t p_bOffset,
              uint32_t p_xOffset,
              uint32_t p_aWrOffset,
              uint32_t p_bWrOffset,
              uint32_t p_xWrOffset,
              uint32_t p_m,
              uint32_t p_k,
              uint32_t p_n) {
        m_aOffset = p_aOffset;
        m_bOffset = p_bOffset;
        m_xOffset = p_xOffset;
        m_aWrOffset = p_aWrOffset;
        m_bWrOffset = p_bWrOffset;
        m_xWrOffset = p_xWrOffset;
        m_m = p_m;
        m_k = p_k;
        m_n = p_n;
    }
    void store(MemInstr<t_InstrBytes>& p_instr) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        uint16_t l_opCode = OpCodeType::OpGemmLdSt;
        p_instr.storeOpCode(l_opCode);
        p_instr.store(2, m_aOffset);
        p_instr.store(6, m_bOffset);
        p_instr.store(10, m_xOffset);
        p_instr.store(14, m_aWrOffset);
        p_instr.store(18, m_bWrOffset);
        p_instr.store(22, m_xWrOffset);
        p_instr.store(26, m_m);
        p_instr.store(30, m_k);
        p_instr.store(34, m_n);
    }
    void load(MemInstr<t_InstrBytes>& p_instr) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        p_instr.load(2, m_aOffset);
        p_instr.load(6, m_bOffset);
        p_instr.load(10, m_xOffset);
        p_instr.load(14, m_aWrOffset);
        p_instr.load(18, m_bWrOffset);
        p_instr.load(22, m_xWrOffset);
        p_instr.load(26, m_m);
        p_instr.load(30, m_k);
        p_instr.load(34, m_n);
    }
};

////////////////////////////TRANSP////////////////////////////
template <unsigned int t_InstrBytes>
class TranspInstr {
   public:
    uint32_t m_aOffset, m_bOffset, m_m, m_n;

   public:
    TranspInstr() {}
    TranspInstr(uint32_t p_aOffset, uint32_t p_bOffset, uint32_t p_m, uint32_t p_n)
        : m_aOffset(p_aOffset), m_bOffset(p_bOffset), m_m(p_m), m_n(p_n) {}

    void init(uint32_t p_aOffset, uint32_t p_bOffset, uint32_t p_m, uint32_t p_n) {
        m_aOffset = p_aOffset;
        m_bOffset = p_bOffset;
        m_m = p_m;
        m_n = p_n;
    }
    void store(MemInstr<t_InstrBytes>& p_instr) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        uint16_t l_opCode = OpCodeType::OpTransp;
        p_instr.storeOpCode(l_opCode);
        p_instr.store(2, m_aOffset);
        p_instr.store(6, m_bOffset);
        p_instr.store(10, m_m);
        p_instr.store(14, m_n);
    }
    void load(MemInstr<t_InstrBytes>& p_instr) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
        p_instr.load(2, m_aOffset);
        p_instr.load(6, m_bOffset);
        p_instr.load(10, m_m);
        p_instr.load(14, m_n);
    }
};

} // namespace
}

#endif
