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
 *  @brief xf_blas program (including instruction and data) memory manager
 *
 *  $DateTime: 2019/06/14$
 */

#ifndef BLAS_GEN_BIN_HPP
#define BLAS_GEN_BIN_HPP

#include <cstring>
#include "blas_program.hpp"

// const opcode defintion for printing
#define GBMV 14
#define SBMV 17
#define SPMV 18
#define TBMV 24
#define TPMV 26

using namespace std;
namespace xf {

namespace blas {

template <typename t_DataType,
          typename t_ResDataType,
          typename t_HPPandleType,
          unsigned int t_MemWidthBytes,
          unsigned int t_ParEntries,
          unsigned int t_InstrSizeBytes = 8,
          unsigned int t_PageSizeBytes = 4096,
          unsigned int t_MaxNumInstrs = 64,
          unsigned int t_InstrPageIdx = 0,
          unsigned int t_ParamPageIdx = 1,
          unsigned int t_StatsPageIdx = 2>
class GenBin {
   public:
    typedef typename Program<t_HPPandleType,
                             t_DataType,
                             t_ResDataType,
                             t_MemWidthBytes,
                             t_ParEntries,
                             t_InstrSizeBytes,
                             t_PageSizeBytes,
                             t_MaxNumInstrs,
                             t_InstrPageIdx,
                             t_ParamPageIdx,
                             t_StatsPageIdx>::ParamB1Type ParamB1Type;
    typedef typename Program<t_HPPandleType,
                             t_DataType,
                             t_ResDataType,
                             t_MemWidthBytes,
                             t_ParEntries,
                             t_InstrSizeBytes,
                             t_PageSizeBytes,
                             t_MaxNumInstrs,
                             t_InstrPageIdx,
                             t_ParamPageIdx,
                             t_StatsPageIdx>::ParamB2Type ParamB2Type;
    typedef typename ParamB2<t_DataType, t_ParEntries>::MatStoreType MatStoreType;

   public:
    static const size_t ParamB1Bytes = Program<t_HPPandleType,
                                               t_DataType,
                                               t_ResDataType,
                                               t_MemWidthBytes,
                                               t_ParEntries,
                                               t_InstrSizeBytes,
                                               t_PageSizeBytes,
                                               t_MaxNumInstrs,
                                               t_InstrPageIdx,
                                               t_ParamPageIdx,
                                               t_StatsPageIdx>::ParamB1Bytes;
    static const size_t ParamB2Bytes = Program<t_HPPandleType,
                                               t_DataType,
                                               t_ResDataType,
                                               t_MemWidthBytes,
                                               t_ParEntries,
                                               t_InstrSizeBytes,
                                               t_PageSizeBytes,
                                               t_MaxNumInstrs,
                                               t_InstrPageIdx,
                                               t_ParamPageIdx,
                                               t_StatsPageIdx>::ParamB2Bytes;

   public:
    GenBin() { assert((t_MemWidthBytes / sizeof(t_DataType)) % t_ParEntries == 0); }
    xfblasStatus_t addB1Instr(string p_opName,
                              uint32_t p_n,
                              t_DataType p_alpha,
                              void* p_x,
                              void* p_y,
                              void* p_xRes,
                              void* p_yRes,
                              t_ResDataType p_res) {
        uint32_t l_opCode32;
        xfblasStatus_t l_status = m_opFinder.getOpCode(p_opName, l_opCode32);
        uint16_t l_opClass, l_opCode;
        if ((l_status == XFBLAS_STATUS_SUCCESS) && (l_opCode32 <= B1_MaxOpCode)) { // BLAS L1 operations
            l_opClass = B1_OP_CLASS;
            l_opCode = l_opCode32;

            Instr l_instr;
            l_instr.m_opClass = l_opClass;
            l_instr.m_opCode = l_opCode;
            l_instr.m_paramOff = m_program.getCurrParamOff();

            ParamB1Type l_param;
            l_param.m_n = p_n;
            l_param.m_alpha = p_alpha;
            l_param.m_resScalar = p_res;
            l_param.m_xAddr = 0;
            l_param.m_yAddr = 0;
            l_param.m_xResAddr = 0;
            l_param.m_yResAddr = 0;

            l_status = regMem(p_n * sizeof(t_DataType), p_x, l_param.m_xAddr);
            l_status = regMem(p_n * sizeof(t_DataType), p_y, l_param.m_yAddr);
            l_status = regMem(p_n * sizeof(t_DataType), p_xRes, l_param.m_xResAddr);
            l_status = regMem(p_n * sizeof(t_DataType), p_yRes, l_param.m_yResAddr);

            uint8_t* l_instrVal = reinterpret_cast<uint8_t*>(&l_instr);
            uint8_t* l_paramVal = reinterpret_cast<uint8_t*>(&l_param);

            m_program.addInstr(l_instrVal, l_paramVal, ParamB1Bytes);
            return (l_status);
        } else {
            return (XFBLAS_STATUS_INVALID_OP);
        }
    }

    xfblasStatus_t addB2Instr(string p_opName,
                              uint32_t p_m,
                              uint32_t p_n,
                              uint32_t p_kl,
                              uint32_t p_ku,
                              t_DataType p_alpha,
                              t_DataType p_beta,
                              void* p_a,
                              void* p_x,
                              void* p_y,
                              void* p_aRes,
                              void* p_yRes) {
        uint32_t l_opCode32;
        xfblasStatus_t l_status = m_opFinder.getOpCode(p_opName, l_opCode32);
        uint16_t l_opClass, l_opCode;
        if ((l_status == XFBLAS_STATUS_SUCCESS) && (l_opCode32 > B1_MaxOpCode) &&
            (l_opCode32 <= B2_MaxOpCode)) { // BLAS L2 operations
            l_opClass = B2_OP_CLASS;
            l_opCode = l_opCode32;

            Instr l_instr;
            l_instr.m_opClass = l_opClass;
            l_instr.m_opCode = l_opCode;
            l_instr.m_paramOff = m_program.getCurrParamOff();

            ParamB2Type l_param;
            l_param.m_m = p_m;
            l_param.m_n = p_n;
            l_param.m_kl = p_kl;
            l_param.m_ku = p_ku;
            l_param.m_alpha = p_alpha;
            l_param.m_beta = p_beta;
            l_param.m_aAddr = 0;
            l_param.m_xAddr = 0;
            l_param.m_yAddr = 0;
            l_param.m_aResAddr = 0;
            l_param.m_yResAddr = 0;

            MatStoreType l_store;
            uint32_t l_rows = 0;
            uint32_t l_entries = 0;
            switch (l_opCode) {
                case SBMV:
                    if (p_kl == 0) {
                        l_store = MatStoreType::SBM_UP;
                        l_rows = p_ku + 1;
                    } else {
                        l_store = MatStoreType::SBM_LO;
                        l_rows = p_kl + 1;
                    }
                    break;
                case SPMV:
                    assert(p_m == p_n);
                    if (p_kl == 0) {
                        l_store = MatStoreType::PM_UP;
                        assert(p_ku == p_n);
                    } else {
                        l_store = MatStoreType::PM_LO;
                        assert(p_kl == p_n);
                    }
                    l_entries = ((p_n / t_ParEntries) + 1) * (p_n / t_ParEntries) * t_ParEntries * t_ParEntries / 2;
                    break;
                case TBMV:
                    if (p_kl == 0) {
                        l_store = MatStoreType::TBM_UP;
                        l_rows = p_ku + 1;
                    } else {
                        l_store = MatStoreType::TBM_LO;
                        l_rows = p_kl + 1;
                    }
                    break;
                case TPMV:
                    assert(p_m == p_n);
                    if (p_kl == 0) {
                        l_store = MatStoreType::PM_UP;
                        assert(p_ku == p_n);
                    } else {
                        l_store = MatStoreType::PM_LO;
                        assert(p_kl == p_n);
                    }
                    l_entries = ((p_n / t_ParEntries) + 1) * (p_n / t_ParEntries) * t_ParEntries * t_ParEntries / 2;
                    break;
                case GBMV:
                    l_store = MatStoreType::GBM;
                    l_rows = p_ku + p_kl + 1;
                    break;
                default:
                    l_store = MatStoreType::GEM;
                    break;
            }
            l_param.m_aStore = l_store;

            if ((l_store == MatStoreType::PM_UP) || (l_store == MatStoreType::PM_LO)) {
                l_status = regMem(l_entries * sizeof(t_DataType), p_a, l_param.m_aAddr);
                l_status = regMem(l_entries * sizeof(t_DataType), p_aRes, l_param.m_aResAddr);
            } else {
                l_status = regMem(l_rows * p_n * sizeof(t_DataType), p_a, l_param.m_aAddr);
                l_status = regMem(l_rows * p_n * sizeof(t_DataType), p_aRes, l_param.m_aResAddr);
            }
            l_status = regMem(p_n * sizeof(t_DataType), p_x, l_param.m_xAddr);
            l_status = regMem(p_m * sizeof(t_DataType), p_y, l_param.m_yAddr);
            l_status = regMem(p_m * sizeof(t_DataType), p_yRes, l_param.m_yResAddr);

            uint8_t* l_instrVal = reinterpret_cast<uint8_t*>(&l_instr);
            uint8_t* l_paramVal = reinterpret_cast<uint8_t*>(&l_param);

            m_program.addInstr(l_instrVal, l_paramVal, ParamB2Bytes);
            return (l_status);
        } else {
            return (XFBLAS_STATUS_INVALID_OP);
        }
    }

    xfblasStatus_t write2BinFile(string p_fileName) {
        xfblasStatus_t l_status = m_program.write2BinFile(p_fileName);
        return (l_status);
    }

    xfblasStatus_t readFromBinFile(string p_fileName) {
        xfblasStatus_t l_status = m_program.readFromBinFile(p_fileName);
        return (l_status);
    }
    void decodeB1Instr(const Instr& p_instr,
                       uint32_t& p_n,
                       t_DataType& p_alpha,
                       t_DataType*& p_x,
                       t_DataType*& p_y,
                       t_DataType*& p_xRes,
                       t_DataType*& p_yRes,
                       t_ResDataType& p_resScalar) {
        m_program.decodeB1Instr(p_instr, p_n, p_alpha, p_x, p_y, p_xRes, p_yRes, p_resScalar);
    }
    void decodeB2Instr(const Instr& p_instr,
                       uint32_t& p_m,
                       uint32_t& p_n,
                       uint32_t& p_kl,
                       uint32_t& p_ku,
                       t_DataType& p_alpha,
                       t_DataType& p_beta,
                       t_DataType*& p_a,
                       t_DataType*& p_x,
                       t_DataType*& p_y,
                       t_DataType*& p_aRes,
                       t_DataType*& p_yRes) {
        m_program.decodeB2Instr(p_instr, p_m, p_n, p_kl, p_ku, p_alpha, p_beta, p_a, p_x, p_y, p_aRes, p_yRes);
    }

    xfblasStatus_t readInstrs(string p_fileName, vector<Instr>& p_instrs) {
        xfblasStatus_t l_status = m_program.readInstrsFromBinFile(p_fileName, p_instrs);
        return (l_status);
    }

    void printProgram() { cout << m_program; }

   private:
    xfblasStatus_t regMem(size_t p_bytes, void* p_memPointer, uint64_t& p_addr) {
        xfblasStatus_t l_status = XFBLAS_STATUS_SUCCESS;
        if (p_memPointer != nullptr) {
            l_status = m_program.regDatMem(p_memPointer, p_memPointer, p_bytes);
            if (l_status != XFBLAS_STATUS_SUCCESS) {
                return (l_status);
            }
            unsigned long long l_bufSize;
            p_addr = reinterpret_cast<uint64_t>(m_program.getDatMem(p_memPointer, l_bufSize));
        }
        return (l_status);
    }

    Program<t_HPPandleType,
            t_DataType,
            t_ResDataType,
            t_MemWidthBytes,
            t_ParEntries,
            t_InstrSizeBytes,
            t_PageSizeBytes,
            t_MaxNumInstrs,
            t_InstrPageIdx,
            t_ParamPageIdx,
            t_StatsPageIdx>
        m_program;
    FindOpCode m_opFinder;
};

} // end namespace blas

} // end namespace xf
#endif
