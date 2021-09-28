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
 *  @brief instruction format for BLAS L1 functions
 *
 *  $DateTime: 2019/06/13 $
 */

#ifndef BLAS_INSTR_HPP
#define BLAS_INSTR_HPP

#include <iostream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <vector>
#include "L3/include/sw/utility/utility.hpp"

using namespace std;

#define B1_MaxOpCode 13
#define B2_MaxOpCode 36
#define B3_MaxOpCode 51
#define NULL_OP 0
#define B1_OP_CLASS 0
#define B2_OP_CLASS 1
#define B3_OP_CLASS 2
#define OUTPUT_WIDTH 7
#define ENTRIES_PER_LINE 16

namespace xf {

namespace blas {
namespace {
template <typename t_DataType>
void printVec(ostream& os, const vector<t_DataType>& p_data, uint32_t p_n) {
    for (unsigned int i = 0; i < p_n; ++i) {
        if ((i % ENTRIES_PER_LINE) == 0) {
            os << "\n";
        }
        os << setw(OUTPUT_WIDTH) << p_data[i] << "  ";
    }
    os << "\n";
}
template <typename t_DataType>
void printMat(ostream& os, const vector<t_DataType>& p_data, uint32_t p_m, uint32_t p_n) {
    for (unsigned int i = 0; i < p_m; ++i) {
        for (unsigned int j = 0; j < p_n; ++j) {
            if ((j % ENTRIES_PER_LINE) == 0) {
                os << "\n";
            }
            os << setw(OUTPUT_WIDTH) << p_data[i * p_n + j] << "  ";
        }
        os << "\n"
           << "\n";
    }
}

template <typename t_DataType, unsigned int t_ParEntries>
void printPackedMatLo(ostream& os, const vector<t_DataType>& p_data, uint32_t p_n) {
    unsigned int l_blocks = p_n / t_ParEntries;
    unsigned int l_off = 0;
    for (unsigned int b = 0; b < l_blocks; ++b) {
        for (unsigned int i = 0; i < t_ParEntries; ++i) {
            for (unsigned int j = 0; j < (b + 1) * t_ParEntries; ++j) {
                os << setw(OUTPUT_WIDTH) << p_data[l_off + j] << "  ";
            }
            l_off += (b + 1) * t_ParEntries;
            os << "\n";
        }
        os << "\n";
    }
}

template <typename t_DataType, unsigned int t_ParEntries>
void printPackedMatUp(ostream& os, const vector<t_DataType>& p_data, uint32_t p_n) {
    unsigned int l_blocks = p_n / t_ParEntries;
    unsigned int l_off = 0;
    for (unsigned int b = l_blocks; b > 0; --b) {
        for (unsigned int i = 0; i < t_ParEntries; ++i) {
            for (unsigned int j = 0; j < b * t_ParEntries; ++j) {
                os << setw(OUTPUT_WIDTH) << p_data[l_off + j] << "  ";
            }
            l_off += b * t_ParEntries;
            os << "\n";
        }
        os << "\n";
    }
}

template <typename t_DataType>
void getVecData(uint64_t p_addr, uint32_t p_n, vector<t_DataType>& p_data) {
    p_data.clear();
    if (p_addr == 0) {
        return;
    }
    size_t l_dataBytes = p_n * sizeof(t_DataType);
    p_data.resize(p_n);
    memcpy((char*)&(p_data[0]), reinterpret_cast<char*>(p_addr), l_dataBytes);
    return;
}
template <typename t_DataType>
void getMatrixData(uint64_t p_addr, uint32_t p_entries, vector<t_DataType>& p_data) {
    p_data.clear();
    if (p_addr == 0) {
        return;
    }
    size_t l_dataBytes = p_entries * sizeof(t_DataType);
    p_data.resize(p_entries);
    memcpy((char*)&(p_data[0]), reinterpret_cast<char*>(p_addr), l_dataBytes);
    return;
}
template <typename t_DataType>
void showVec(ostream& os, uint32_t p_n, uint64_t p_addr, string p_str) {
    vector<t_DataType> l_data;
    getVecData<t_DataType>(p_addr, p_n, l_data);
    if (l_data.size() != 0) {
        os << p_str << "\n";
        printVec<t_DataType>(os, l_data, p_n);
    }
}
template <typename t_DataType>
void showMatrix(ostream& os, uint32_t p_m, uint32_t p_n, uint64_t p_addr, string p_str) {
    vector<t_DataType> l_data;
    getMatrixData(p_addr, p_m * p_n, l_data);
    if (l_data.size() != 0) {
        os << p_str << "\n";
        printMat<t_DataType>(os, l_data, p_m, p_n);
    }
}

template <typename t_DataType, unsigned int t_ParEntries>
void showPackedMatrixLo(ostream& os, uint32_t p_m, uint32_t p_n, uint64_t p_addr, string p_str) {
    vector<t_DataType> l_data;
    assert(p_m == p_n);
    uint32_t l_blocks = p_n / t_ParEntries;
    uint32_t l_numEntries = (l_blocks + 1) * l_blocks * t_ParEntries * t_ParEntries / 2;
    getMatrixData(p_addr, l_numEntries, l_data);
    if (l_data.size() != 0) {
        os << p_str << "\n";
        printPackedMatLo<t_DataType, t_ParEntries>(os, l_data, p_n);
    }
}

template <typename t_DataType, unsigned int t_ParEntries>
void showPackedMatrixUp(ostream& os, uint32_t p_m, uint32_t p_n, uint64_t p_addr, string p_str) {
    vector<t_DataType> l_data;
    assert(p_m == p_n);
    uint32_t l_blocks = p_n / t_ParEntries;
    uint32_t l_numEntries = (l_blocks + 1) * l_blocks * t_ParEntries * t_ParEntries / 2;
    getMatrixData(p_addr, l_numEntries, l_data);
    if (l_data.size() != 0) {
        os << p_str << "\n";
        printPackedMatUp<t_DataType, t_ParEntries>(os, l_data, p_n);
    }
}

} // namespace
class FindOpCode {
   public:
    FindOpCode() {
        m_opMap = {{"null_op", 0},
                   {"amax", 1},
                   {"amin", 2},
                   {"asum", 3},
                   {"axpy", 4},
                   {"copy", 5},
                   {"dot", 6},
                   {"nrm2", 7},
                   {"rot", 8},
                   {"rotg", 9},
                   {"rotm", 10},
                   {"rotmg", 11},
                   {"scal", 12},
                   {"swap", 13},
                   {"gbmv", 14},
                   {"gemv", 15},
                   {"ger", 16},
                   {"sbmv", 17},
                   {"spmv", 18},
                   {"spr", 19},
                   {"spr2", 20},
                   {"symv", 21},
                   {"syr", 22},
                   {"syr2", 23},
                   {"tbmv", 24},
                   {"tbsv", 25},
                   {"tpmv", 26},
                   {"tpsv", 27},
                   {"trmv", 28},
                   {"trsv", 29},
                   {"hemv", 30},
                   {"hbmv", 31},
                   {"hpmv", 32},
                   {"her", 33},
                   {"her2", 34},
                   {"hpr", 35},
                   {"hpr2", 36},
                   {"gemm", 37},
                   {"gemm3m", 38},
                   {"gemmBatched", 39},
                   {"gemmStridedBatched", 40},
                   {"symm", 41},
                   {"syrj", 42},
                   {"syr2k", 43},
                   {"syrkx", 44},
                   {"trmm", 45},
                   {"trsm", 46},
                   {"trsmBatched", 47},
                   {"hemm", 48},
                   {"herk", 49},
                   {"her2k", 50},
                   {"herkx", 51}};
    }
    xfblasStatus_t getOpCode(const string& p_opName, uint32_t& p_opCode) {
        if (m_opMap.find(p_opName) == m_opMap.end()) {
            return XFBLAS_STATUS_INVALID_VALUE;
        } else {
            p_opCode = m_opMap[p_opName];
            return XFBLAS_STATUS_SUCCESS;
        }
    }
    xfblasStatus_t getOpName(uint32_t p_opCode, string& p_opName) {
        xfblasStatus_t l_status = XFBLAS_STATUS_INVALID_VALUE;
        p_opName = "no_op";
        for (auto it = m_opMap.begin(); it != m_opMap.end(); ++it) {
            if (it->second == p_opCode) {
                p_opName = it->first;
                l_status = XFBLAS_STATUS_SUCCESS;
                break;
            }
        }
        return (l_status);
    }

   private:
    unordered_map<string, uint32_t> m_opMap;
};

// all offsets are defined as byte offsets
template <typename t_DataType, typename t_ResDataType>
class ParamB1 {
   public:
    ParamB1() {}

   public:
    void print(ostream& os) {
        os << "n=" << m_n << " alpha=" << setw(OUTPUT_WIDTH) << m_alpha << " resGolden=" << setw(OUTPUT_WIDTH)
           << m_resScalar << "\n";

        showVec<t_DataType>(os, m_n, m_xAddr, "x:");
        showVec<t_DataType>(os, m_n, m_yAddr, "y:");
        showVec<t_DataType>(os, m_n, m_xResAddr, "xRes:");
        showVec<t_DataType>(os, m_n, m_yResAddr, "xRes:");
    }

   public:
    uint32_t m_n;
    t_DataType m_alpha;
    uint64_t m_xAddr;
    uint64_t m_yAddr;
    uint64_t m_xResAddr;
    uint64_t m_yResAddr;
    t_ResDataType m_resScalar;
};

template <typename T1, typename T2>
ostream& operator<<(ostream& os, ParamB1<T1, T2>& p_val) {
    p_val.print(os);
    return (os);
}

template <typename t_DataType, unsigned int t_ParEntries>
class ParamB2 {
   public:
    typedef enum { GEM, GBM, PM_LO, PM_UP, SBM_LO, SBM_UP, TBM_LO, TBM_UP } MatStoreType;

   public:
    ParamB2() { m_aStore = GEM; }

   public:
    void getVecData(uint64_t p_addr, uint32_t p_n, vector<t_DataType>& p_data) {
        p_data.clear();
        if (p_addr == 0) {
            return;
        }
        size_t l_dataBytes = p_n * sizeof(t_DataType);
        p_data.resize(p_n);
        memcpy((char*)&(p_data[0]), reinterpret_cast<char*>(p_addr), l_dataBytes);
        return;
    }
    void print(ostream& os) {
        os << "m=" << m_m << " n=" << m_n << " kl=" << m_kl << " ku=" << m_ku << " alpha=" << setw(OUTPUT_WIDTH)
           << m_alpha << " beta=" << setw(OUTPUT_WIDTH) << m_beta << "\n";
        uint32_t l_rows = 0;
        switch (m_aStore) {
            case GBM:
                l_rows = m_kl + m_ku + 1;
                break;
            case SBM_LO:
                l_rows = m_kl + 1;
                break;
            case SBM_UP:
                l_rows = m_ku + 1;
                break;
            case TBM_LO:
                l_rows = m_kl + 1;
                break;
            case TBM_UP:
                l_rows = m_ku + 1;
                break;
            default:
                l_rows = m_m;
                break;
        }
        if (m_aStore == PM_LO) {
            showPackedMatrixLo<t_DataType, t_ParEntries>(os, l_rows, m_n, m_aAddr, "A:");
        } else if (m_aStore == PM_UP) {
            showPackedMatrixUp<t_DataType, t_ParEntries>(os, l_rows, m_n, m_aAddr, "A:");
        } else {
            showMatrix<t_DataType>(os, l_rows, m_n, m_aAddr, "A:");
        }

        showVec<t_DataType>(os, m_n, m_xAddr, "x:");
        showVec<t_DataType>(os, m_m, m_yAddr, "y:");
        showMatrix<t_DataType>(os, l_rows, m_n, m_aResAddr, "ARes:");
        showVec<t_DataType>(os, m_m, m_yResAddr, "yRes:");
    }

   public:
    MatStoreType m_aStore;
    uint32_t m_m;
    uint32_t m_n;
    uint32_t m_kl;
    uint32_t m_ku;
    t_DataType m_alpha;
    t_DataType m_beta;
    uint64_t m_aAddr;
    uint64_t m_xAddr;
    uint64_t m_yAddr;
    uint64_t m_aResAddr;
    uint64_t m_yResAddr;
};

template <typename T1, unsigned int T2>
ostream& operator<<(ostream& os, ParamB2<T1, T2>& p_val) {
    p_val.print(os);
    return (os);
}
class Instr {
   public:
    Instr() {}

   public:
    void print(ostream& os) {
        FindOpCode l_opFinder;
        string l_opName;
        xfblasStatus_t l_status = l_opFinder.getOpName(m_opCode, l_opName);
        assert(l_status == XFBLAS_STATUS_SUCCESS);
        os << "Operation: " << l_opName << "\n";
    }

   public:
    uint16_t m_opClass;
    uint16_t m_opCode;
    int32_t m_paramOff;
};

ostream& operator<<(ostream& os, Instr& p_instr) {
    p_instr.print(os);
    return (os);
}

} // end namespace blas

} // end namespace xf

#endif
