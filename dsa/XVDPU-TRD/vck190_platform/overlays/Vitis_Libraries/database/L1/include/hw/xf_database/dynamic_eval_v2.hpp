/*
 * Copyright 2020 Xilinx, Inc.
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

/**
 * @file dynamic_eval_v2.hpp
 * @brief This file contains run-time-configurable expression evaluation
 * primitive.
 *
 * This file is part of Vitis Database Library.
 */
#ifndef XF_DATABASE_DYNAMIC_EVAL_V2_HPP
#define XF_DATABASE_DYNAMIC_EVAL_V2_HPP

#if !defined(__SYNTHESIS__)
#include <iostream>
#endif

#include "xf_database/utils.hpp"

#include <cstdint>
#include <ap_int.h>
#include <hls_stream.h>

namespace xf {
namespace database {

enum {
    DYN_EVAL_NCOL = 4,
    DYN_EVAL_NSTEP = 7,

    DYN_EVAL_INDEX_BITS = (3UL),
    DYN_EVAL_INDEX_VAR_MASK = (0x7UL),
    DYN_EVAL_INDEX_IMM_MASK = (0xfUL),
    DYN_EVAL_INDEX_TMP_MASK = (0x1fUL),

    DYN_EVAL_IS_VAR_BIT = (1UL << DYN_EVAL_INDEX_BITS),
    DYN_EVAL_IS_IMM_BIT = (1UL << (DYN_EVAL_INDEX_BITS + 1)),
    DYN_EVAL_IS_TMP_BIT = (1UL << (DYN_EVAL_INDEX_BITS + 2)),

    DYN_EVAL_REG_V0 = (0UL | DYN_EVAL_IS_VAR_BIT),
    DYN_EVAL_REG_V1 = (1UL | DYN_EVAL_IS_VAR_BIT),
    DYN_EVAL_REG_V2 = (2UL | DYN_EVAL_IS_VAR_BIT),
    DYN_EVAL_REG_V3 = (3UL | DYN_EVAL_IS_VAR_BIT),

    DYN_EVAL_REG_I0 = (0UL | DYN_EVAL_IS_IMM_BIT),
    DYN_EVAL_REG_I1 = (1UL | DYN_EVAL_IS_IMM_BIT),
    DYN_EVAL_REG_I2 = (2UL | DYN_EVAL_IS_IMM_BIT),
    DYN_EVAL_REG_I3 = (3UL | DYN_EVAL_IS_IMM_BIT),

    DYN_EVAL_REG_T0 = (0UL | DYN_EVAL_IS_TMP_BIT),
    DYN_EVAL_REG_T1 = (1UL | DYN_EVAL_IS_TMP_BIT),
    DYN_EVAL_REG_T2 = (2UL | DYN_EVAL_IS_TMP_BIT),
    DYN_EVAL_REG_T3 = (3UL | DYN_EVAL_IS_TMP_BIT),
    DYN_EVAL_REG_T4 = (4UL | DYN_EVAL_IS_TMP_BIT),
    DYN_EVAL_REG_T5 = (5UL | DYN_EVAL_IS_TMP_BIT),
    DYN_EVAL_REG_T6 = (6UL | DYN_EVAL_IS_TMP_BIT),

    DYN_EVAL_OP_NOP = (0UL),
    DYN_EVAL_OP_ADD = (1UL),
    DYN_EVAL_OP_SUB = (2UL),
    DYN_EVAL_OP_MUL = (3UL),
    DYN_EVAL_OP_DIV = (4UL),
};

namespace details {

#ifndef __SYNTHESIS__
inline const char* dynEvalOpName(unsigned op) {
    const char* ret = "";
    switch (op) {
        case DYN_EVAL_OP_NOP:
            ret = "NOP";
            break;
        case DYN_EVAL_OP_ADD:
            ret = "ADD";
            break;
        case DYN_EVAL_OP_SUB:
            ret = "SUB";
            break;
        case DYN_EVAL_OP_MUL:
            ret = "MUL";
            break;
        case DYN_EVAL_OP_DIV:
            ret = "DIV";
            break;
        default:
            ret = "UNKNOWN";
    }
    return ret;
}
#endif

template <typename T>
void eval_tmp(T var[DYN_EVAL_NCOL],
              T imm[DYN_EVAL_NCOL], //
              T tmpi[DYN_EVAL_NSTEP],
              T tmpo[DYN_EVAL_NSTEP], //
              ap_uint<16> inst,
              int i) {
#pragma HLS inline
    T a, b, t;
    uint8_t rh = (uint8_t)(inst & 0x3fU);
    uint8_t lh = (uint8_t)((inst >> 6) & 0x3fU);
    uint8_t op = (uint8_t)((inst >> 12) & 0x0fU);
    if (lh & DYN_EVAL_IS_VAR_BIT) {
        a = var[lh & DYN_EVAL_INDEX_VAR_MASK];
    } else if (lh & DYN_EVAL_IS_IMM_BIT) {
        a = imm[lh & DYN_EVAL_INDEX_IMM_MASK];
    } else if (lh & DYN_EVAL_IS_TMP_BIT) {
        uint8_t ti = lh & DYN_EVAL_INDEX_TMP_MASK;
        XF_DATABASE_ASSERT(ti < i && "Can only use result of previous steps");
        a = tmpi[ti];
    }
    if (rh & DYN_EVAL_IS_VAR_BIT) {
        b = var[rh & DYN_EVAL_INDEX_VAR_MASK];
    } else if (rh & DYN_EVAL_IS_IMM_BIT) {
        b = imm[rh & DYN_EVAL_INDEX_IMM_MASK];
    } else if (rh & DYN_EVAL_IS_TMP_BIT) {
        uint8_t ti = rh & DYN_EVAL_INDEX_TMP_MASK;
        XF_DATABASE_ASSERT(ti < i && "Can only use result of previous steps");
        b = tmpi[ti];
    }
#if 0
    std::cerr << "op = " << dynEvalOpName(op) << std::endl;
#endif
    switch (op) {
        case (DYN_EVAL_OP_NOP):
            t = a;
            break;
        case (DYN_EVAL_OP_ADD):
            t = a + b;
            break;
        case (DYN_EVAL_OP_SUB):
            t = a - b;
            break;
        case (DYN_EVAL_OP_MUL):
            t = a * b;
            break;
        default:
            XF_DATABASE_ASSERT(0 && "Unsupported OP!");
    }
    // std::cout << ", ret = " << t << "\n";
    for (int j = 0; j < i; ++j) {
        tmpo[j] = tmpi[j];
    }
    tmpo[i] = t;
}

template <typename T>
T eval_core(T var[DYN_EVAL_NCOL], T imm[DYN_EVAL_NCOL], ap_uint<16> inst[DYN_EVAL_NSTEP]) {
    T tmp[DYN_EVAL_NSTEP + 1][DYN_EVAL_NSTEP];
#pragma HLS array_partition variable = tmp complete dim = 0
    for (int i = 0; i < DYN_EVAL_NSTEP; ++i) {
#pragma HLS unroll
        eval_tmp(var, imm, tmp[i], tmp[i + 1], inst[i], i);
    }
    return tmp[DYN_EVAL_NSTEP][DYN_EVAL_NSTEP - 1];
}

template <int sz, typename T>
struct LoadImm;

template <typename T>
struct LoadImm<4, T> {
    void operator()(hls::stream<ap_uint<32> >& cfgs, T imm[DYN_EVAL_NCOL]) {
        for (int i = 0; i < DYN_EVAL_NCOL; ++i) {
            imm[i] = cfgs.read();
        }
    }
};

template <typename T>
struct LoadImm<8, T> {
    void operator()(hls::stream<ap_uint<32> >& cfgs, T imm[DYN_EVAL_NCOL]) {
        for (int i = 0; i < DYN_EVAL_NCOL; ++i) {
            T l = cfgs.read();
            T h = cfgs.read();
            imm[i] = (h << 32) | l;
        }
    }
};

template <typename T>
inline void load_imm(hls::stream<ap_uint<32> >& cfgs, T imm[DYN_EVAL_NCOL]) {
    LoadImm<sizeof(T), T> f;
    f(cfgs, imm);
}

} // namespace details

/**
 * @brief Dynamic expression evaluation version 2.
 *
 * This primitive has four fixed number of column inputs, and allows up to four constants to be specified via
 * configuration. The operation between the column values and constants can be defined dynamically through the
 * configuration at run-time. The same configuration is used for all rows until the end of input.
 *
 * The constant numbers are assumed to be no more than 32-bits.
 *
 * @tparam T Type of input streams
 *
 * @param cfgs configuration bits of ops and constants.
 *
 * @param col0_istrm input Stream1
 * @param col1_istrm input Stream2
 * @param col2_istrm input Stream3
 * @param col3_istrm input Stream4
 * @param e_istrm end flag of input stream
 *
 * @param ret_ostrm output Stream
 * @param e_ostrm end flag of output stream
 */
template <typename T>
void dynamicEvalV2(hls::stream<ap_uint<32> >& cfgs,
                   //
                   hls::stream<T>& col0_istrm,
                   hls::stream<T>& col1_istrm,
                   hls::stream<T>& col2_istrm,
                   hls::stream<T>& col3_istrm,
                   hls::stream<bool>& e_istrm,
                   //
                   hls::stream<T>& ret_ostrm,
                   hls::stream<bool>& e_ostrm) {
    // config
    ap_uint<16> inst[DYN_EVAL_NSTEP];
#pragma HLS array_partition variable = inst complete
    for (int i = 0; i < DYN_EVAL_NSTEP; i += 2) {
        uint32_t inst2 = cfgs.read();
        inst[i] = (ap_uint<16>)(inst2 & 0xffffU);
        if (i + 1 < DYN_EVAL_NSTEP) inst[i + 1] = (ap_uint<16>)((inst2 >> 16) & 0xffffU);
    }
    T imm[DYN_EVAL_NCOL];
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "eval cannot process data width other than 32 and 64");
    details::load_imm(cfgs, imm);
    // process
    bool e = e_istrm.read();
DYN_EVAL_ROW_LOOP:
    while (!e) {
#pragma HLS pipeline II = 1
        T var[DYN_EVAL_NCOL];
        var[0] = col0_istrm.read();
        var[1] = col1_istrm.read();
        var[2] = col2_istrm.read();
        var[3] = col3_istrm.read();
        T ret = details::eval_core(var, imm, inst);
        ret_ostrm.write(ret);
        e_ostrm.write(false);
        // next ?
        e = e_istrm.read();
    }
    e_ostrm.write(true);
}

} // namespace database
} // namespace xf

#endif // XF_DATABASE_DYNAMIC_EVAL_V2_HPP
