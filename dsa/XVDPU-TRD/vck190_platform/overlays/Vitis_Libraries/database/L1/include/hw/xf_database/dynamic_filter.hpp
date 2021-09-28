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

/**
 * @file dynamic_filter.hpp
 * @brief This file contains run-time-configurable filter primitive.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_DYNAMIC_FILTER_H
#define XF_DATABASE_DYNAMIC_FILTER_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_stream.h"
#include <cstddef>

#include "xf_database/enums.hpp"
#include "xf_database/types.hpp"
#include "xf_database/utils.hpp"

namespace xf {
namespace database {
namespace details {
/// @brief Static Information about true table module.
template <int NCOL>
struct true_table_info {
    /// type of config stream.
    typedef ap_uint<32> cfg_type;
    /// width of address.
    static const size_t addr_width = NCOL + NCOL * (NCOL - 1) / 2;
    /// number of dwords to read for complete table.
    static const size_t dwords_num = addr_width > 5 ? (1ul << (addr_width - 5)) : 1ul;
};

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {

// @tparam W input data width
// @tparam NOCL number of columns
template <int NCOL, int W>
struct DynamicFilterInfo {
    // will error out if mis-used.
};
// special
template <int W>
struct DynamicFilterInfo<4, W> {
    static constexpr int NCOL = 4;
    typedef ap_uint<32> cfg_type;
    static const int var_var_cmp_num = NCOL * (NCOL - 1) / 2;
    static const int dwords_num = (2 * ((W - 1) / 32 + 1) + 1) * NCOL +
                                  (var_var_cmp_num + (32 / FilterOpWidth - 1)) / (32 / FilterOpWidth) +
                                  details::true_table_info<NCOL>::dwords_num;
};
} // namespace database
} // namespace xf

// ------------------------------------------------------------
namespace xf {
namespace database {
namespace details {

using namespace database::enums;

static void broadcast(hls::stream<bool>& e_strm, hls::stream<bool>& e1_strm, hls::stream<bool>& e2_strm) {
    bool e = e_strm.read();
    while (!e) {
        e1_strm.write(false);
        e2_strm.write(false);
        e = e_strm.read();
    }
    e1_strm.write(true);
    e2_strm.write(true);
}

// ------------------------------------------------------------

template <int W>
struct var_const_cmp_info {
    typedef struct {
        ap_uint<FilterOpWidth> lop;
        ap_uint<FilterOpWidth> rop;
        ap_uint<W> l;
        ap_uint<W> r;
    } cfg_type;
};

template <int W>
bool var_const_cmp(typename var_const_cmp_info<W>::cfg_type cfg, ap_uint<W> xu) {
    ap_int<W> x;
    x.range(W - 1, 0) = xu.range(W - 1, 0);
    ap_int<W> l;
    l.range(W - 1, 0) = cfg.l;
    ap_uint<W> lu;
    lu.range(W - 1, 0) = cfg.l;
    ap_int<W> r;
    r.range(W - 1, 0) = cfg.r;
    ap_uint<W> ru;
    ru.range(W - 1, 0) = cfg.r;

    bool bl = false, br = false;

    // one adder should be enough with a bit extra logic,
    // let Vivado do the magic.
    if (cfg.lop == FOP_DC) {
        bl = true;
    } else if (cfg.lop == FOP_EQ) {
        bl = (x == l);
    } else if (cfg.lop == FOP_NE) {
        bl = (x != l);
    } else if (cfg.lop == FOP_GT) {
        bl = (x > l);
    } else if (cfg.lop == FOP_GE) {
        bl = (x >= l);
    } else if (cfg.lop == FOP_GTU) {
        bl = (xu > lu);
    } else if (cfg.lop == FOP_GEU) {
        bl = (xu >= lu);
    } else {
        XF_DATABASE_ASSERT(0 && "Unsupported FOP in var_const_cmp left");
    }

    if (cfg.rop == FOP_DC) {
        br = true;
    } else if (cfg.rop == FOP_EQ) {
        br = (x == r);
    } else if (cfg.rop == FOP_NE) {
        br = (x != r);
    } else if (cfg.rop == FOP_LT) {
        br = (x < r);
    } else if (cfg.rop == FOP_LE) {
        br = (x <= r);
    } else if (cfg.rop == FOP_LTU) {
        br = (xu < ru);
    } else if (cfg.rop == FOP_LEU) {
        br = (xu <= ru);
    } else {
        XF_DATABASE_ASSERT(0 && "Unsupported FOP in var_const_cmp right");
    }

    bool ret = bl && br;
    return ret;
}

// ------------------------------------------------------------

template <int W>
struct var_var_cmp_info {
    typedef struct { ap_uint<FilterOpWidth> op; } cfg_type;
};

template <int W>
bool var_var_cmp(typename var_var_cmp_info<W>::cfg_type cfg, ap_uint<W> xu, ap_uint<W> yu) {
    ap_int<W> x;
    x.range(W - 1, 0) = xu.range(W - 1, 0);
    ap_int<W> y;
    y.range(W - 1, 0) = yu.range(W - 1, 0);
    bool ret = false;
    // one adder should be enough with a bit extra logic,
    // let Vivado do the magic.
    if (cfg.op == FOP_DC) {
        ret = true;
    } else if (cfg.op == FOP_EQ) {
        ret = (x == y);
    } else if (cfg.op == FOP_NE) {
        ret = (x != y);
    } else if (cfg.op == FOP_GT) {
        ret = (x > y);
    } else if (cfg.op == FOP_LT) {
        ret = (x < y);
    } else if (cfg.op == FOP_GE) {
        ret = (x >= y);
    } else if (cfg.op == FOP_LE) {
        ret = (x <= y);
    } else if (cfg.op == FOP_GTU) {
        ret = (xu > yu);
    } else if (cfg.op == FOP_LTU) {
        ret = (xu < yu);
    } else if (cfg.op == FOP_GEU) {
        ret = (xu >= yu);
    } else if (cfg.op == FOP_LEU) {
        ret = (xu <= yu);
    } else {
        XF_DATABASE_ASSERT(0 && "Unsupported FOP in var_var_cmp");
    }
    return ret;
}

// ------------------------------------------------------------

template <int W>
void compare_ops(hls::stream<typename var_const_cmp_info<W>::cfg_type>& cmpv0c_cfg_strm,
                 hls::stream<typename var_const_cmp_info<W>::cfg_type>& cmpv1c_cfg_strm,
                 hls::stream<typename var_const_cmp_info<W>::cfg_type>& cmpv2c_cfg_strm,
                 hls::stream<typename var_const_cmp_info<W>::cfg_type>& cmpv3c_cfg_strm,
                 hls::stream<typename var_var_cmp_info<W>::cfg_type>& cmpv0v1_cfg_strm,
                 hls::stream<typename var_var_cmp_info<W>::cfg_type>& cmpv0v2_cfg_strm,
                 hls::stream<typename var_var_cmp_info<W>::cfg_type>& cmpv0v3_cfg_strm,
                 hls::stream<typename var_var_cmp_info<W>::cfg_type>& cmpv1v2_cfg_strm,
                 hls::stream<typename var_var_cmp_info<W>::cfg_type>& cmpv1v3_cfg_strm,
                 hls::stream<typename var_var_cmp_info<W>::cfg_type>& cmpv2v3_cfg_strm,
                 //
                 hls::stream<ap_uint<W> >& v0_strm,
                 hls::stream<ap_uint<W> >& v1_strm,
                 hls::stream<ap_uint<W> >& v2_strm,
                 hls::stream<ap_uint<W> >& v3_strm,
                 hls::stream<bool>& e_v_strm,
                 //
                 hls::stream<ap_uint<4 + 6> >& addr_strm,
                 hls::stream<bool>& e_addr_strm) {
    typename var_const_cmp_info<W>::cfg_type cmpv0c = cmpv0c_cfg_strm.read();
    typename var_const_cmp_info<W>::cfg_type cmpv1c = cmpv1c_cfg_strm.read();
    typename var_const_cmp_info<W>::cfg_type cmpv2c = cmpv2c_cfg_strm.read();
    typename var_const_cmp_info<W>::cfg_type cmpv3c = cmpv3c_cfg_strm.read();

    typename var_var_cmp_info<W>::cfg_type cmpv0v1 = cmpv0v1_cfg_strm.read();
    typename var_var_cmp_info<W>::cfg_type cmpv0v2 = cmpv0v2_cfg_strm.read();
    typename var_var_cmp_info<W>::cfg_type cmpv0v3 = cmpv0v3_cfg_strm.read();
    typename var_var_cmp_info<W>::cfg_type cmpv1v2 = cmpv1v2_cfg_strm.read();
    typename var_var_cmp_info<W>::cfg_type cmpv1v3 = cmpv1v3_cfg_strm.read();
    typename var_var_cmp_info<W>::cfg_type cmpv2v3 = cmpv2v3_cfg_strm.read();

#if !defined(__SYNTHESIS__) && _XFDB_DYN_FILTER_DEBUG == 1
    int cnt = 0;
#endif

    bool e = e_v_strm.read();
FILTER_MAIN_LOOP:
    while (!e) {
#pragma HLS pipeline II = 1
        e = e_v_strm.read();
        ap_uint<W> v0 = v0_strm.read();
        ap_uint<W> v1 = v1_strm.read();
        ap_uint<W> v2 = v2_strm.read();
        ap_uint<W> v3 = v3_strm.read();

        bool bv0c = var_const_cmp(cmpv0c, v0);
        bool bv1c = var_const_cmp(cmpv1c, v1);
        bool bv2c = var_const_cmp(cmpv2c, v2);
        bool bv3c = var_const_cmp(cmpv3c, v3);

        bool bv0v1 = var_var_cmp(cmpv0v1, v0, v1);
        bool bv0v2 = var_var_cmp(cmpv0v2, v0, v2);
        bool bv0v3 = var_var_cmp(cmpv0v3, v0, v3);
        bool bv1v2 = var_var_cmp(cmpv1v2, v1, v2);
        bool bv1v3 = var_var_cmp(cmpv1v3, v1, v3);
        bool bv2v3 = var_var_cmp(cmpv2v3, v2, v3);

        ap_uint<10> bvec;
        bvec[0] = bv0c;
        bvec[1] = bv1c;
        bvec[2] = bv2c;
        bvec[3] = bv3c;
        bvec[4] = bv0v1;
        bvec[5] = bv0v2;
        bvec[6] = bv0v3;
        bvec[7] = bv1v2;
        bvec[8] = bv1v3;
        bvec[9] = bv2v3;

        addr_strm.write(bvec);
        e_addr_strm.write(false);
#if !defined(__SYNTHESIS__) && _XFDB_DYN_FILTER_DEBUG == 1
        // std::cout << "bvec " << cnt << ": " << bvec.range(3, 0).to_string(2) <<
        // std::endl;
        ++cnt;
#endif
    }
    e_addr_strm.write(true);
#if !defined(__SYNTHESIS__) && _XFDB_DYN_FILTER_DEBUG == 1
    std::cout << "compare_ops has generated " << cnt << " addresses.\n";
#endif
}

// ------------------------------------------------------------

template <typename T, typename T2>
inline void _write_array(T* t, int i, T2 v) {
    t[i] = v;
}

template <typename T, typename T2>
inline void _read_array(const T* t, int i, T2* v) {
    *v = t[i];
}

template <int NCOL>
void true_table(hls::stream<ap_uint<32> >& cfg_strm,
                hls::stream<ap_uint<true_table_info<NCOL>::addr_width> >& addr_strm,
                hls::stream<bool>& e_addr_strm,
                hls::stream<bool>& b_strm,
                hls::stream<bool>& e_b_strm) {
    //
    const size_t addr_width = (size_t)true_table_info<NCOL>::addr_width;
    bool truetable[(1 << addr_width)];
    // XXX break config into multiple 32-bit words, to avoid too wide stream.
    for (unsigned i = 0; i < true_table_info<NCOL>::dwords_num; ++i) {
        ap_uint<32> dw = cfg_strm.read();
    TRUE_TABLE_INIT32:
        for (int j = 0; j < 32; ++j) {
#pragma HLS pipeline II = 1
            //_write_array(truetable, (i * 32 + j), dw[j]);
            truetable[(i << 5) + j] = dw[j];
        }
    }
//
#if !defined(__SYNTHESIS__) && _XFDB_DYN_FILTER_DEBUG == 1
    std::cout << "true_table has finished configuration.\n";
    int cnt = 0;
#endif
    bool e = e_addr_strm.read();
TRUE_TABLE_READ:
    while (!e) {
#pragma HLS pipeline II = 1
        e = e_addr_strm.read();
        ap_uint<addr_width> addr = addr_strm.read();
        bool b;
        //_read_array(truetable, addr, &b);
        b = truetable[addr];
        b_strm.write(b);
        e_b_strm.write(false);
#if !defined(__SYNTHESIS__) && _XFDB_DYN_FILTER_DEBUG == 1
        ++cnt;
#endif
    }
    e_b_strm.write(true);
#if !defined(__SYNTHESIS__) && _XFDB_DYN_FILTER_DEBUG == 1
    std::cout << "true_table has done " << cnt << " lookups.\n";
#endif
}

// ------------------------------------------------------------

template <int WP>
void pred_pass(hls::stream<ap_uint<WP> >& p_strm,
               hls::stream<bool>& e_p_strm,
               hls::stream<bool>& b_strm,
               hls::stream<bool>& e_b_strm,
               //
               hls::stream<ap_uint<WP> >& pay_out_strm,
               hls::stream<bool>& e_out_strm) {
    bool ep = e_p_strm.read();
    bool eb = e_b_strm.read();
    XF_DATABASE_ASSERT(!(ep ^ eb) && "payload and boolean should match");

#if !defined(__SYNTHESIS__) && _XFDB_DYN_FILTER_DEBUG == 1
    int keep = 0, drop = 0;
#endif

FILTER_PRED_PASS:
    while (!(ep || eb)) {
        ep = e_p_strm.read();
        eb = e_b_strm.read();
        XF_DATABASE_ASSERT(!(ep ^ eb) && "payload and boolean should match");
        ap_uint<WP> p = p_strm.read();
        bool b = b_strm.read();
        if (b) {
            pay_out_strm.write(p);
            e_out_strm.write(false);
#if !defined(__SYNTHESIS__) && _XFDB_DYN_FILTER_DEBUG == 1
            ++keep;
#endif
        } else {
            ;
#if !defined(__SYNTHESIS__) && _XFDB_DYN_FILTER_DEBUG == 1
            ++drop;
#endif
        }
    }
    e_out_strm.write(true);
#if !defined(__SYNTHESIS__) && _XFDB_DYN_FILTER_DEBUG == 1
    std::cout << "pred_pass has kept " << keep << " rows, and dropped " << drop << " rows.\n";
#endif
}

// ------------------------------------------------------------
/**
 * @brief parse the 32 bit config stream into each block's config.
 *
 * @tparam NCOL number of variable columns.
 * @tparam W the width of data.
 */
template <int NCOL, int W>
void parse_filter_config(hls::stream<ap_uint<32> >& filter_cfg_strm,
                         //
                         hls::stream<typename var_const_cmp_info<W>::cfg_type> cmpvc_cfg_strms[NCOL],
                         hls::stream<typename var_var_cmp_info<W>::cfg_type>
                             cmpvv_cfg_strms[database::DynamicFilterInfo<NCOL, W>::var_var_cmp_num],
                         hls::stream<typename true_table_info<NCOL>::cfg_type>& tt_cfg_strm) {
    /* config variable-in-range comparison
     *
     * | 0 | immediate left  |
     * | 0 | immediate right |
     * | 0 . . . | lop | rop |
     *      ... x NCOL
     */
    {
        for (int i = 0; i < NCOL; ++i) {
            typename var_const_cmp_info<W>::cfg_type cfg;
            for (int il = 0; il < W; il += 32) {
                int iu = ((il + 31) > (W - 1)) ? (W - 1) : (il + 31);
                cfg.l.range(iu, il) = filter_cfg_strm.read();
            }
            for (int il = 0; il < W; il += 32) {
                int iu = ((il + 31) > (W - 1)) ? (W - 1) : (il + 31);
                cfg.r.range(iu, il) = filter_cfg_strm.read();
            }
            ap_uint<32> dw = filter_cfg_strm.read();
            cfg.lop = dw.range(FilterOpWidth * 2 - 1, FilterOpWidth);
            cfg.rop = dw.range(FilterOpWidth - 1, 0);
            cmpvc_cfg_strms[i].write(cfg);
        }
    }

    /* config variable-to-variable comparison
     *
     * | 0 | opN | ... | op2     | op1     |
     * | 0 . . . . . . | op(N+2) | op(N+1) |
     */
    {
        const int NVV = database::DynamicFilterInfo<NCOL, W>::var_var_cmp_num;
        char nb = 0;
        ap_uint<32> dw;
        for (int i = 0; i < NVV; ++i) {
            if (nb < FilterOpWidth) {
                dw = filter_cfg_strm.read();
                nb = 32;
            }
            typename var_var_cmp_info<W>::cfg_type cfg;
            cfg.op = dw.range(FilterOpWidth - 1, 0);
            cmpvv_cfg_strms[i].write(cfg);
            //
            dw = dw.range(31, FilterOpWidth);
            nb -= FilterOpWidth;
        }
    }

    /* config truetable
     *
     * | . . dword . . . |
     *  ... x dwords_num
     */
    {
        const int NTTW = true_table_info<NCOL>::dwords_num;

        for (int i = 0; i < NTTW; ++i) {
            ap_uint<32> dw = filter_cfg_strm.read();
            tt_cfg_strm.write(dw);
        }
    }
}

// ------------------------------------------------------------

} // namespace details
} // namespace database
} // namespace xf

// ------------------------------------------------------------
namespace xf {
namespace database {

/**
 * @brief Filter payloads according to conditions set during run-time.
 *
 * This primitive, with its 3 overloads, supports filtering rows using up to four columns as conditions.
 * The payload columns should be grouped together into this primitive, using ``combineCol`` primitive, and its total
 * width is not explicitly limited (but naturally bound by resources).
 *
 * The filter conditions consists of whether each of the conditions is within a given range, and relations between any
 * two conditions. The configuration is set once before processing the rows, and reused until the last row.
 * For configuration generation, please refer to the "Design Internals" Section of the document and corresponding test
 * case of this primitive.
 *
 * @tparam W width of all condition column streams, in bits.
 * @tparam WP width of payload column, in bits.
 *
 * @param filter_cfg_strm stream of raw config bits for this primitive.
 * @param v0_strm condition column stream 0.
 * @param v1_strm condition column stream 1.
 * @param v2_strm condition column stream 2.
 * @param v3_strm condition column stream 3.
 * @param pay_in_strm payload input stream.
 * @param e_in_strm end flag stream for input table.
 * @param pay_out_strm payload output stream.
 * @param e_pay_out_strm end flag stream for payload output.
 */
template <int W, int WP>
void dynamicFilter(hls::stream<ap_uint<32> >& filter_cfg_strm,
                   //
                   hls::stream<ap_uint<W> >& v0_strm,
                   hls::stream<ap_uint<W> >& v1_strm,
                   hls::stream<ap_uint<W> >& v2_strm,
                   hls::stream<ap_uint<W> >& v3_strm,
                   hls::stream<ap_uint<WP> >& pay_in_strm,
                   hls::stream<bool>& e_in_strm,
                   //
                   hls::stream<ap_uint<WP> >& pay_out_strm,
                   hls::stream<bool>& e_pay_out_strm) {
#pragma HLS dataflow

    // parse dynamic config.
    hls::stream<typename details::var_const_cmp_info<W>::cfg_type> cmpvc_cfg_strms[4];

    hls::stream<typename details::var_var_cmp_info<W>::cfg_type>
        cmpvv_cfg_strms[DynamicFilterInfo<4, W>::var_var_cmp_num];

    hls::stream<typename details::true_table_info<4>::cfg_type> tt_cfg_strm;

#pragma HLS array_partition variable = cmpvc_cfg_strms
#pragma HLS array_partition variable = cmpvv_cfg_strms

    // split end signal for value and payload.
    hls::stream<bool> e_v_strm;
    hls::stream<bool> e_p_strm;

    hls::stream<ap_uint<details::true_table_info<4>::addr_width> > addr_strm;
    hls::stream<bool> e_addr_strm;

    hls::stream<bool> b_strm;
    hls::stream<bool> e_b_strm;

#pragma HLS stream variable = e_v_strm depth = 8
#pragma HLS stream variable = e_p_strm depth = 32
#pragma HLS stream variable = addr_strm depth = 8
#pragma HLS stream variable = e_addr_strm depth = 8
#pragma HLS stream variable = b_strm depth = 8
#pragma HLS stream variable = e_b_strm depth = 8

    details::broadcast(e_in_strm, e_v_strm, e_p_strm);

    details::parse_filter_config<4, W>(filter_cfg_strm, // 32b
                                       //
                                       cmpvc_cfg_strms, // ((32b im) * 2 + (4b op) * 2 = 96b) * 4
                                       cmpvv_cfg_strms, // (4b op) * 6
                                       tt_cfg_strm);    // 10b addr: 2^10 = 32b * 32

    details::compare_ops(cmpvc_cfg_strms[0], cmpvc_cfg_strms[1], cmpvc_cfg_strms[2],
                         cmpvc_cfg_strms[3], //
                         cmpvv_cfg_strms[0], cmpvv_cfg_strms[1], cmpvv_cfg_strms[2], cmpvv_cfg_strms[3],
                         cmpvv_cfg_strms[4], cmpvv_cfg_strms[5],
                         //
                         v0_strm, v1_strm, v2_strm, v3_strm, e_v_strm,
                         //
                         addr_strm, e_addr_strm);

    details::true_table<4>(tt_cfg_strm,
                           //
                           addr_strm, e_addr_strm,
                           //
                           b_strm, e_b_strm);

    details::pred_pass(pay_in_strm, e_p_strm, b_strm, e_b_strm,
                       //
                       pay_out_strm, e_pay_out_strm);
}

} // namespace database
} // namespace xf

// -------------- handy wrappers for less input ---------------
namespace xf {
namespace database {
namespace details {
template <int W, int WP>
void filter_3_to_4(hls::stream<ap_uint<W> >& v0_strm,
                   hls::stream<ap_uint<W> >& v1_strm,
                   hls::stream<ap_uint<W> >& v2_strm,
                   hls::stream<ap_uint<WP> >& pay_strm,
                   hls::stream<bool>& e_strm,
                   //
                   hls::stream<ap_uint<W> >& va_strm1,
                   hls::stream<ap_uint<W> >& vb_strm1,
                   hls::stream<ap_uint<W> >& vc_strm1,
                   hls::stream<ap_uint<W> >& vd_strm1,
                   hls::stream<ap_uint<WP> >& pay_strm1,
                   hls::stream<bool>& e_strm1) {
    bool e = e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<W> v0 = v0_strm.read();
        ap_uint<W> v1 = v1_strm.read();
        ap_uint<W> v2 = v2_strm.read();
        ap_uint<WP> p = pay_strm.read();
        e = e_strm.read();
        //
        va_strm1.write(v0);
        vb_strm1.write(v1);
        vc_strm1.write(v2);
        vd_strm1.write(v2);
        pay_strm1.write(p);
        e_strm1.write(false);
    }
    e_strm1.write(true);
}
} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Filter payloads according to conditions set during run-time.
 *
 * This function is a wrapper-around the four-condition-column
 * dynamic_filter, just duplicating the columns to feed all its inputs.
 * Thus they share the same configuration bit pattern.
 * All op related to the 4th column should be set to ``FOP_DC``.
 *
 * @tparam W width of all condition column streams, in bits.
 * @tparam WP width of payload column, in bits.
 *
 * @param filter_cfg_strm stream of raw config bits for this primitive.
 * @param v0_strm condition column stream 0.
 * @param v1_strm condition column stream 1.
 * @param v2_strm condition column stream 2.
 * @param pay_in_strm payload input stream.
 * @param e_in_strm end flag stream for input table.
 * @param pay_out_strm payload output stream.
 * @param e_pay_out_strm end flag stream for payload output.
 */
template <int W, int WP>
void dynamicFilter(hls::stream<ap_uint<32> >& filter_cfg_strm,
                   //
                   hls::stream<ap_uint<W> >& v0_strm,
                   hls::stream<ap_uint<W> >& v1_strm,
                   hls::stream<ap_uint<W> >& v2_strm,
                   hls::stream<ap_uint<WP> >& pay_in_strm,
                   hls::stream<bool>& e_in_strm,
                   //
                   hls::stream<ap_uint<WP> >& pay_out_strm,
                   hls::stream<bool>& e_pay_out_strm) {
#pragma HLS dataflow

    hls::stream<ap_uint<W> > va_strm1;
    hls::stream<ap_uint<W> > vb_strm1;
    hls::stream<ap_uint<W> > vc_strm1;
    hls::stream<ap_uint<W> > vd_strm1;
    hls::stream<ap_uint<WP> > pay_strm1;
    hls::stream<bool> e_strm1;

#pragma HLS stream variable = va_strm1 depth = 8
#pragma HLS stream variable = vb_strm1 depth = 8
#pragma HLS stream variable = vc_strm1 depth = 8
#pragma HLS stream variable = vd_strm1 depth = 8
#pragma HLS stream variable = pay_strm1 depth = 32
#pragma HLS stream variable = e_strm1 depth = 8

    details::filter_3_to_4(v0_strm, v1_strm, v2_strm, pay_in_strm,
                           e_in_strm, //
                           va_strm1, vb_strm1, vc_strm1, vd_strm1, pay_strm1, e_strm1);

    dynamicFilter(filter_cfg_strm, //
                  va_strm1, vb_strm1, vc_strm1, vd_strm1, pay_strm1,
                  e_strm1, //
                  pay_out_strm, e_pay_out_strm);
}

} // namespace database
} // namespace xf

namespace xf {
namespace database {
namespace details {
template <int W, int WP>
void filter_2_to_4(hls::stream<ap_uint<W> >& v0_strm,
                   hls::stream<ap_uint<W> >& v1_strm,
                   hls::stream<ap_uint<WP> >& pay_strm,
                   hls::stream<bool>& e_strm,
                   //
                   hls::stream<ap_uint<W> >& va_strm1,
                   hls::stream<ap_uint<W> >& vb_strm1,
                   hls::stream<ap_uint<W> >& vc_strm1,
                   hls::stream<ap_uint<W> >& vd_strm1,
                   hls::stream<ap_uint<WP> >& pay_strm1,
                   hls::stream<bool>& e_strm1) {
    bool e = e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<W> v0 = v0_strm.read();
        ap_uint<W> v1 = v1_strm.read();
        ap_uint<WP> p = pay_strm.read();
        e = e_strm.read();
        //
        va_strm1.write(v0);
        vb_strm1.write(v1);
        vc_strm1.write(v1);
        vd_strm1.write(v1);
        pay_strm1.write(p);
        e_strm1.write(false);
    }
    e_strm1.write(true);
}
} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Filter payloads according to conditions set during run-time.
 *
 * This function is a wrapper-around the four-condition-column
 * dynamic_filter, just duplicating the columns to feed all its inputs.
 * Thus they share the same configuration bit pattern.
 * All op related to the 3rd and 4th columns should be set to ``FOP_DC``.
 *
 * @tparam W width of all condition column streams, in bits.
 * @tparam WP width of payload column, in bits.
 *
 * @param filter_cfg_strm stream of raw config bits for this primitive.
 * @param v0_strm condition column stream 0.
 * @param v1_strm condition column stream 1.
 * @param pay_in_strm payload input stream.
 * @param e_in_strm end flag stream for input table.
 * @param pay_out_strm payload output stream.
 * @param e_pay_out_strm end flag stream for payload output.
 */
template <int W, int WP>
void dynamicFilter(hls::stream<ap_uint<32> >& filter_cfg_strm,
                   //
                   hls::stream<ap_uint<W> >& v0_strm,
                   hls::stream<ap_uint<W> >& v1_strm,
                   hls::stream<ap_uint<WP> >& pay_in_strm,
                   hls::stream<bool>& e_in_strm,
                   //
                   hls::stream<ap_uint<WP> >& pay_out_strm,
                   hls::stream<bool>& e_pay_out_strm) {
#pragma HLS dataflow

    hls::stream<ap_uint<W> > va_strm1;
    hls::stream<ap_uint<W> > vb_strm1;
    hls::stream<ap_uint<W> > vc_strm1;
    hls::stream<ap_uint<W> > vd_strm1;
    hls::stream<ap_uint<WP> > pay_strm1;
    hls::stream<bool> e_strm1;

#pragma HLS stream variable = va_strm1 depth = 8
#pragma HLS stream variable = vb_strm1 depth = 8
#pragma HLS stream variable = vc_strm1 depth = 8
#pragma HLS stream variable = vd_strm1 depth = 8
#pragma HLS stream variable = pay_strm1 depth = 8
#pragma HLS stream variable = e_strm1 depth = 8

    details::filter_2_to_4(v0_strm, v1_strm, pay_in_strm, e_in_strm, //
                           va_strm1, vb_strm1, vc_strm1, vd_strm1, pay_strm1, e_strm1);

    dynamicFilter(filter_cfg_strm, //
                  va_strm1, vb_strm1, vc_strm1, vd_strm1, pay_strm1,
                  e_strm1, //
                  pay_out_strm, e_pay_out_strm);
}

} // namespace database
} // namespace xf

namespace xf {
namespace database {
namespace details {
template <int W, int WP>
void filter_1_to_4(hls::stream<ap_uint<W> >& v0_strm,
                   hls::stream<ap_uint<WP> >& pay_strm,
                   hls::stream<bool>& e_strm,
                   //
                   hls::stream<ap_uint<W> >& va_strm1,
                   hls::stream<ap_uint<W> >& vb_strm1,
                   hls::stream<ap_uint<W> >& vc_strm1,
                   hls::stream<ap_uint<W> >& vd_strm1,
                   hls::stream<ap_uint<WP> >& pay_strm1,
                   hls::stream<bool>& e_strm1) {
    bool e = e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<W> v0 = v0_strm.read();
        ap_uint<WP> p = pay_strm.read();
        e = e_strm.read();
        //
        va_strm1.write(v0);
        vb_strm1.write(v0);
        vc_strm1.write(v0);
        vd_strm1.write(v0);
        pay_strm1.write(p);
        e_strm1.write(false);
    }
    e_strm1.write(true);
}
} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Filter payloads according to conditions set during run-time.
 *
 * This function is a wrapper-around the four-condition-column
 * dynamic_filter, just duplicating the columns to feed all its inputs.
 * Thus they share the same configuration bit pattern.
 * All op related to the 2nd to 4th columns should be set to ``FOP_DC``.
 *
 * @tparam W width of all condition column streams, in bits.
 * @tparam WP width of payload column, in bits.
 *
 * @param filter_cfg_strm stream of raw config bits for this primitive.
 * @param v0_strm condition column stream 0.
 * @param pay_in_strm payload input stream.
 * @param e_in_strm end flag stream for input table.
 * @param pay_out_strm payload output stream.
 * @param e_pay_out_strm end flag stream for payload output.
 */
template <int W, int WP>
void dynamicFilter(hls::stream<ap_uint<32> >& filter_cfg_strm,
                   //
                   hls::stream<ap_uint<W> >& v0_strm,
                   hls::stream<ap_uint<WP> >& pay_in_strm,
                   hls::stream<bool>& e_in_strm,
                   //
                   hls::stream<ap_uint<WP> >& pay_out_strm,
                   hls::stream<bool>& e_pay_out_strm) {
#pragma HLS dataflow

    hls::stream<ap_uint<W> > va_strm1;
    hls::stream<ap_uint<W> > vb_strm1;
    hls::stream<ap_uint<W> > vc_strm1;
    hls::stream<ap_uint<W> > vd_strm1;
    hls::stream<ap_uint<WP> > pay_strm1;
    hls::stream<bool> e_strm1;

#pragma HLS stream variable = va_strm1 depth = 8
#pragma HLS stream variable = vb_strm1 depth = 8
#pragma HLS stream variable = vc_strm1 depth = 8
#pragma HLS stream variable = vd_strm1 depth = 8
#pragma HLS stream variable = pay_strm1 depth = 8
#pragma HLS stream variable = e_strm1 depth = 8

    details::filter_3_to_4(v0_strm, pay_in_strm, e_in_strm, //
                           va_strm1, vb_strm1, vc_strm1, vd_strm1, pay_strm1, e_strm1);

    dynamicFilter(filter_cfg_strm, //
                  va_strm1, vb_strm1, vc_strm1, vd_strm1, pay_strm1,
                  e_strm1, //
                  pay_out_strm, e_pay_out_strm);
}
} // namespace database
} // namespace xf
#endif // XF_DATABASE_DYNAMIC_FILTER_H
