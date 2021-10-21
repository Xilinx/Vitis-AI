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
 * @file direct_group_aggregate.hpp
 * @brief DIRECT AGGREGATE template function implementation.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_DIRECT_GROUP_AGGREGATE_HPP
#define XF_DATABASE_DIRECT_GROUP_AGGREGATE_HPP

#ifndef __cplusplus
#error "xf_database_direct_aggregate hls::stream<> interface, and thus requires C++"
#endif

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#include "xf_database/enums.hpp"
#include "xf_database/utils.hpp"

namespace xf {
namespace database {
namespace details {

// max
template <int DATINW, int DATOUTW, int KINW, int DIRECTW>
void direct_aggr_max(hls::stream<ap_uint<DATINW> >& vin_strm,
                     hls::stream<bool>& in_e_strm,
                     hls::stream<ap_uint<DATOUTW> >& vout_strm,
                     hls::stream<bool>& out_e_strm,
                     hls::stream<ap_uint<KINW> >& kin_strm,
                     hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    bool o_isFinal = 0;

    ap_int<DATOUTW> arg_out[(1 << DIRECTW)];
    ap_uint<1> flag[(1 << DIRECTW)];

    ap_int<DATOUTW> state_c, state_r0, state_r1, state_r2;
    ap_uint<DIRECTW> addr_c, addr_r0, addr_r1, addr_r2;
    addr_r0 = addr_r1 = addr_r2 = -1; // The 0x7ff should never be accessed

    state_c = 0;
    state_r0 = 0;
    state_r1 = 0;
    state_r2 = 0;
    addr_c = 0;
    ///   @warning : the DIRECTW >= KINW in the dirt_aggr
    //    ap_uint<32> direct_key;

    bool i_isFinal = in_e_strm.read();
    o_isFinal = i_isFinal;
    // initialize min
    ap_int<DATOUTW> min;
    min(DATOUTW - 2, 0) = 0;
    min[DATOUTW - 1] = 1;

// initialize arg_out to zero
INIT_arg_outLOOP:
    for (int i = 0; i < ((1 << DIRECTW)); i++) {
#pragma HLS PIPELINE II = 1
        arg_out[i] = min;
        flag[i] = 0;
    }

    while (!o_isFinal) {
#pragma HLS dependence array inter false
#pragma HLS PIPELINE II = 1

        // 1)Get data's address(addr == cur_key.o_data == keys' dictionary)
        ap_uint<KINW> cur_key = kin_strm.read();
        addr_c = cur_key; // 32bit to DIRECTW width convert in direct_aggr

        // 2)Get data to form the RAM( arg_out)
        ap_int<DATINW> tmp = vin_strm.read();
        ap_int<DATOUTW> cur_val = tmp;

        // 2.2) Get the isFinal
        i_isFinal = in_e_strm.read();
        o_isFinal = i_isFinal;

        // 3)Read RAM and select the state_c based on the addr, addr0 and addr1
        if (addr_c == addr_r0)
            state_c = state_r0;
        else if (addr_c == addr_r1)
            state_c = state_r1;
        else if (addr_c == addr_r2)
            state_c = state_r2;
        else {
            state_c = arg_out[addr_c];
            flag[addr_c] = 1;
        }
        // 4)Write back to RAM
        state_c = ((state_c > cur_val) ? state_c : cur_val);

        arg_out[addr_c] = state_c;
        // 5)shift the whole data line 1 cycle for RAM content( state) and ADDRESS
        // (addr)
        state_r2 = state_r1;
        state_r1 = state_r0;
        state_r0 = state_c;
        addr_r2 = addr_r1;
        addr_r1 = addr_r0;
        addr_r0 = addr_c;

    } // end while

    for (int i = 0; i < (1 << DIRECTW); i++) {
#pragma HLS PIPELINE
        if (flag[i]) {
            vout_strm.write(arg_out[i]);
            kout_strm.write(i);
            out_e_strm.write(0);
        } else {
        }
    }
    out_e_strm.write(1);
} // end direct_aggr_max

// min
template <int DATINW, int DATOUTW, int KINW, int DIRECTW>
void direct_aggr_min(hls::stream<ap_uint<DATINW> >& vin_strm,
                     hls::stream<bool>& in_e_strm,
                     hls::stream<ap_uint<DATOUTW> >& vout_strm,
                     hls::stream<bool>& out_e_strm,
                     hls::stream<ap_uint<KINW> >& kin_strm,
                     hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    bool o_isFinal = 0;

    ap_int<DATOUTW> arg_out[(1 << DIRECTW)];
    ap_uint<1> flag[(1 << DIRECTW)];

    ap_int<DATOUTW> state_c, state_r0, state_r1, state_r2, state_fst;
    ap_uint<DIRECTW> addr_c, addr_r0, addr_r1, addr_r2, addr_fst;
    addr_r0 = addr_r1 = addr_r2 = -1; // The 0x7ff should never be accessed
    state_c = 0;
    state_r0 = 0;
    state_r1 = 0;
    state_r2 = 0;
    state_fst = 0;
    addr_c = 0;
    addr_fst = 0;

    ///   @warning : the DIRECTW >= KINW in the dirt_aggr
    //    ap_uint<32> direct_key;

    bool i_isFinal = in_e_strm.read();
    o_isFinal = i_isFinal;
    // initialize max
    ap_int<DATOUTW> max;
    max(DATOUTW - 2, 0) = 0xffffffffffffffff;
    max[DATOUTW - 1] = 0;
// initialize arg_out to zero
INIT_arg_outLOOP:
    for (int i = 0; i < ((1 << DIRECTW)); i++) {
#pragma HLS PIPELINE II = 1
        arg_out[i] = max; // this value for min() need to be set to maxvalue
        flag[i] = 0;
    }

    while (!o_isFinal) {
#pragma HLS dependence array inter false
#pragma HLS PIPELINE II = 1

        // 1)Get data's address(addr == cur_key.o_data == keys' dictionary)
        ap_uint<KINW> cur_key = kin_strm.read();
        addr_c = cur_key; // 32bit to DIRECTW width convert in direct_aggr

        // 2)Get data to form the RAM( arg_out)
        ap_int<DATINW> tmp = vin_strm.read();
        ap_int<DATOUTW> cur_val = tmp;
        //_XF_DB_FPRINT(fp,"cur_val = %lld\n",cur_val.VAL);

        // 2.2) Get the isFinal
        i_isFinal = in_e_strm.read();
        o_isFinal = i_isFinal;

        // 3)Read RAM and select the state_c based on the addr, addr0 and addr1
        if (addr_c == addr_r0)
            state_c = state_r0;
        else if (addr_c == addr_r1)
            state_c = state_r1;
        else if (addr_c == addr_r2)
            state_c = state_r2;
        else {
            state_c = arg_out[addr_c];
            flag[addr_c] = 1;
        }
        // 4)Write back to RAM
        state_c = ((state_c < cur_val) ? state_c : cur_val);

        arg_out[addr_c] = state_c;
        // 5)shift the whole data line 1 cycle for RAM content( state) and ADDRESS
        // (addr)
        state_r2 = state_r1;
        state_r1 = state_r0;
        state_r0 = state_c;
        addr_r2 = addr_r1;
        addr_r1 = addr_r0;
        addr_r0 = addr_c;

    } // end while

    for (int i = 0; i < (1 << DIRECTW); i++) {
#pragma HLS PIPELINE
        if (flag[i]) {
            vout_strm.write(arg_out[i]);
            kout_strm.write(i);
            out_e_strm.write(0);
        } else {
        }
    }
    out_e_strm.write(1);
} // end direct_aggr_min

// sum
template <int DATINW, int DATOUTW, int KINW, int DIRECTW>
void direct_aggr_sum(hls::stream<ap_uint<DATINW> >& vin_strm,
                     hls::stream<bool>& in_e_strm,
                     hls::stream<ap_uint<DATOUTW> >& vout_strm,
                     hls::stream<bool>& out_e_strm,
                     hls::stream<ap_uint<KINW> >& kin_strm,
                     hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    bool o_isFinal = 0;

    ap_int<DATOUTW> arg_out[(1 << DIRECTW)];
    ap_uint<1> flag[(1 << DIRECTW)];

    ap_int<DATOUTW> state_c, state_r0, state_r1, state_r2;
    ap_uint<DIRECTW> addr_c, addr_r0, addr_r1, addr_r2;
    addr_r0 = addr_r1 = addr_r2 = -1; // The 0x7ff should never be accessed
    state_c = 0;
    state_r0 = 0;
    state_r1 = 0;
    state_r2 = 0;
    addr_c = 0;

    bool i_isFinal = in_e_strm.read();
    o_isFinal = i_isFinal;
// initialize arg_out to zero
INIT_arg_outLOOP:
    for (int i = 0; i < ((1 << DIRECTW)); i++) {
#pragma HLS PIPELINE II = 1
        arg_out[i] = 0;
        flag[i] = 0;
    }

    while (!o_isFinal) {
#pragma HLS dependence array inter false
#pragma HLS PIPELINE II = 1

        // 1)Get data's address(addr == cur_key.o_data == keys' dictionary)
        ap_uint<KINW> cur_key = kin_strm.read();
        addr_c = cur_key; // 32bit to DIRECTW width convert in direct_aggr

        // 2)Get data to form the RAM( arg_out)
        ap_int<DATINW> tmp = vin_strm.read();
        ap_int<DATOUTW> cur_val = tmp;
        //_XF_DB_FPRINT(fp,"cur_val = %lld\n",cur_val.VAL);

        // 2.2) Get the isFinal
        i_isFinal = in_e_strm.read();
        o_isFinal = i_isFinal;

        // 3)Read RAM and select the state_c based on the addr, addr0 and addr1
        if (addr_c == addr_r0)
            state_c = state_r0;
        else if (addr_c == addr_r1)
            state_c = state_r1;
        else if (addr_c == addr_r2)
            state_c = state_r2;
        else {
            state_c = arg_out[addr_c];
            flag[addr_c] = 1;
        }
        // 4)Write back to RAM
        state_c = state_c + cur_val;

        arg_out[addr_c] = state_c;
        // 5)shift the whole data line 1 cycle for RAM content( state) and ADDRESS
        // (addr)
        state_r2 = state_r1;
        state_r1 = state_r0;
        state_r0 = state_c;
        addr_r2 = addr_r1;
        addr_r1 = addr_r0;
        addr_r0 = addr_c;

    } // end while

    for (int i = 0; i < (1 << DIRECTW); i++) {
#pragma HLS PIPELINE
        if (flag[i]) {
            vout_strm.write(arg_out[i]);
            kout_strm.write(i);
            out_e_strm.write(0);
        } else {
        }
    }
    out_e_strm.write(1);

} // end direct_aggr_sum

// countone
template <int DATINW, int DATOUTW, int KINW, int DIRECTW>
void direct_aggr_countone(hls::stream<ap_uint<DATINW> >& vin_strm,
                          hls::stream<bool>& in_e_strm,
                          hls::stream<ap_uint<DATOUTW> >& vout_strm,
                          hls::stream<bool>& out_e_strm,
                          hls::stream<ap_uint<KINW> >& kin_strm,
                          hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    bool o_isFinal = 0;

    ap_int<DATOUTW> arg_out[(1 << DIRECTW)];
    ap_uint<1> flag[(1 << DIRECTW)];

    ap_int<DATOUTW> state_c, state_r0, state_r1, state_r2, state_fst;
    ap_uint<DIRECTW> addr_c, addr_r0, addr_r1, addr_r2, addr_fst;
    addr_r0 = addr_r1 = addr_r2 = -1; // The 0x7ff should never be accessed
    state_c = 0;
    state_r0 = 0;
    state_r1 = 0;
    state_r2 = 0;
    state_fst = 0;
    addr_c = 0;
    addr_fst = 0;

    bool i_isFinal = in_e_strm.read();
    o_isFinal = i_isFinal;
// initialize arg_out to zero
INIT_arg_outLOOP:
    for (int i = 0; i < ((1 << DIRECTW)); i++) {
#pragma HLS PIPELINE II = 1
        arg_out[i] = 0;
        flag[i] = 0;
    }

    while (!o_isFinal) {
#pragma HLS dependence array inter false
#pragma HLS PIPELINE II = 1

        // 1)Get data's address(addr == cur_key.o_data == keys' dictionary)
        ap_uint<KINW> cur_key = kin_strm.read();
        addr_c = cur_key; // 32bit to DIRECTW width convert in direct_aggr

        // 2)Get data to form the RAM( arg_out)
        ap_int<DATINW> tmp = vin_strm.read();
        ap_int<DATOUTW> cur_val = tmp;

        // 2.2) Get the isFinal
        i_isFinal = in_e_strm.read();
        o_isFinal = i_isFinal;

        // 3)Read RAM and select the state_c based on the addr, addr0 and addr1
        if (addr_c == addr_r0)
            state_c = state_r0;
        else if (addr_c == addr_r1)
            state_c = state_r1;
        else if (addr_c == addr_r2)
            state_c = state_r2;
        else {
            state_c = arg_out[addr_c];
            flag[addr_c] = 1;
        }
        // 4)Write back to RAM
        state_c++;

        arg_out[addr_c] = state_c;
        // 5)shift the whole data line 1 cycle for RAM content( state) and ADDRESS
        // (addr)
        state_r2 = state_r1;
        state_r1 = state_r0;
        state_r0 = state_c;
        addr_r2 = addr_r1;
        addr_r1 = addr_r0;
        addr_r0 = addr_c;

    } // end while

    for (int i = 0; i < (1 << DIRECTW); i++) {
#pragma HLS PIPELINE
        if (flag[i]) {
            vout_strm.write(arg_out[i]);
            kout_strm.write(i);
            out_e_strm.write(0);
        } else {
        }
    }
    out_e_strm.write(1);
} // end direct_aggr_countone

// avg
template <int DATINW, int DATOUTW, int KINW, int DIRECTW>
void direct_aggr_avg(hls::stream<ap_uint<DATINW> >& vin_strm,
                     hls::stream<bool>& in_e_strm,
                     hls::stream<ap_uint<DATOUTW> >& vout_strm,
                     hls::stream<bool>& out_e_strm,
                     hls::stream<ap_uint<KINW> >& kin_strm,
                     hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    bool o_isFinal = 0;

    ap_int<DATOUTW> sum[(1 << DIRECTW)];
    ap_int<DATOUTW> cnt[(1 << DIRECTW)];
    ap_uint<1> flag[(1 << DIRECTW)];

    ap_int<DATOUTW> state_c, state_r0, state_r1, state_r2;
    ap_int<DATOUTW> agsum_c, agsum_r0, agsum_r1, agsum_r2;
    ap_uint<DIRECTW> addr_c, addr_r0, addr_r1, addr_r2;
    addr_r0 = addr_r1 = addr_r2 = -1; // The 0x7ff should never be accessed
    state_c = state_r0 = state_r1 = state_r2 = 0;
    agsum_c = agsum_r0 = agsum_r1 = agsum_r2 = 0;
    addr_c = 0;

    bool i_isFinal = in_e_strm.read();
    o_isFinal = i_isFinal;
// initialize arg_out to zero
INIT_arg_outLOOP:
    for (int i = 0; i < ((1 << DIRECTW)); i++) {
#pragma HLS PIPELINE II = 1
        sum[i] = 0;
        cnt[i] = 0;
        flag[i] = 0;
    }

    while (!o_isFinal) {
#pragma HLS dependence array inter false
#pragma HLS PIPELINE II = 1

        // 1)Get data's address(addr == cur_key.o_data == keys' dictionary)
        ap_uint<KINW> cur_key = kin_strm.read();
        addr_c = cur_key; // 32bit to DIRECTW width convert in direct_aggr

        // 2)Get data to form the RAM( arg_out)
        ap_int<DATINW> tmp = vin_strm.read();
        ap_int<DATOUTW> cur_val = tmp;

        // 2.2) Get the isFinal
        i_isFinal = in_e_strm.read();
        o_isFinal = i_isFinal;

        // 3)Read RAM and select the state_c based on the addr, addr0 and addr1
        if (addr_c == addr_r0) {
            state_c = state_r0;
            agsum_c = agsum_r0;
        } else if (addr_c == addr_r1) {
            state_c = state_r1;
            agsum_c = agsum_r1;
        } else if (addr_c == addr_r2) {
            state_c = state_r2;
            agsum_c = agsum_r2;
        } else {
            state_c = cnt[addr_c];
            agsum_c = sum[addr_c];
            flag[addr_c] = 1;
        }

        // 4)Write back to RAM
        state_c++;
        agsum_c = agsum_c + cur_val;

        cnt[addr_c] = state_c;
        sum[addr_c] = agsum_c;
        // 5)shift the whole data line 1 cycle for RAM content( state) and ADDRESS
        // (addr)
        state_r2 = state_r1;
        state_r1 = state_r0;
        state_r0 = state_c;
        agsum_r2 = agsum_r1;
        agsum_r1 = agsum_r0;
        agsum_r0 = agsum_c;
        addr_r2 = addr_r1;
        addr_r1 = addr_r0;
        addr_r0 = addr_c;

    } // end while

    for (int i = 0; i < (1 << DIRECTW); i++) {
#pragma HLS PIPELINE
        if (flag[i]) {
            vout_strm.write(sum[i] / cnt[i]);
            kout_strm.write(i);
            out_e_strm.write(0);
        } else {
        }
    }
    out_e_strm.write(1);
} // end direct_aggr_avg

// variance
template <int DATINW, int DATOUTW, int KINW, int DIRECTW>
void direct_aggr_variance(hls::stream<ap_uint<DATINW> >& vin_strm,
                          hls::stream<bool>& in_e_strm,
                          hls::stream<ap_uint<DATOUTW> >& vout_strm,
                          hls::stream<bool>& out_e_strm,
                          hls::stream<ap_uint<KINW> >& kin_strm,
                          hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    bool o_isFinal = 0;

    ap_int<DATOUTW> sum[(1 << DIRECTW)];
    ap_int<DATOUTW> tow[(1 << DIRECTW)];
    ap_int<DATOUTW> cnt[(1 << DIRECTW)];
    ap_uint<1> flag[(1 << DIRECTW)];

    ap_int<DATOUTW> state_c, state_r0, state_r1, state_r2;
    ap_int<DATOUTW> agsum_c, agsum_r0, agsum_r1, agsum_r2;
    ap_int<DATOUTW> agtow_c, agtow_r0, agtow_r1, agtow_r2;
    ap_uint<DIRECTW> addr_c, addr_r0, addr_r1, addr_r2;
    ap_int<DATOUTW> variance;
    addr_r0 = addr_r1 = addr_r2 = -1; // The 0x7ff should never be accessed
    state_c = state_r0 = state_r1 = state_r2 = 0;
    agsum_c = agsum_r0 = agsum_r1 = agsum_r2 = 0;
    agtow_c = agtow_r0 = agtow_r1 = agtow_r2 = 0;
    variance = 0;
    addr_c = 0;

    bool i_isFinal = in_e_strm.read();
    o_isFinal = i_isFinal;
// initialize arg_out to zero
INIT_arg_outLOOP:
    for (int i = 0; i < ((1 << DIRECTW)); i++) {
#pragma HLS PIPELINE II = 1
        sum[i] = 0;
        cnt[i] = 0;
        tow[i] = 0;
        flag[i] = 0;
    }

    while (!o_isFinal) {
#pragma HLS dependence array inter false
#pragma HLS PIPELINE II = 1

        // 1)Get data's address(addr == cur_key.o_data == keys' dictionary)
        ap_uint<KINW> cur_key = kin_strm.read();
        addr_c = cur_key; // 32bit to DIRECTW width convert in direct_aggr

        // 2)Get data to form the RAM( arg_out)
        ap_int<DATINW> tmp = vin_strm.read();
        ap_int<DATOUTW> cur_val = tmp;

        // 2.2) Get the isFinal
        i_isFinal = in_e_strm.read();
        o_isFinal = i_isFinal;

        // 3)Read RAM and select the state_c based on the addr, addr0 and addr1
        if (addr_c == addr_r0) {
            state_c = state_r0;
            agsum_c = agsum_r0;
            agtow_c = agtow_r0;
        } else if (addr_c == addr_r1) {
            state_c = state_r1;
            agsum_c = agsum_r1;
            agtow_c = agtow_r1;
        } else if (addr_c == addr_r2) {
            state_c = state_r2;
            agsum_c = agsum_r2;
            agtow_c = agtow_r2;
        } else {
            state_c = cnt[addr_c];
            agsum_c = sum[addr_c];
            agtow_c = tow[addr_c];
            flag[addr_c] = 1;
        }

        // 4)Write back to RAM
        state_c++;
        agsum_c = agsum_c + cur_val;
        agtow_c = agtow_c + cur_val * cur_val;

        cnt[addr_c] = state_c;
        sum[addr_c] = agsum_c;
        tow[addr_c] = agtow_c;
        // 5)shift the whole data line 1 cycle for RAM content( state) and ADDRESS
        // (addr)
        state_r2 = state_r1;
        state_r1 = state_r0;
        state_r0 = state_c;

        agsum_r2 = agsum_r1;
        agsum_r1 = agsum_r0;
        agsum_r0 = agsum_c;

        agtow_r2 = agtow_r1;
        agtow_r1 = agtow_r0;
        agtow_r0 = agtow_c;

        addr_r2 = addr_r1;
        addr_r1 = addr_r0;
        addr_r0 = addr_c;

    } // end while

    for (int i = 0; i < (1 << DIRECTW); i++) {
#pragma HLS PIPELINE
        if (flag[i]) {
            variance = (tow[i] - sum[i] * sum[i] / cnt[i]) / cnt[i];
            vout_strm.write(variance);
            kout_strm.write(i);
            out_e_strm.write(0);
        } else {
        }
    }
    out_e_strm.write(1);
} // end direct_aggr_variance

// normL1
template <int DATINW, int DATOUTW, int KINW, int DIRECTW>
void direct_aggr_normL1(hls::stream<ap_uint<DATINW> >& vin_strm,
                        hls::stream<bool>& in_e_strm,
                        hls::stream<ap_uint<DATOUTW> >& vout_strm,
                        hls::stream<bool>& out_e_strm,
                        hls::stream<ap_uint<KINW> >& kin_strm,
                        hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    bool o_isFinal = 0;

    ap_int<DATOUTW> arg_out[(1 << DIRECTW)];
    ap_uint<1> flag[(1 << DIRECTW)];

    ap_int<DATOUTW> state_c, state_r0, state_r1, state_r2;
    ap_uint<DIRECTW> addr_c, addr_r0, addr_r1, addr_r2;
    addr_r0 = addr_r1 = addr_r2 = -1; // The 0x7ff should never be accessed
    state_c = state_r0 = state_r1 = state_r2 = 0;
    addr_c = 0;

    bool i_isFinal = in_e_strm.read();
    o_isFinal = i_isFinal;
// initialize arg_out to zero
INIT_arg_outLOOP:
    for (int i = 0; i < ((1 << DIRECTW)); i++) {
#pragma HLS PIPELINE II = 1
        arg_out[i] = 0;
        flag[i] = 0;
    }

    while (!o_isFinal) {
#pragma HLS dependence array inter false

#pragma HLS PIPELINE II = 1

        // 1)Get data's address(addr == cur_key.o_data == keys' dictionary)
        ap_uint<KINW> cur_key = kin_strm.read();
        addr_c = cur_key; // 32bit to DIRECTW width convert in direct_aggr

        // 2)Get data to form the RAM( arg_out)
        ap_int<DATINW> tmp = vin_strm.read();
        ap_int<DATOUTW> cur_val = tmp;

        // 2.2) Get the isFinal
        i_isFinal = in_e_strm.read();
        o_isFinal = i_isFinal;

        // 3)Read RAM and select the state_c based on the addr, addr0 and addr1
        if (addr_c == addr_r0)
            state_c = state_r0;
        else if (addr_c == addr_r1)
            state_c = state_r1;
        else if (addr_c == addr_r2)
            state_c = state_r2;
        else {
            state_c = arg_out[addr_c];
            flag[addr_c] = 1;
        }
        // 4)Write back to RAM
        state_c = state_c + ((cur_val > 0) ? cur_val : (ap_int<DATOUTW>)-cur_val);

        arg_out[addr_c] = state_c;
        // 5)shift the whole data line 1 cycle for RAM content( state) and ADDRESS
        // (addr)
        state_r2 = state_r1;
        state_r1 = state_r0;
        state_r0 = state_c;
        addr_r2 = addr_r1;
        addr_r1 = addr_r0;
        addr_r0 = addr_c;

    } // end while

    for (int i = 0; i < (1 << DIRECTW); i++) {
#pragma HLS PIPELINE
        if (flag[i]) {
            vout_strm.write(arg_out[i]);
            kout_strm.write(i);
            out_e_strm.write(0);
        } else {
        }
    }
    out_e_strm.write(1);

} // end direct_aggr_normL1

// normL2
template <int DATINW, int DATOUTW, int KINW, int DIRECTW>
void direct_aggr_normL2(hls::stream<ap_uint<DATINW> >& vin_strm,
                        hls::stream<bool>& in_e_strm,
                        hls::stream<ap_uint<DATOUTW> >& vout_strm,
                        hls::stream<bool>& out_e_strm,
                        hls::stream<ap_uint<KINW> >& kin_strm,
                        hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    XF_DATABASE_ASSERT((DATOUTW < 34) && "hls::sqrt only support (I_<34)&&(F_<31)");

    bool o_isFinal = 0;

    ap_int<DATOUTW> arg_out[(1 << DIRECTW)];
    ap_uint<1> flag[(1 << DIRECTW)];

    ap_int<DATOUTW> state_c, state_r0, state_r1, state_r2;
    ap_uint<DIRECTW> addr_c, addr_r0, addr_r1, addr_r2;
    addr_r0 = addr_r1 = addr_r2 = -1; // The 0x7ff should never be accessed
    state_c = state_r0 = state_r1 = state_r2 = 0;
    addr_c = 0;

    bool i_isFinal = in_e_strm.read();
    o_isFinal = i_isFinal;
// initialize arg_out to zero
INIT_arg_outLOOP:
    for (int i = 0; i < ((1 << DIRECTW)); i++) {
#pragma HLS PIPELINE II = 1
        arg_out[i] = 0;
        flag[i] = 0;
    }

    while (!o_isFinal) {
#pragma HLS dependence array inter false
#pragma HLS PIPELINE II = 1

        // 1)Get data's address(addr == cur_key.o_data == keys' dictionary)
        ap_uint<KINW> cur_key = kin_strm.read();
        addr_c = cur_key; // 32bit to DIRECTW width convert in direct_aggr

        // 2)Get data to form the RAM( arg_out)
        ap_int<DATINW> tmp = vin_strm.read();
        ap_int<DATOUTW> cur_val = tmp;
        //_XF_DB_FPRINT(fp,"cur_val = %lld\n",cur_val.VAL);

        // 2.2) Get the isFinal
        i_isFinal = in_e_strm.read();
        o_isFinal = i_isFinal;

        // 3)Read RAM and select the state_c based on the addr, addr0 and addr1
        if (addr_c == addr_r0)
            state_c = state_r0;
        else if (addr_c == addr_r1)
            state_c = state_r1;
        else if (addr_c == addr_r2)
            state_c = state_r2;
        else {
            state_c = arg_out[addr_c];
            flag[addr_c] = 1;
        }
        // 4)Write back to RAM
        state_c = state_c + (cur_val * cur_val);

        arg_out[addr_c] = state_c;
        // 5)shift the whole data line 1 cycle for RAM content( state) and ADDRESS
        // (addr)
        state_r2 = state_r1;
        state_r1 = state_r0;
        state_r0 = state_c;
        addr_r2 = addr_r1;
        addr_r1 = addr_r0;
        addr_r0 = addr_c;

    } // end while

    for (int i = 0; i < (1 << DIRECTW); i++) {
#pragma HLS PIPELINE
        if (flag[i]) {
            vout_strm.write(hls::sqrt(arg_out[i]));
            kout_strm.write(i);
            out_e_strm.write(0);
        } else {
        }
    }
    out_e_strm.write(1);

} // end direct_aggr_normL2

// initialize ram
template <int DATINW, int DATOUTW, int DIRECTW>
void initialize_ram(ap_uint<32> op, ap_int<DATOUTW> sum[(1 << DIRECTW)], ap_int<DATOUTW + 1> cnt[(1 << DIRECTW)]) {
#pragma HLS inline off

    ap_uint<DATOUTW> value = 0;
    if (op == AOP_MAX) {
        if (DATOUTW == 32)
            value = 0x080000000;
        else if (DATOUTW == 64)
            value = 0x08000000000000000;
    } else if (op == AOP_MIN) {
        if (DATOUTW == 32)
            value = 0x07fffffff;
        else if (DATOUTW == 64)
            value = 0x07fffffffffffffff;
    } else {
        value = 0;
    }

// initialize arg_out to zero
INIT_arg_outLOOP:
    for (int i = 0; i < ((1 << DIRECTW)); i++) {
#pragma HLS PIPELINE II = 1
        sum[i] = 0;     // for sum, norm
        cnt[i] = value; // for min, max and cnt
    }
} // end initialize ram

// calculate min/max according to op
template <int DATINW, int DATOUTW, int DIRECTW>
void direct_aggr_min_max(ap_uint<32> op,
                         ap_int<DATOUTW + 1> aggr[(1 << DIRECTW)],
                         hls::stream<ap_uint<DATINW> >& vin_strm,
                         hls::stream<bool>& in_e_strm,
                         hls::stream<ap_uint<DATOUTW> >& vout_strm,
                         hls::stream<bool>& out_e_strm,
                         hls::stream<ap_uint<DIRECTW> >& kin_strm,
                         hls::stream<ap_uint<DIRECTW> >& kout_strm) {
#pragma HLS inline off

    bool sign = op == AOP_MIN;
    ap_int<DATOUTW> state_n, state_c, state_r0, state_r1, state_r2, state_r3, state_r4, state_r5;
    ap_uint<DIRECTW> addr_c, addr_r0, addr_r1, addr_r2, addr_r3, addr_r4, addr_r5;
    addr_r0 = addr_r1 = addr_r2 = 1 << DIRECTW; // The 0x7ff should never be accessed
    addr_c = 0;
    state_n = 0;
    state_c = 0;
    state_r0 = 0;
    state_r1 = 0;
    state_r2 = 0;
    state_r3 = 0;
    state_r4 = 0;
    state_r5 = 0;
    addr_r3 = addr_r4 = addr_r5 = 0;

    ///   @warning : the DIRECTW >= KINW in the dirt_aggr
    bool isFinal = in_e_strm.read();
    while (!isFinal) {
#pragma HLS dependence variable = aggr pointer inter false
#pragma HLS PIPELINE II = 1

        // 1)Get data's address(addr == cur_key.o_data == keys' dictionary)
        ap_uint<DIRECTW> cur_key = kin_strm.read();
        addr_c = cur_key; // 32bit to DIRECTW width convert in direct_aggr

        // 2)Get data to form input stream
        ap_int<DATINW> tmp = vin_strm.read();
        ap_int<DATOUTW> cur_val = tmp;
        isFinal = in_e_strm.read();

        // 3)Read RAM and select the state_c based on the addr, addr0 and addr1
        if (addr_c == addr_r0)
            state_c = state_r0;
        else if (addr_c == addr_r1)
            state_c = state_r1;
        else if (addr_c == addr_r2)
            state_c = state_r2;
        else if (addr_c == addr_r3)
            state_c = state_r3;
        else if (addr_c == addr_r4)
            state_c = state_r4;
        else if (addr_c == addr_r5)
            state_c = state_r5;
        else
            state_c = aggr[addr_c](DATOUTW - 1, 0);

        // 4)calculate new aggr
        state_n = ((state_c > cur_val) ^ sign ? state_c : cur_val);

        // 5)write back to ram
        aggr[addr_c] = (1, state_n);

        // 5)shift the whole data line 1 cycle for RAM content( state) and ADDRESS
        // (addr)
        state_r5 = state_r4;
        state_r4 = state_r3;
        state_r3 = state_r2;
        state_r2 = state_r1;
        state_r1 = state_r0;
        state_r0 = state_n;

        addr_r5 = addr_r4;
        addr_r4 = addr_r3;
        addr_r3 = addr_r2;
        addr_r2 = addr_r1;
        addr_r1 = addr_r0;
        addr_r0 = addr_c;
    } // end while

    // output
    for (int i = 0; i < (1 << DIRECTW); i++) {
#pragma HLS PIPELINE

        ap_int<DATOUTW + 1> temp = aggr[i];
        if (temp[DATOUTW] == 1) {
            vout_strm.write(temp(DATOUTW - 1, 0));
            kout_strm.write(i);
            out_e_strm.write(0);
        } else {
            // no operation
        }
    }
    out_e_strm.write(1);
} // end direct_aggr_max

// calculate sum/cnt/mean/normalization according to op
template <int DATINW, int DATOUTW, int DIRECTW>
void direct_aggr_sum_cnt_mean_norm(ap_uint<32> op,
                                   ap_int<DATOUTW> sum[(1 << DIRECTW)],
                                   ap_int<DATOUTW + 1> cnt[(1 << DIRECTW)],
                                   hls::stream<ap_uint<DATINW> >& vin_strm,
                                   hls::stream<bool>& in_e_strm,
                                   hls::stream<ap_uint<DATOUTW> >& vout_strm,
                                   hls::stream<bool>& out_e_strm,
                                   hls::stream<ap_uint<DIRECTW> >& kin_strm,
                                   hls::stream<ap_uint<DIRECTW> >& kout_strm) {
#pragma HLS inline off

    ap_int<DATOUTW> state_sn, state_s, state_s0, state_s1, state_s2, state_s3, state_s4, state_s5;
    ap_int<DATOUTW> state_cn, state_c, state_c0, state_c1, state_c2, state_c3, state_c4, state_c5;
    ap_uint<DIRECTW> addr_c, addr_r0, addr_r1, addr_r2, addr_r3, addr_r4, addr_r5;

    addr_r0 = addr_r1 = addr_r2 = 1 << DIRECTW; // The 0x7ff should never be accessed
    addr_c = 0;

    state_sn = state_s = state_s0 = state_s1 = state_s2 = state_s3 = state_s4 = state_s5 = 0;
    state_cn = state_c = state_c0 = state_c1 = state_c2 = state_c3 = state_c4 = state_c5 = 0;
    addr_r3 = addr_r4 = addr_r5 = 0;

    ///   @warning : the DIRECTW >= KINW in the dirt_aggr
    bool isFinal = in_e_strm.read();
    while (!isFinal) {
#pragma HLS dependence variable = sum pointer inter false
#pragma HLS dependence variable = cnt pointer inter false
#pragma HLS PIPELINE II = 1

        // 1)Get data's address(addr == cur_key.o_data == keys' dictionary)
        ap_uint<DIRECTW> cur_key = kin_strm.read();
        addr_c = cur_key; // 32bit to DIRECTW width convert in direct_aggr

        // 2)Get data to form input stream
        ap_int<DATINW> tmp = vin_strm.read();
        isFinal = in_e_strm.read();

        // 3)Read RAM and select the state_c based on the addr, addr0 and addr1
        if (addr_c == addr_r0) {
            state_c = state_c0;
            state_s = state_s0;
        } else if (addr_c == addr_r1) {
            state_c = state_c1;
            state_s = state_s1;
        } else if (addr_c == addr_r2) {
            state_c = state_c2;
            state_s = state_s2;
        } else if (addr_c == addr_r3) {
            state_c = state_c3;
            state_s = state_s3;
        } else if (addr_c == addr_r4) {
            state_c = state_c4;
            state_s = state_s4;
        } else if (addr_c == addr_r5) {
            state_c = state_c5;
            state_s = state_s5;
        } else {
            state_c = cnt[addr_c];
            state_s = sum[addr_c];
        }

        // 4)calculate aggr
        ap_int<DATOUTW> cur_val;
        if (op == AOP_NORML1) {
            cur_val = tmp > 0 ? (ap_int<DATOUTW>)tmp : (ap_int<DATOUTW>)-tmp;
        } else {
            cur_val = tmp;
        }

        ap_uint<1> non_zero;
        if (op == AOP_COUNTNONZEROS) {
            non_zero = tmp != 0;
        } else {
            non_zero = 1;
        }

        state_sn = state_s + cur_val;
        state_cn = state_c + non_zero;

        sum[addr_c] = state_sn;
        cnt[addr_c] = (1, state_cn);

        // 5)shift the whole data line 1 cycle for RAM content( state) and ADDRESS
        // (addr)
        state_s5 = state_s4;
        state_s4 = state_s3;
        state_s3 = state_s2;
        state_s2 = state_s1;
        state_s1 = state_s0;
        state_s0 = state_sn;

        state_c5 = state_c4;
        state_c4 = state_c3;
        state_c3 = state_c2;
        state_c2 = state_c1;
        state_c1 = state_c0;
        state_c0 = state_cn;

        addr_r5 = addr_r4;
        addr_r4 = addr_r3;
        addr_r3 = addr_r2;
        addr_r2 = addr_r1;
        addr_r1 = addr_r0;
        addr_r0 = addr_c;
    } // end while

    for (int i = 0; i < (1 << DIRECTW); i++) {
#pragma HLS PIPELINE

        ap_int<DATOUTW> sum_temp = sum[i];
        ap_int<DATOUTW + 1> temp = cnt[i];
        ap_uint<1> flag = temp[DATOUTW];
        ap_int<DATOUTW> cnt_temp = temp(DATOUTW - 1, 0);
        ap_int<DATOUTW> out;

        if (op == AOP_SUM || op == AOP_NORML1) {
            out = sum_temp;
        } else if (op == AOP_COUNT || op == AOP_COUNTNONZEROS) {
            out = cnt_temp;
        } else if (op == AOP_MEAN) {
            out = sum_temp / cnt_temp;
        }

        if (flag) {
            vout_strm.write(out);
            kout_strm.write(i);
            out_e_strm.write(0);
        } else {
        }
    }
    out_e_strm.write(1);

} // end direct_aggr_normL2

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {

/**
 * @brief Group-by aggregation with limited key width.
 *
 * This primitive is suitable for scenario in which the width of group key is limited, so that a on-chip array directly
 * addressed by the key can be created to store the aggregation value.
 * The total storage required is ``row size * (2 ^ key width)``.
 *
 * The following aggregate operators are supported:
 *  - AOP_MAX
 *  - AOP_MIN
 *  - AOP_SUM
 *  - AOP_COUNT
 *  - AOP_MEAN
 *  - AOP_VARIANCE
 *  - AOP_NORML1
 *  - AOP_NORML2
 *
 * The return value is typed the same as the input payload value.
 * \rst
 * .. CAUTION::
 *     Attention should be paid for overflow in sum or count.
 * \endrst
 * @tparam op the aggregate operator, as defined in AggregateOp enum.
 * @tparam DATINW  the width of input payload
 * @tparam DATOUTW the width of output aggr-payload
 * @tparam DIRECTW the width of input and output key
 *
 * @param vin_strm value input
 * @param in_e_strm end flag stream for input data
 * @param vout_strm value output
 * @param out_e_strm end flag stream for output data
 * @param kin_strm group-by key input
 * @param kout_strm group-by key output
 */
template <int op, int DATINW, int DATOUTW, int DIRECTW>
void directGroupAggregate(hls::stream<ap_uint<DATINW> >& vin_strm,
                          hls::stream<bool>& in_e_strm,
                          hls::stream<ap_uint<DATOUTW> >& vout_strm,
                          hls::stream<bool>& out_e_strm,
                          hls::stream<ap_uint<DIRECTW> >& kin_strm,
                          hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    if (op == AOP_MAX) {
        details::direct_aggr_max(vin_strm, in_e_strm, vout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_MIN) {
        details::direct_aggr_min(vin_strm, in_e_strm, vout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_SUM) {
        details::direct_aggr_sum(vin_strm, in_e_strm, vout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_COUNT) {
        details::direct_aggr_countone(vin_strm, in_e_strm, vout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_MEAN) {
        details::direct_aggr_avg(vin_strm, in_e_strm, vout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_VARIANCE) {
        details::direct_aggr_variance(vin_strm, in_e_strm, vout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_NORML1) {
        details::direct_aggr_normL1(vin_strm, in_e_strm, vout_strm, out_e_strm, kin_strm, kout_strm);
    } else if (op == AOP_NORML2) {
        details::direct_aggr_normL2(vin_strm, in_e_strm, vout_strm, out_e_strm, kin_strm, kout_strm);
    }

} // direct_aggregate

/**
 * @brief Group-by aggregation with limited key width, runtime programmable.
 *
 * This primitive is suitable for scenario in which the width of group key is limited, so that a on-chip array directly
 * addressed by the key can be created to store the aggregation value.
 * The total storage required is ``row size * (2 ^ key width)``.
 *
 * The following aggregate operators are supported:
 *  - AOP_MAX
 *  - AOP_MIN
 *  - AOP_SUM
 *  - AOP_COUNT
 *  - AOP_MEAN
 *  - AOP_NORM1
 *
 * The return value is typed the same as the input payload value.
 * \rst
 * .. CAUTION::
 *     Attention should be paid for overflow in sum or count.
 * \endrst
 * @tparam DATINW  the width of input payload
 * @tparam DATOUTW the width of output aggr-payload
 * @tparam DIRECTW the width of input and output key
 *
 * @param op the aggregate operator, as defined in AggregateOp enum.
 * @param vin_strm value input
 * @param in_e_strm end flag stream for input data
 * @param vout_strm value output
 * @param out_e_strm end flag stream for output data
 * @param kin_strm group-by key input
 * @param kout_strm group-by key output
 */
template <int DATINW, int DATOUTW, int DIRECTW>
void directGroupAggregate(ap_uint<32> op,
                          hls::stream<ap_uint<DATINW> >& vin_strm,
                          hls::stream<bool>& in_e_strm,
                          hls::stream<ap_uint<DATOUTW> >& vout_strm,
                          hls::stream<bool>& out_e_strm,
                          hls::stream<ap_uint<DIRECTW> >& kin_strm,
                          hls::stream<ap_uint<DIRECTW> >& kout_strm) {
#pragma HLS inline off

    ap_int<DATOUTW> sum[(1 << DIRECTW)];
#pragma HLS bind_storage variable = sum type = ram_2p impl = uram
    ap_int<DATOUTW + 1> cnt[(1 << DIRECTW)];
#pragma HLS bind_storage variable = cnt type = ram_2p impl = uram

    // initialize ram
    details::initialize_ram<DATINW, DATOUTW, DIRECTW>(op, sum, cnt);

    // calculate aggr
    if (op == AOP_MIN || op == AOP_MAX) {
        details::direct_aggr_min_max<DATINW, DATOUTW, DIRECTW>(op, cnt, vin_strm, in_e_strm, vout_strm, out_e_strm,
                                                               kin_strm, kout_strm);
    } else if (op == AOP_SUM || op == AOP_COUNT || op == AOP_COUNTNONZEROS || op == AOP_MEAN || op == AOP_NORML1) {
        details::direct_aggr_sum_cnt_mean_norm<DATINW, DATOUTW, DIRECTW>(op, sum, cnt, vin_strm, in_e_strm, vout_strm,
                                                                         out_e_strm, kin_strm, kout_strm);
    }
} // direct_aggregate

} // namespace database
} // namespace xf
#endif // XF_DATABASE_DIRECT_GROUP_AGGREGATE_HPP
