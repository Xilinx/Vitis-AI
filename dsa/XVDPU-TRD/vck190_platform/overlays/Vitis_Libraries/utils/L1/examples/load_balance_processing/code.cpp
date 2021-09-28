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

#include "code.hpp"

#include "xf_utils_hw/stream_one_to_n.hpp"
#include "xf_utils_hw/stream_n_to_one.hpp"

// extract the meaningful data from the input data, and updata it.
ap_uint<W_PU> update_data(ap_uint<W_PU> data) {
#pragma HLS inline
    ap_uint<W_PRC> p = data.range(W_PRC - 1, 0);
    ap_uint<W_DSC> d = data.range(W_PRC + W_DSC - 1, W_PRC);
    ap_uint<W_PU> nd = 0;
    nd.range(W_PRC - 1, 0) = p * 2;
    nd.range(W_DSC + W_PRC - 1, W_PRC) = d + 2;
    return nd;
}

/**
 * @brief update each data as output
 * read and write
 * @param c_istrm input stream
 * @param e_c_istrm end flag for input stream
 * @param c_ostrm output stream
 * @param e_c_ostrm end flag for output stream
 *
 */
void process_core_pass(hls::stream<ap_uint<W_PU> >& c_istrm,
                       hls::stream<bool>& e_c_istrm,
                       hls::stream<ap_uint<W_PU> >& c_ostrm,
                       hls::stream<bool>& e_c_ostrm) {
    bool last = e_c_istrm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        bool em = c_istrm.empty();
        if (false == em) {
            ap_uint<W_PU> d = c_istrm.read();
            ap_uint<W_PU> nd = update_data(d);
            c_ostrm.write(nd);
            e_c_ostrm.write(false);
            last = e_c_istrm.read();
        }
    } // while

    e_c_ostrm.write(true);
}
/**
 * @brief  update each data as output, but work intermittently.
 * @param f_sw flag for work first or sleep, work if it is false, sleep if ture.
 * @param prd the period of work
 * @param c_istrm input stream
 * @param e_c_istrm end flag for input stream
 * @param c_ostrm output stream
 * @param e_c_ostrm end flag for output stream
 *
 */
void process_core_intermission(bool f_sw,
                               int prd,
                               hls::stream<ap_uint<W_PU> >& c_istrm,
                               hls::stream<bool>& e_c_istrm,
                               hls::stream<ap_uint<W_PU> >& c_ostrm,
                               hls::stream<bool>& e_c_ostrm) {
    /*
     *****************************************************************
     * for example, an ideal case as
     * when f_sw = true, prd =4
     *
     *    sleep    --   work   --   sleep    --   work     ...
     *   4 cycles      4 cycles    4 cycles     4 cycles
     *
     *
     * when f_sw = false, prd =4
     *
     *     work    --   sleep   --   work    --   sleep     ...
     *   4 cycles      4 cycles    4 cycles     4 cycles
     ******************************************************************/
    int c = 0;
    bool sw = f_sw;
    bool last = e_c_istrm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        bool em = c_istrm.empty();
        if (false == sw && false == em) {
            // work
            ap_uint<W_PU> d = c_istrm.read();
            ap_uint<W_PU> nd = update_data(d);
            c_ostrm.write(nd);
            e_c_ostrm.write(false);
            last = e_c_istrm.read();
        } else {
            // sleep
        } // if - else
        if (++c == prd) {
            sw = !sw;
            c = 0;
        } // if
    }     // while
    e_c_ostrm.write(true);
}

/**
 * @brief Multiple  PUs work in parallel
 * Some work and others sleep at the same time.
 * @param c_istrms input streams
 * @param e_c_istrms end flag for input streams
 * @param c_ostrms output stream
 * @param e_c_ostrms end flag for output streams
 *
 */
void process_mpu(hls::stream<ap_uint<W_PU> > c_istrms[NPU],
                 hls::stream<bool> e_c_istrms[NPU],
                 hls::stream<ap_uint<W_PU> > c_ostrms[NPU],
                 hls::stream<bool> e_c_ostrms[NPU]) {
/*
 * Assume NPU = 16.
 * All PUs work in parellel at an ideal case as belows:
 *  PU0   ------------------------------------
 *  PU1   ------------------------------------
 *  PU2   --  --  --  --  --  --  --  --  --
 *  PU3     --  --  --  --  --  --  --  --  --
 *  PU4   ----    ----    ----    ----    ----
 *  PU5       ----    ----    ----    ----
 *  PU6       ----    ----    ----    ----
 *  PU7       ----    ----    ----    ----
 *  PU8   --------        --------        ----
 *  PU9           --------        --------
 *  PU10          --------        --------
 *  PU11          --------        --------
 *  PU12  --------        --------        ----
 *  PU13          --------        --------
 *  PU14          --------        --------
 *  PU15          --------        --------
 *
 * Here, the mark(-) stands for work and blank does sleep.
 *
 */
#pragma HLS dataflow
    // PU0 and PU1 are always working.
    for (int i = 0; i < 2; ++i) {
#pragma HLS unroll
        process_core_pass(c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
    }
    // The other PUs work at sometimes
    for (int i = 2; i < NPU; ++i) {
#pragma HLS unroll
        if (i < 4)
            process_core_intermission(i % 2 == 0, 2, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
        else if (i < 8)
            process_core_intermission(i % 4 == 0, 4, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
        else
            process_core_intermission(i % 4 == 0, 8, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
    }
}

/**
 * @brief Simutlate that a big task is coumputed by Mutiple Process Units.
 * Assume each input data is a package which could be splitted to a few of small width data, and each small data is
 *processed by a Process Uint(PU)
 * The PUs work at different speeds, so it is important to dispatch data to PUs and collect data from PUs. Here,
 *distribution on load balance is used.
 *
 * @param istrm input stream
 * @param e_istrm end flag for input stream
 * @param ostrm input stream
 * @param e_ostrm end flag for output stream
 **/
void test_core(hls::stream<ap_uint<W_STRM> >& istrm,
               hls::stream<bool>& e_istrm,
               hls::stream<ap_uint<W_STRM> >& ostrm,
               hls::stream<bool>& e_ostrm) {
/*
 * One input stream(istrm) is splitted to multitple streams, and each services a PU.
 * All output streams from PUs are merged to one stream(ostrm).
 * Here, the speeds of PUs are not same.
 *
 * For example, there are 8 PUs, like this:
 *
 *              split           merge
 *              1-->8           8-->1
 *
 *                |----> PU0 ---->|
 *                |               |
 *                |----> PU1 ---->|
 *                |               |
 *                |----> PU2 ---->|
 *                |               |
 *                |----> PU3 ---->|
 * istrm  ----->  |               |-----> ostrm
 *                |----> PU4 ---->|
 *                |               |
 *                |----> PU5 ---->|
 *                |               |
 *                |----> PU6 ---->|
 *                |               |
 *                |----> PU7 ---->|
 *
 */

/*       one to n                     PUs                   n to one
* istrm ---------> data_inner_strms -------> new_data_strms ----------> ostrms
*
*/

#pragma HLS dataflow

    hls::stream<ap_uint<W_PU> > data_inner_strms[NPU];
#pragma HLS stream variable = data_inner_strms depth = 8
    hls::stream<bool> e_data_inner_strms[NPU];
#pragma HLS stream variable = e_data_inner_strms depth = 8

    hls::stream<ap_uint<W_PU> > new_data_strms[NPU];
#pragma HLS stream variable = new_data_strms depth = 8
    hls::stream<bool> e_new_data_strms[NPU];
#pragma HLS stream variable = e_new_data_strms depth = 8

    xf::common::utils_hw::streamOneToN<W_STRM, W_PU, NPU>(istrm, e_istrm, data_inner_strms, e_data_inner_strms,
                                                          xf::common::utils_hw::LoadBalanceT());

    process_mpu(data_inner_strms, e_data_inner_strms, new_data_strms, e_new_data_strms);

    xf::common::utils_hw::streamNToOne<W_PU, W_STRM, NPU>(new_data_strms, e_new_data_strms, ostrm, e_ostrm,
                                                          xf::common::utils_hw::LoadBalanceT());
}
