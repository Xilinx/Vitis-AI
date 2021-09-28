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

#include "xf_utils_hw/axi_to_stream.hpp"
#include "xf_utils_hw/stream_one_to_n.hpp"
#include "xf_utils_hw/stream_n_to_one.hpp"
#include "xf_utils_hw/stream_to_axi.hpp"

/**
 * @brief extract the meaningful data from the input data, and updata it.
 * @param data  input data
 * @return updated data
 */
ap_uint<W_PU> update_data(ap_uint<W_PU> data) {
#pragma HLS inline
    ap_uint<W_PRC> p = data.range(W_PRC - 1, 0);
    ap_uint<W_DSC> d = data.range(W_PRC + W_DSC - 1, W_PRC);
    ap_uint<W_PU> nd = 0;
    nd.range(W_PRC - 1, 0) = p * 2;
    nd.range(W_DSC + W_PRC - 1, W_PRC) = d + 2;
    return nd;
}
// extract the meaningful data from the input data, then calculate.
ap_uint<W_PU> calculate(ap_uint<W_PU> data) {
#pragma HLS inline
    ap_uint<W_PU> p = data.range(W_PRC - 1, 0);
    ap_uint<W_PU> d = data.range(W_PRC + W_DSC - 1, 0);
    ap_uint<W_PU> nd = p * d;
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
        // int i = k;// + offset;
        process_core_pass(c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
    }
    // The other PUs work at sometimes
    for (int i = 2; i < NPU; ++i) {
#pragma HLS unroll
        int k = i;
        if (k < 4)
            process_core_intermission(k % 2 == 0, 2, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
        else if (k < 8)
            process_core_intermission(k % 4 == 0, 4, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
        else
            process_core_intermission(k % 4 == 0, 8, c_istrms[i], e_c_istrms[i], c_ostrms[i], e_c_ostrms[i]);
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
void update_mpu(hls::stream<ap_uint<W_STRM> >& istrm,
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
                                                          //   xf::common::utils_hw::round_robin_t());
                                                          xf::common::utils_hw::LoadBalanceT());

    process_mpu(data_inner_strms, e_data_inner_strms, new_data_strms, e_new_data_strms);

    xf::common::utils_hw::streamNToOne<W_PU, W_STRM, NPU>(
        new_data_strms, e_new_data_strms, ostrm, e_ostrm,
        //                        xf::common::utils_hw::round_robin_t());
        xf::common::utils_hw::LoadBalanceT());
}

// ------------------------------------------------------------
// top functions
/**
 * @brief Update data
 * A few of data are packeged to a wide width data which is tranferred by axi-port. Extract and update each data from
 * the wide width data.
 * For example, 8 32-bit data are combined to a 256-bit data. Each 32-bit data is updated and output in the same formart
 * as input.
 * Here, each W_AXI bits data in in_buf includes multiple data(ap_uint<W_DATA>) which will be updated.
 *
 * @param in_buf the input buffer
 * @param out_buf the output buffer
 * @param len the number of input data in in_buf
 *
 */
void top_core(ap_uint<W_AXI>* in_buf, ap_uint<W_AXI>* out_buf, const int len) {
#pragma HLS INTERFACE m_axi port = in_buf depth = DDR_DEPTH offset = slave bundle = gmem_in0 latency = \
    8 num_read_outstanding = 32 max_read_burst_length = 32

#pragma HLS INTERFACE s_axilite port = in_buf bundle = control

#pragma HLS INTERFACE m_axi port = out_buf depth = DDR_DEPTH offset = slave bundle = gmem_out1 latency = \
    8 num_write_outstanding = 32 max_write_burst_length = 32

#pragma HLS INTERFACE s_axilite port = out_buf bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS dataflow
    hls::stream<t_strm> axi_istrm;
#pragma HLS stream variable = axi_istrm depth = 8
    hls::stream<bool> e_axi_istrm;
#pragma HLS stream variable = e_axi_istrm depth = 8

    hls::stream<t_strm> axi_ostrm;
#pragma HLS stream variable = axi_ostrm depth = 8
    hls::stream<bool> e_axi_ostrm;
#pragma HLS stream variable = e_axi_ostrm depth = 8

    /*
     ----------------         --------------------------------------------------------         -------------------
    |                |       |                                                        |       |                   |
    |                |       |                                                        |       |                   |
    | axi to stream  |  -->  | stream to n streams   --> MPU  ---> n streams to one   |  -->  |  stream to axi    |
    |                |       |                                                        |       |                   |
    |                |       |                                                        |       |                   |
     ----------------         ---------------------------------------------------------        -------------------
    */

    // axi --> stream --> compute  --> stream --> axi
    // in_buf        axi_port     inner_stream   axi_port   out_buf
    // W_DATA ------> W_AXI ------> W_STRM -----> W_AXI ---> W_DATA

    // axi to stream
    // in_buf --> axi_istrm
    xf::common::utils_hw::axiToStream<BURST_LENTH, W_AXI, t_strm>(in_buf, len, axi_istrm, e_axi_istrm);

    // compute by mutiple process uinits
    // axi_istrm --> axi_ostrm
    update_mpu(axi_istrm, e_axi_istrm, axi_ostrm, e_axi_ostrm);

    // stream to axi
    // axi_ostrm --> out_buf
    xf::common::utils_hw::streamToAxi<BURST_LENTH, W_AXI, W_STRM>(out_buf, axi_ostrm, e_axi_ostrm);
}
