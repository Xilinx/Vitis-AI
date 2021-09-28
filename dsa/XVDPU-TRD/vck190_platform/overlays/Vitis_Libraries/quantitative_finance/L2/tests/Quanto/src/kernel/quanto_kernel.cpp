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
 * @file quanto_kernel.cpp
 * @brief HLS implementation of the Quanto kernel which parallelizes the single
 * closed-form solver
 */

#include <ap_fixed.h>
#include <hls_stream.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "bus_interface.hpp"
#include "hls_math.h"
#include "xf_fintech/cf_bsm.hpp"

/// @brief Specific implementation of this kernel
///
#define DT float
#define DT_EQ_INT uint32_t
#define NUM_KERNELS 2
#define BUS_WIDTH (NUM_KERNELS*sizeof(DT)*8)

// Create a type which contains as many streams as we have kernels and a stream
// thereof
typedef struct WideDataType { DT data[NUM_KERNELS]; } WideDataType;
typedef hls::stream<WideDataType> WideStreamType;

extern "C" {

/// @brief Wrapper closed-form solver to process in and out streams
/// @param[in]  s_stream     Stream of containing parallel input parameters
/// @param[in]  v_stream     Stream of containing parallel input parameters
/// @param[in]  rd_stream    Stream of containing parallel input parameters
/// @param[in]  t_stream     Stream of containing parallel input parameters
/// @param[in]  k_stream     Stream of containing parallel input parameters
/// @param[in]  rf_stream    Stream of containing parallel input parameters
/// @param[in]  q_stream     Stream of containing parallel input parameters
/// @param[in]  E_stream     Stream of containing parallel input parameters
/// @param[in]  fxv_stream   Stream of containing parallel input parameters
/// @param[in]  corr_stream  Stream of containing parallel input parameters
/// @param[in]  call         Controls whether call or put is calculated
/// @param[in]  size         Total number of input data sets to process
/// @param[out] price_stream Stream of containing parallel BSM price
/// @param[out] delta_stream Stream of containing parallel BSM Greeks
/// @param[out] gamma_stream Stream of containing parallel BSM Greeks
/// @param[out] vega_stream  Stream of containing parallel BSM Greeks
/// @param[out] theta_stream Stream of containing parallel BSM Greeks
/// @param[out] rho_stream   Stream of containing parallel BSM Greeks
void quanto_stream_wrapper(WideStreamType& s_stream,
                           WideStreamType& v_stream,
                           WideStreamType& rd_stream,
                           WideStreamType& t_stream,
                           WideStreamType& k_stream,
                           WideStreamType& rf_stream,
                           WideStreamType& q_stream,
                           WideStreamType& E_stream,
                           WideStreamType& fxv_stream,
                           WideStreamType& corr_stream,
                           unsigned int call,
                           unsigned int size,
                           WideStreamType& price_stream,
                           WideStreamType& delta_stream,
                           WideStreamType& gamma_stream,
                           WideStreamType& vega_stream,
                           WideStreamType& theta_stream,
                           WideStreamType& rho_stream) {
    for (unsigned int i = 0; i < size; i += NUM_KERNELS) {
        WideDataType s, v, rd, t, k, rf, q, E, fxv, corr, price, delta, gamma, vega, theta, rho;
#pragma HLS PIPELINE II = 1

        // This will read NUM_KERNEL's worth of streams
        s = s_stream.read();
        v = v_stream.read();
        rd = rd_stream.read();
        t = t_stream.read();
        k = k_stream.read();
        rf = rf_stream.read();
        q = q_stream.read();
        E = E_stream.read();
        fxv = fxv_stream.read();
        corr = corr_stream.read();

    parallel_quanto:
        for (unsigned int j = 0; j < NUM_KERNELS; ++j) {
#pragma HLS UNROLL
            DT qq = -rf.data[j] + rd.data[j] + q.data[j] + (corr.data[j] * v.data[j] * fxv.data[j]);
            xf::fintech::cfBSMEngine<DT>(s.data[j], v.data[j], rd.data[j], t.data[j], k.data[j], qq, call,
                                         &(price.data[j]), &(delta.data[j]), &(gamma.data[j]), &(vega.data[j]),
                                         &(theta.data[j]), &(rho.data[j]));
            price.data[j] = price.data[j] * E.data[j];
        }

        price_stream.write(price);
        delta_stream.write(delta);
        gamma_stream.write(gamma);
        vega_stream.write(vega);
        theta_stream.write(theta);
        rho_stream.write(rho);
    }
}

/// @brief Kernel top level
///
/// This is the top level kernel and represents the interface presented to the
/// host.
///
/// @param[in]  s_in      Input parameters read as a vector bus type
/// @param[in]  v_in      Input parameters read as a vector bus type
/// @param[in]  rd_in     Input parameters read as a vector bus type
/// @param[in]  t_in      Input parameters read as a vector bus type
/// @param[in]  k_in      Input parameters read as a vector bus type
/// @param[in]  rf_in     Input parameters read as a vector bus type
/// @param[in]  q_in      Input parameters read as a vector bus type
/// @param[in]  E_in      Input parameters read as a vector bus type
/// @param[in]  fxv_in    Input parameters read as a vector bus type
/// @param[in]  corr_in   Input parameters read as a vector bus type
/// @param[in]  call      Controls whether call or put is calculated
/// @param[in]  num       Total number of input data sets to process
/// @param[out] price_out Output parameters read as a vector bus type
/// @param[out] delta_out Output parameters read as a vector bus type
/// @param[out] gamma_out Output parameters read as a vector bus type
/// @param[out] vega_out  Output parameters read as a vector bus type
/// @param[out] theta_out Output parameters read as a vector bus type
/// @param[out] rho_out   Output parameters read as a vector bus type
void quanto_kernel(ap_uint<BUS_WIDTH>* s_in,
                   ap_uint<BUS_WIDTH>* v_in,
                   ap_uint<BUS_WIDTH>* rd_in,
                   ap_uint<BUS_WIDTH>* t_in,
                   ap_uint<BUS_WIDTH>* k_in,
                   ap_uint<BUS_WIDTH>* rf_in,
                   ap_uint<BUS_WIDTH>* q_in,
                   ap_uint<BUS_WIDTH>* E_in,
                   ap_uint<BUS_WIDTH>* fxv_in,
                   ap_uint<BUS_WIDTH>* corr_in,
                   unsigned int call,
                   unsigned int num,
                   ap_uint<BUS_WIDTH>* price_out,
                   ap_uint<BUS_WIDTH>* delta_out,
                   ap_uint<BUS_WIDTH>* gamma_out,
                   ap_uint<BUS_WIDTH>* vega_out,
                   ap_uint<BUS_WIDTH>* theta_out,
                   ap_uint<BUS_WIDTH>* rho_out) {
/// @brief Define the AXI parameters.  Each input/output parameter has a
/// separate port
#pragma HLS INTERFACE m_axi port = s_in offset = slave bundle = in0_port
#pragma HLS INTERFACE m_axi port = v_in offset = slave bundle = in1_port
#pragma HLS INTERFACE m_axi port = rd_in offset = slave bundle = in2_port
#pragma HLS INTERFACE m_axi port = t_in offset = slave bundle = in3_port
#pragma HLS INTERFACE m_axi port = k_in offset = slave bundle = in4_port
#pragma HLS INTERFACE m_axi port = rf_in offset = slave bundle = in5_port
#pragma HLS INTERFACE m_axi port = q_in offset = slave bundle = in6_port
#pragma HLS INTERFACE m_axi port = E_in offset = slave bundle = in7_port
#pragma HLS INTERFACE m_axi port = fxv_in offset = slave bundle = in8_port
#pragma HLS INTERFACE m_axi port = corr_in offset = slave bundle = in9_port
#pragma HLS INTERFACE m_axi port = price_out offset = slave bundle = out0_port
#pragma HLS INTERFACE m_axi port = delta_out offset = slave bundle = out1_port
#pragma HLS INTERFACE m_axi port = gamma_out offset = slave bundle = out2_port
#pragma HLS INTERFACE m_axi port = vega_out offset = slave bundle = out3_port
#pragma HLS INTERFACE m_axi port = theta_out offset = slave bundle = out4_port
#pragma HLS INTERFACE m_axi port = rho_out offset = slave bundle = out5_port

#pragma HLS INTERFACE s_axilite port = s_in bundle = control
#pragma HLS INTERFACE s_axilite port = v_in bundle = control
#pragma HLS INTERFACE s_axilite port = rd_in bundle = control
#pragma HLS INTERFACE s_axilite port = t_in bundle = control
#pragma HLS INTERFACE s_axilite port = k_in bundle = control
#pragma HLS INTERFACE s_axilite port = rf_in bundle = control
#pragma HLS INTERFACE s_axilite port = q_in bundle = control
#pragma HLS INTERFACE s_axilite port = E_in bundle = control
#pragma HLS INTERFACE s_axilite port = fxv_in bundle = control
#pragma HLS INTERFACE s_axilite port = corr_in bundle = control
#pragma HLS INTERFACE s_axilite port = price_out bundle = control
#pragma HLS INTERFACE s_axilite port = delta_out bundle = control
#pragma HLS INTERFACE s_axilite port = gamma_out bundle = control
#pragma HLS INTERFACE s_axilite port = vega_out bundle = control
#pragma HLS INTERFACE s_axilite port = theta_out bundle = control
#pragma HLS INTERFACE s_axilite port = rho_out bundle = control

#pragma HLS INTERFACE s_axilite port = call bundle = control
#pragma HLS INTERFACE s_axilite port = num bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    WideStreamType s_stream("s_stream");
    WideStreamType v_stream("v_stream");
    WideStreamType rd_stream("rd_stream");
    WideStreamType t_stream("t_stream");
    WideStreamType k_stream("k_stream");
    WideStreamType rf_stream("rf_stream");
    WideStreamType q_stream("q_stream");
    WideStreamType E_stream("E_stream");
    WideStreamType fxv_stream("fxv_stream");
    WideStreamType corr_stream("corr_stream");

    WideStreamType price_stream("price_stream");
    WideStreamType delta_stream("delta_stream");
    WideStreamType gamma_stream("gamma_stream");
    WideStreamType vega_stream("vega_stream");
    WideStreamType theta_stream("theta_stream");
    WideStreamType rho_stream("row_stream");

#pragma HLS STREAM variable = s_stream depth = 32
#pragma HLS STREAM variable = v_stream depth = 32
#pragma HLS STREAM variable = rd_stream depth = 32
#pragma HLS STREAM variable = t_stream depth = 32
#pragma HLS STREAM variable = k_stream depth = 32
#pragma HLS STREAM variable = rf_stream depth = 32
#pragma HLS STREAM variable = q_stream depth = 32
#pragma HLS STREAM variable = E_stream depth = 32
#pragma HLS STREAM variable = fxv_stream depth = 32
#pragma HLS STREAM variable = corr_stream depth = 32
#pragma HLS STREAM variable = price_stream depth = 32
#pragma HLS STREAM variable = delta_stream depth = 32
#pragma HLS STREAM variable = gamma_stream depth = 32
#pragma HLS STREAM variable = vega_stream depth = 32
#pragma HLS STREAM variable = theta_stream depth = 32
#pragma HLS STREAM variable = rho_stream depth = 32

    unsigned int vector_size = BUS_WIDTH / (8 * sizeof(DT));
    unsigned int ddr_words = num / vector_size;

// Run the whole following region as data flow
#pragma HLS dataflow

    // Convert the bus (here DDR BUS_WIDTH bits) into a number of parallel streams
    // according to NUM_KERNELS
    bus_to_stream<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(s_in, s_stream, ddr_words);
    bus_to_stream<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(v_in, v_stream, ddr_words);
    bus_to_stream<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(rd_in, rd_stream, ddr_words);
    bus_to_stream<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(t_in, t_stream, ddr_words);
    bus_to_stream<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(k_in, k_stream, ddr_words);
    bus_to_stream<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(rf_in, rf_stream, ddr_words);
    bus_to_stream<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(q_in, q_stream, ddr_words);
    bus_to_stream<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(E_in, E_stream, ddr_words);
    bus_to_stream<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(fxv_in, fxv_stream, ddr_words);
    bus_to_stream<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(corr_in, corr_stream, ddr_words);

    // This wrapper takes in the parallel streams and processes them using
    // NUM_KERNELS separate kernels
    quanto_stream_wrapper(s_stream, v_stream, rd_stream, t_stream, k_stream, rf_stream, q_stream, E_stream, fxv_stream,
                          corr_stream, call, num, price_stream, delta_stream, gamma_stream, vega_stream, theta_stream,
                          rho_stream);

    // Convert the NUM_KERNELS streams back to the wide data bus
    stream_to_bus<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(price_stream, price_out,
                                                                                       ddr_words);
    stream_to_bus<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(delta_stream, delta_out,
                                                                                       ddr_words);
    stream_to_bus<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(gamma_stream, gamma_out,
                                                                                       ddr_words);
    stream_to_bus<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(vega_stream, vega_out,
                                                                                       ddr_words);
    stream_to_bus<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(theta_stream, theta_out,
                                                                                       ddr_words);
    stream_to_bus<DT, DT_EQ_INT, WideDataType, WideStreamType, BUS_WIDTH, NUM_KERNELS>(rho_stream, rho_out, ddr_words);
}
} // extern C
