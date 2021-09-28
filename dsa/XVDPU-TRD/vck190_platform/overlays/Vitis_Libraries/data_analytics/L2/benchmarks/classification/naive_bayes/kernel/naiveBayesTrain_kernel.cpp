
#include "xf_data_analytics/classification/naive_bayes.hpp"
#include "naiveBayesTrain_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

template <int PU>
void scan(ap_uint<512>* buf_in, hls::stream<ap_uint<64> > o_data_strm[PU], hls::stream<bool> o_e_strm[PU]) {
    const int num = buf_in[0](31, 0);
    for (int i = 0; i < num; i++) {
#pragma HLS pipeline II = 1
        ap_uint<512> t = buf_in[i + 1];
        for (int j = 0; j < PU; j++) {
#pragma HLS unroll
            ap_uint<64> m = t(64 * j + 63, 64 * j);
            o_data_strm[j].write(m);
            o_e_strm[j].write(false);
        }
    }

    for (int j = 0; j < PU; j++) {
#pragma HLS unroll
        o_e_strm[j].write(true);
    }
}

template <int PU>
void write(ap_uint<512>* buf_out0,
           ap_uint<512>* buf_out1,
           const int num_of_class,

           hls::stream<int>& i_terms_strm,
           hls::stream<ap_uint<64> > i_data0_strm[PU],
           hls::stream<ap_uint<64> > i_data1_strm[PU]) {
    const int num_of_terms = i_terms_strm.read();
    const int n0 = ((num_of_terms + PU - 1) / PU) * num_of_class;
    const int n1 = (num_of_class + PU - 1) / PU;

    ap_uint<512> t0, t1;
    t0(31, 0) = num_of_terms;
    t0(63, 32) = n0;
    t1(31, 0) = num_of_class;
    t1(63, 32) = n1;
    buf_out0[0] = t0;
    buf_out1[0] = t1;
#ifndef __SYNTHESIS__
    std::cout << "n0:" << n0 << ", n1:" << n1 << std::endl;
#endif

    for (int i = 0; i < n0; i++) {
#pragma HLS pipeline II = 1
        ap_uint<512> t = 0;
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            t(64 * p + 63, 64 * p) = i_data0_strm[p].read();
        }

        buf_out0[i + 1] = t;
    }

    for (int i = 0; i < n1; i++) {
#pragma HLS pipeline II = 1
        ap_uint<512> t = 0;
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            t(64 * p + 63, 64 * p) = i_data1_strm[p].read();
        }

        buf_out1[i + 1] = t;
    }
}

extern "C" void naiveBayesTrain_kernel(const int num_of_class,
                                       const int num_of_terms,
                                       ap_uint<512>* buf_in,
                                       ap_uint<512>* buf_out0,
                                       ap_uint<512>* buf_out1) {
// clang-format off
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	   num_write_outstanding = 2 num_read_outstanding = 32 \
	   max_write_burst_length = 2 max_read_burst_length = 32 \
	   bundle = gmem0_0 port = buf_in

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	   num_write_outstanding = 32 num_read_outstanding = 2 \
	   max_write_burst_length = 32 max_read_burst_length = 2 \
	   bundle = gmem0_1 port = buf_out0

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	   num_write_outstanding = 32 num_read_outstanding = 2 \
	   max_write_burst_length = 32 max_read_burst_length = 2 \
	   bundle = gmem0_2 port = buf_out1

#pragma HLS INTERFACE s_axilite port = num_of_class bundle = control
#pragma HLS INTERFACE s_axilite port = num_of_terms bundle = control
#pragma HLS INTERFACE s_axilite port = buf_in bundle = control
#pragma HLS INTERFACE s_axilite port = buf_out0 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_out1 bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
// clang-format on

#pragma HLS INLINE off
#pragma HLS DATAFLOW

    const int WF = 64;
    const int PU = 512 / WF;
    const int WL = 3;

    hls::stream<ap_uint<64> > scan_data_array[PU];
#pragma HLS stream variable = scan_data_array depth = 64
    hls::stream<bool> scan_e_array[PU];
#pragma HLS stream variable = scan_e_array depth = 64

    hls::stream<int> terms_strm;
#pragma HLS stream variable = terms_strm depth = 2
    hls::stream<ap_uint<64> > mnnb_d0_strm[PU];
#pragma HLS stream variable = mnnb_d0_strm depth = 32
    hls::stream<ap_uint<64> > mnnb_d1_strm[PU];
#pragma HLS stream variable = mnnb_d1_strm depth = 32

    scan<PU>(buf_in, scan_data_array, scan_e_array);

    xf::data_analytics::classification::naiveBayesTrain<32, WL, unsigned int>(
        num_of_class, num_of_terms, scan_data_array, scan_e_array, terms_strm, mnnb_d0_strm, mnnb_d1_strm);

    write<PU>(buf_out0, buf_out1, num_of_class, terms_strm, mnnb_d0_strm, mnnb_d1_strm);
}
