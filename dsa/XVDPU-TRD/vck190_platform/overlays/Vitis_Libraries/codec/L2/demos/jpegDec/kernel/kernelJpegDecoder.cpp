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
 * @file kernelJpegDecoder.cpp
 * @brief kernelJpegDecoder template function implementation and kernel_decoder warpper.
 *
 * This file is part of HLS algorithm library.
 */

#include "kernelJpegDecoder.hpp"
// ------------------------------------------------------------
// @brief Level 2 : kernel for jfif parser + huffman decoder + iQ_iDCT
// a.input the jpg 420/422/444 baseline file
// b.output the as the 8x8 's Column scan order YUV (0~255), like [Y*allpixels,U*0.5*allpixels, V*0.5*allpixels], and
// image infos
// c.Fault tolerance: If the picture's format is incorrect, error codes will directly end the kernel
// and wait for the input of the next image. Error codes cloud help to position at which decoding stage does the error
// occur
// d.performance: input throughput: 150MB/s~300MB/s(1symbol/clk), output 1~1.6GB/s (max 8B/clk),
// frequency 250MHz for kernel, for only huffman core 286MHz by vivado 2018.3

extern "C" void kernelJpegDecoder(ap_uint<AXI_WIDTH>* jpeg_pointer,
                                  const int size,
                                  ap_uint<64>* yuv_mcu_pointer,
                                  ap_uint<32>* infos) {
    // clang-format off
	//const uint64_t max_pix = MAX_NUM_PIX;//for 8K*8K
	const uint64_t max_pix = MAX_DEC_PIX;//for 800*800
	const uint64_t max_yuv = MAXCMP_BC * 8;//blocknum * 8 rows
	const uint64_t burst_lenth = BURST_LENTH;
#pragma HLS INTERFACE m_axi port = jpeg_pointer     depth = 65000 offset = slave bundle = gmem_in0 \
					  latency = 64 num_read_outstanding = 32 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = yuv_mcu_pointer 	depth = 230400 offset = slave bundle = gmem_in1 \
					  latency = 64 num_write_outstanding = 32 max_write_burst_length = 32
#pragma HLS INTERFACE m_axi port = infos 			depth = 1024   offset = slave bundle = gmem_in2 \
					  latency = 64 num_write_outstanding = 32 max_write_burst_length = 32
	#pragma HLS INTERFACE s_axilite port=jpeg_pointer      	bundle=control
	#pragma HLS INTERFACE s_axilite port=yuv_mcu_pointer    bundle=control
	#pragma HLS INTERFACE s_axilite port=size      			bundle=control
	#pragma HLS INTERFACE s_axilite port=infos    		    bundle=control
	#pragma HLS INTERFACE s_axilite port=return         	bundle=control

	xf::codec::kernelJpegDecoderTop(jpeg_pointer, size, yuv_mcu_pointer, infos);
}
