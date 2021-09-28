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
 * @file kernelJpegDecoder.hpp
 * @brief kernelJpegDecoder template function implementation and kernel_decoder warpper.
 *
 * This file is part of HLS algorithm library.
 */

#include "XAcc_jpegdecoder.hpp"
#include "XAcc_jfifparser.hpp"
#include "XAcc_idct.hpp"
// ------------------------------------------------------------
/**
 * @brief Level 2 : kernel for jfif parser + huffman decoder + iQ_iDCT
 *
 * @tparam CH_W size of data path in dataflow region, in bit.
 *         when CH_W is 16, the decoder could decode one symbol per cycle in about 99% cases.
 *         when CH_W is 8 , the decoder could decode one symbol per cycle in about 80% cases, but use less resource.
 *
 * @param jpeg_pointer the input jpeg to be read from DDR.
 * @param size the total bytes to be read from DDR.
 * @param yuv_mcu_pointer the output yuv to DDR in mcu order.
 * @param info information of the image, maybe use in the recovery image.
 */
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
                                  ap_uint<32>* infos);
