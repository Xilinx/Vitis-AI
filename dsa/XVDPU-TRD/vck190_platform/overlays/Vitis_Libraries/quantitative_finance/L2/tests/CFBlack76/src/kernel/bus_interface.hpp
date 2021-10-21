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
 * @file bus_interface.hpp
 * @brief Templated functions to convert vector bus into parallel HLS streams
 */

#ifndef _XF_FINTECH_BUS_INTERFACE_HPP_
#define _XF_FINTECH_BUS_INTERFACE_HPP_

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

/// @brief Converts a vector of input values into parallel streams
///
/// For maximum data bandwidth utilization the data is packed into a vector to
/// fill the full data width of the bus.
/// In the case of a DDR data medium, the bus is 512-bits wide and can hold 16
/// floats or 8 doubles.  This function will
/// demux this vector into a compile time controlled number of streams
/// (typically to match the number of processing
/// engines which comprise the core of the kernel).
///
/// @tparam     DT                Data type (float/double) of the parameter
/// packed into the vector bus
/// @tparam     DT_INT_EQUIVALENT Equivalently sized integer type of DT
/// @tparam     WDT               Wide Data Type - the container for the
/// parallel parameters
/// @tparam     WST               Wide Stream Type - the stream container of the
/// WDT
/// @tparam     BUS_WIDTH         Size of bus in bits
/// @tparam     NUM_STREAMS       Number of parallel streams to construct
/// (matches size of the WDT, WST)
/// @param[in]  in                Pointer to an address containing the vector
/// data (must be correctly aligned)
/// @param[out] in_stream         Stream representation of this input data
/// @param[in]  size              Number of vector reads to make
template <typename DT,
          typename DT_INT_EQUIVALENT,
          typename WDT,
          typename WST,
          unsigned int BUS_WIDTH,
          unsigned int NUM_STREAMS>
void bus_to_stream(ap_uint<BUS_WIDTH>* in, WST& in_stream, unsigned int size) {
    unsigned int bits_per_data_type = 8 * sizeof(DT);
    unsigned int vector_words = BUS_WIDTH / bits_per_data_type;

mem_rd:
    for (unsigned int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1

        ap_uint<BUS_WIDTH> temp0 = in[i];
        DT_INT_EQUIVALENT temp1 = 0;
        WDT temp2;

    mem_rd_vector:
        for (unsigned int j = 0; j < vector_words; j += NUM_STREAMS) {
#pragma HLS ARRAY_PARTITION variable = temp2 complete
        mem_rd_per_stream:
            for (unsigned int k = 0; k < NUM_STREAMS; k++) {
#pragma HLS UNROLL
                temp1 = temp0.range(bits_per_data_type * (j + k + 1) - 1, bits_per_data_type * (j + k));
                temp2.data[k] = *(DT*)(&temp1);
            }
            in_stream.write(temp2);
        }
    }
}

/// @brief Converts parallel streams into vector of output values
///
/// For maximum data bandwidth utilization the data is packed into a vector to
/// fill the full data width of the bus.
/// In the case of a DDR data medium, the bus is 512-bits wide and can hold 16
/// floats or 8 doubles.  This function will
/// take a compile time controlled number of streams (typically to match the
/// number of processing engines which
/// comprise the core of the kernel) and muxes them into the vector bus.
///
/// @tparam     DT                Data type (float/double) of the parameter
/// packed into the vector bus
/// @tparam     DT_INT_EQUIVALENT Equivalently sized integer type of DT
/// @tparam     WDT               Wide Data Type - the container for the
/// parallel parameters
/// @tparam     WST               Wide Stream Type - the stream container of the
/// WDT
/// @tparam     BUS_WIDTH         Size of bus in bits (eg for DDR -> 512)
/// @tparam     NUM_STREAMS       Number of parallel streams to construct
/// (matches size of the WDT, WST)
/// @param[in]  out_stream        Stream representation of data to be written to
/// bus
/// @param[out] out               Pointer to an address to write the vector data
/// (must be correctly aligned)
/// @param[in]  size              Number of vector writes to make
template <typename DT,
          typename DT_INT_EQUIVALENT,
          typename WDT,
          typename WST,
          unsigned int BUS_WIDTH,
          unsigned int NUM_STREAMS>
void stream_to_bus(WST& out_stream, ap_uint<BUS_WIDTH>* out, unsigned int size) {
    unsigned int bits_per_data_type = 8 * sizeof(DT);
    unsigned int vector_words = BUS_WIDTH / bits_per_data_type;

mem_wr:
    for (unsigned int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II = 1

        DT temp0 = 0.0f;
        ap_uint<BUS_WIDTH> temp1 = 0;
        WDT temp2;

    mem_wr_vector:
        for (unsigned int j = 0; j < vector_words; j += NUM_STREAMS) {
#pragma HLS ARRAY_PARTITION variable = temp2 complete
            temp2 = out_stream.read();
        mem_wr_per_kernel:
            for (unsigned int k = 0; k < NUM_STREAMS; k++) {
#pragma HLS UNROLL
                temp0 = temp2.data[k];
                temp1.range(bits_per_data_type * (j + k + 1) - 1, bits_per_data_type * (j + k)) =
                    *(DT_INT_EQUIVALENT*)(&temp0);
            }
        }
        out[i] = temp1;
    }
}

#endif
