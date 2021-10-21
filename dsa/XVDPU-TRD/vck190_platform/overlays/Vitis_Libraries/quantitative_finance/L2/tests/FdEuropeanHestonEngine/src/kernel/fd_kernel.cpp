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
/**********
 * Copyright (c) 2019, Xilinx, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * **********/
/**
 *  @brief FPGA FD accelerator kernel
 *
 *  $DateTime: 2018/02/05 02:36:41 $
 */

#include <assert.h>
#include "ap_fixed.h"
#include "xf_fintech/fd_solver.hpp"
#include "math.h"

constexpr size_t mylog2(size_t n) {
    return ((n < 2) ? 0 : 1 + mylog2(n / 2));
}

// Extern C required for sw_emu
extern "C" {

void fd_kernel(ap_uint<512>* A,
               ap_uint<512>* Ar,
               ap_uint<512>* Ac,
               unsigned int nnz,
               ap_uint<512>* A1,
               ap_uint<512>* A2,
               ap_uint<512>* X1,
               ap_uint<512>* X2,
               ap_uint<512>* b,
               ap_uint<512>* u0,
               unsigned int M1,
               unsigned int M2,
               unsigned int N,
               ap_uint<512>* price) {
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmemm num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 64
#pragma HLS INTERFACE m_axi port = Ar offset = slave bundle = gmemm num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 64
#pragma HLS INTERFACE m_axi port = Ac offset = slave bundle = gmemm num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 64
#pragma HLS INTERFACE m_axi port = A1 offset = slave bundle = gmemm num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 64
#pragma HLS INTERFACE m_axi port = A2 offset = slave bundle = gmemm num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 64
#pragma HLS INTERFACE m_axi port = X1 offset = slave bundle = gmemm num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 64
#pragma HLS INTERFACE m_axi port = X2 offset = slave bundle = gmemm num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 64
#pragma HLS INTERFACE m_axi port = b offset = slave bundle = gmemm num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 64
#pragma HLS INTERFACE m_axi port = u0 offset = slave bundle = gmemm num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 64
#pragma HLS INTERFACE m_axi port = price offset = slave bundle = gmemm num_write_outstanding = \
    16 num_read_outstanding = 16 max_write_burst_length = 16 max_read_burst_length = 16 depth = 16 latency = 64

#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = Ar bundle = control
#pragma HLS INTERFACE s_axilite port = Ac bundle = control
#pragma HLS INTERFACE s_axilite port = nnz bundle = control
#pragma HLS INTERFACE s_axilite port = A1 bundle = control
#pragma HLS INTERFACE s_axilite port = A2 bundle = control
#pragma HLS INTERFACE s_axilite port = X1 bundle = control
#pragma HLS INTERFACE s_axilite port = X2 bundle = control
#pragma HLS INTERFACE s_axilite port = b bundle = control
#pragma HLS INTERFACE s_axilite port = u0 bundle = control
#pragma HLS INTERFACE s_axilite port = M1 bundle = control
#pragma HLS INTERFACE s_axilite port = M2 bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = price bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    static const unsigned int data_elements_per_ddr_word = FD_DATA_WORDS_IN_DDR;
    static const unsigned int index_elements_per_ddr_word = 64 / sizeof(unsigned int);
    static const unsigned int bits_per_data_type = 8 * sizeof(FD_DATA_TYPE);
    static const unsigned int bits_per_index = 8 * sizeof(unsigned int);
    static const unsigned int a_data_depth_ddr =
        (nnz + FD_DATA_WORDS_IN_DDR - 1) / FD_DATA_WORDS_IN_DDR;   // Pass from host???
    static const unsigned int a_index_depth_ddr = (nnz + 15) / 16; // Pass from host???
    static const unsigned int m_data_depth_ddr = (FD_M_SIZE / FD_DATA_WORDS_IN_DDR);
    static const unsigned int dim2_size1 = 3;
    static const unsigned int dim2_size2 = 5;
    static const unsigned int m_size = FD_M_SIZE;
    static const unsigned int a_size = m_size * 10;
    static const unsigned int float_size = sizeof(FD_DATA_TYPE);
    static const unsigned int index_width = (float_size * FD_DATA_WORDS_IN_DDR) / 4;
    static const unsigned int log2_m_size = mylog2(FD_M_SIZE);

// Chip memory
#ifdef __SYNTHESIS__
    FD_DATA_TYPE A_chip[a_size];
    unsigned int Ar_chip[a_size];
    unsigned int Ac_chip[a_size];
    FD_DATA_TYPE A1_chip[m_size][dim2_size1];
    FD_DATA_TYPE A2_chip[m_size][dim2_size2];
    FD_DATA_TYPE X1_chip[m_size][dim2_size1];
    FD_DATA_TYPE X2_chip[m_size][dim2_size2];
    FD_DATA_TYPE b_chip[m_size];
    FD_DATA_TYPE u0_chip[m_size];
    FD_DATA_TYPE price_chip[m_size];
#else
    auto A_chip = new FD_DATA_TYPE[a_size];
    auto Ar_chip = new unsigned int[a_size];
    auto Ac_chip = new unsigned int[a_size];
    auto A1_chip = new FD_DATA_TYPE[m_size][dim2_size1];
    auto A2_chip = new FD_DATA_TYPE[m_size][dim2_size2];
    auto X1_chip = new FD_DATA_TYPE[m_size][dim2_size1];
    auto X2_chip = new FD_DATA_TYPE[m_size][dim2_size2];
    auto b_chip = new FD_DATA_TYPE[m_size];
    auto u0_chip = new FD_DATA_TYPE[m_size];
    auto price_chip = new FD_DATA_TYPE[m_size];
#endif

// Parition arrays for parallel access
#pragma HLS array_partition variable = A_chip cyclic factor = data_elements_per_ddr_word dim = 1
#pragma HLS array_partition variable = Ar_chip cyclic factor = index_elements_per_ddr_word dim = 1
#pragma HLS array_partition variable = Ac_chip cyclic factor = index_elements_per_ddr_word dim = 1
#pragma HLS array_partition variable = b_chip cyclic factor = data_elements_per_ddr_word dim = 1
#pragma HLS array_partition variable = u0_chip cyclic factor = data_elements_per_ddr_word dim = 1
#pragma HLS array_partition variable = A1_chip cyclic factor = data_elements_per_ddr_word dim = 1
#pragma HLS array_partition variable = X1_chip cyclic factor = data_elements_per_ddr_word dim = 1
#pragma HLS array_partition variable = A2_chip cyclic factor = data_elements_per_ddr_word dim = 1
#pragma HLS array_partition variable = X2_chip cyclic factor = data_elements_per_ddr_word dim = 1

    // The array and vectors are copied from DDR into on-chip storage
    // The data is read in a 512-bit DDR word and then split into the individual
    // data/index words using some pointer mangling
    for (unsigned int i = 0; i < a_data_depth_ddr; ++i) {
#pragma HLS PIPELINE
        ap_uint<512> wide_temp = A[i];
        for (unsigned int j = 0; j < data_elements_per_ddr_word; ++j) {
#pragma HLS UNROLL
            FD_DATA_EQ_TYPE temp = wide_temp.range(bits_per_data_type * (j + 1) - 1, bits_per_data_type * j);
            A_chip[data_elements_per_ddr_word * i + j] = *(FD_DATA_TYPE*)(&temp);
        }
    }

    for (unsigned int i = 0; i < a_index_depth_ddr; ++i) {
#pragma HLS PIPELINE
        ap_uint<512> wide_temp = Ar[i];
        for (unsigned int j = 0; j < index_elements_per_ddr_word; ++j) {
#pragma HLS UNROLL
            Ar_chip[index_elements_per_ddr_word * i + j] =
                wide_temp.range(bits_per_index * (j + 1) - 1, bits_per_index * j);
        }
    }

    for (unsigned int i = 0; i < a_index_depth_ddr; ++i) {
#pragma HLS PIPELINE
        ap_uint<512> wide_temp = Ac[i];
        for (unsigned int j = 0; j < index_elements_per_ddr_word; ++j) {
#pragma HLS UNROLL
            Ac_chip[index_elements_per_ddr_word * i + j] =
                wide_temp.range(bits_per_index * (j + 1) - 1, bits_per_index * j);
        }
    }

    for (unsigned int i = 0; i < m_data_depth_ddr; ++i) {
#pragma HLS PIPELINE
        ap_uint<512> wide_temp = b[i];
        for (unsigned int j = 0; j < data_elements_per_ddr_word; ++j) {
#pragma HLS UNROLL
            FD_DATA_EQ_TYPE temp = wide_temp.range(bits_per_data_type * (j + 1) - 1, bits_per_data_type * j);
            b_chip[data_elements_per_ddr_word * i + j] = *(FD_DATA_TYPE*)(&temp);
        }
    }

    for (unsigned int i = 0; i < m_data_depth_ddr; ++i) {
#pragma HLS PIPELINE
        ap_uint<512> wide_temp = u0[i];
        for (unsigned int j = 0; j < data_elements_per_ddr_word; ++j) {
#pragma HLS UNROLL
            FD_DATA_EQ_TYPE temp = wide_temp.range(bits_per_data_type * (j + 1) - 1, bits_per_data_type * j);
            u0_chip[data_elements_per_ddr_word * i + j] = *(FD_DATA_TYPE*)(&temp);
        }
    }

    // Unpack 1D flat arrays to 2D
    for (unsigned int d = 0; d < dim2_size1; ++d) {
        for (unsigned int i = 0; i < m_data_depth_ddr; ++i) {
#pragma HLS PIPELINE
            ap_uint<512> wide_temp = A1[m_data_depth_ddr * d + i];
            for (unsigned int j = 0; j < data_elements_per_ddr_word; ++j) {
#pragma HLS UNROLL
                FD_DATA_EQ_TYPE temp = wide_temp.range(bits_per_data_type * (j + 1) - 1, bits_per_data_type * j);
                A1_chip[data_elements_per_ddr_word * i + j][d] = *(FD_DATA_TYPE*)(&temp);
            }
        }
    }

    for (unsigned int d = 0; d < dim2_size1; ++d) {
        for (unsigned int i = 0; i < m_data_depth_ddr; ++i) {
#pragma HLS PIPELINE
            ap_uint<512> wide_temp = X1[m_data_depth_ddr * d + i];
            for (unsigned int j = 0; j < data_elements_per_ddr_word; ++j) {
#pragma HLS UNROLL
                FD_DATA_EQ_TYPE temp = wide_temp.range(bits_per_data_type * (j + 1) - 1, bits_per_data_type * j);
                X1_chip[data_elements_per_ddr_word * i + j][d] = *(FD_DATA_TYPE*)(&temp);
            }
        }
    }

    for (unsigned int d = 0; d < dim2_size2; ++d) {
        for (unsigned int i = 0; i < m_data_depth_ddr; ++i) {
#pragma HLS PIPELINE
            ap_uint<512> wide_temp = A2[m_data_depth_ddr * d + i];
            for (unsigned int j = 0; j < data_elements_per_ddr_word; ++j) {
#pragma HLS UNROLL
                FD_DATA_EQ_TYPE temp = wide_temp.range(bits_per_data_type * (j + 1) - 1, bits_per_data_type * j);
                A2_chip[data_elements_per_ddr_word * i + j][d] = *(FD_DATA_TYPE*)(&temp);
            }
        }
    }

    for (unsigned int d = 0; d < dim2_size2; ++d) {
        for (unsigned int i = 0; i < m_data_depth_ddr; ++i) {
#pragma HLS PIPELINE
            ap_uint<512> wide_temp = X2[m_data_depth_ddr * d + i];
            for (unsigned int j = 0; j < data_elements_per_ddr_word; ++j) {
#pragma HLS UNROLL
                FD_DATA_EQ_TYPE temp = wide_temp.range(bits_per_data_type * (j + 1) - 1, bits_per_data_type * j);
                X2_chip[data_elements_per_ddr_word * i + j][d] = *(FD_DATA_TYPE*)(&temp);
            }
        }
    }

    // Run the kernel
    xf::fintech::FdDouglas<FD_DATA_TYPE, FD_DATA_WORDS_IN_DDR, index_width, a_size, m_size, log2_m_size, dim2_size1,
                           dim2_size2>(A_chip, Ar_chip, Ac_chip, a_data_depth_ddr * data_elements_per_ddr_word, A1_chip,
                                       A2_chip, X1_chip, X2_chip, b_chip, u0_chip, M1, M2, N, price_chip);

    // for(unsigned int i=0;i<m_size;++i) price[i] = price_chip[i];

    // Vectorize result back to DDR size
    for (unsigned int i = 0; i < m_data_depth_ddr; ++i) {
#pragma HLS PIPELINE
        ap_uint<512> wide_temp;
        for (unsigned int j = 0; j < data_elements_per_ddr_word; ++j) {
#pragma HLS UNROLL
            FD_DATA_TYPE temp = price_chip[data_elements_per_ddr_word * i + j];
            wide_temp.range(bits_per_data_type * (j + 1) - 1, bits_per_data_type * j) = *(FD_DATA_EQ_TYPE*)(&temp);
        }
        price[i] = wide_temp;
    }

#ifndef __SYNTHESIS__
    delete[] A_chip;
    delete[] Ar_chip;
    delete[] Ac_chip;
    delete[] A1_chip;
    delete[] A2_chip;
    delete[] X1_chip;
    delete[] X2_chip;
    delete[] b_chip;
    delete[] u0_chip;
    delete[] price_chip;
#endif
}

} // extern C
