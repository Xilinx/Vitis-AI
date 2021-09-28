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
 * @file symMatMoverB2.hpp
 * @brief data movers for symmetric matrices and corresponding vectors.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_SYMMATMOVERB2_HPP
#define XF_BLAS_SYMMATMOVERB2_HPP

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_shift_reg.h"

namespace xf {

namespace blas {

template <typename t_DataType, unsigned int t_ParEntries>
void readSymUp2Stream(unsigned int p_n,
                      t_DataType* p_a,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outSymUpTransp,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outTransp,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outForward) {
    unsigned int l_blocks = p_n / t_ParEntries;
    unsigned int l_blocksMinusOne = l_blocks - 1;
    unsigned int i = 0; // block row index
    unsigned int j = 0; // block col index
    while (i < l_blocks) {
#pragma HLS PIPELINE II = t_ParEntries REWIND
        // read one block
        for (int br = 0; br < t_ParEntries; ++br) {
            WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
            for (int bl = 0; bl < t_ParEntries; ++bl) {
#pragma HLS UNROLL
                l_val[bl] = (i > j) ? (p_a[(j * p_n + br * l_blocks + i) * t_ParEntries + bl])
                                    : p_a[(i * p_n + br * l_blocks + j) * t_ParEntries + bl];
            }
            if (i == j) {
                p_outSymUpTransp.write(l_val);
            } else if (i > j) {
                p_outTransp.write(l_val);
            } else {
                p_outForward.write(l_val);
            }
        }
        if (j == l_blocksMinusOne) {
            i++;
            j = 0;
        } else {
            j++;
        }
    }
}

template <typename t_DataType, unsigned int t_ParEntries>
void mergeSymUpMat(unsigned int p_n,
                   hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_inSymUpTransp,
                   hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_inTransp,
                   hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_inForward,
                   hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    unsigned int l_blocks = p_n / t_ParEntries;
    for (unsigned int i = 0; i < l_blocks; ++i) {
        for (unsigned int j = 0; j < l_blocks; ++j) {
            for (unsigned int br = 0; br < t_ParEntries; ++br) {
#pragma HLS PIPELINE REWIND
                WideType<t_DataType, t_ParEntries> l_val;
                l_val = (i == j) ? p_inSymUpTransp.read() : (i < j) ? p_inForward.read() : p_inTransp.read();
                p_out.write(l_val);
            }
        }
    }
}

template <typename t_DataType, unsigned int t_ParEntries>
void readSymLo2Stream(unsigned int p_n,
                      t_DataType* p_a,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outSymUpTransp,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outTransp,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outForward) {
    uint16_t l_blocks = p_n / t_ParEntries;
    unsigned int l_blocksMinusOne = l_blocks - 1;
    unsigned int i = 0; // block row index
    unsigned int j = 0; // block col index
    while (i < l_blocks) {
#pragma HLS PIPELINE II = t_ParEntries REWIND
        // read one block
        for (int br = 0; br < t_ParEntries; ++br) {
            WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
            for (int bl = 0; bl < t_ParEntries; ++bl) {
#pragma HLS UNROLL
                l_val[bl] = (i < j) ? (p_a[(j * p_n + br * l_blocks + i) * t_ParEntries + bl])
                                    : p_a[(i * p_n + br * l_blocks + j) * t_ParEntries + bl];
            }
            if (i == j) {
                p_outSymUpTransp.write(l_val);
            } else if (i < j) {
                p_outTransp.write(l_val);
            } else {
                p_outForward.write(l_val);
            }
        }
        if (j == l_blocksMinusOne) {
            i++;
            j = 0;
        } else {
            j++;
        }
    }
}

template <typename t_DataType, unsigned int t_ParEntries>
void mergeSymLoMat(unsigned int p_n,
                   hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_inSymUpTransp,
                   hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_inTransp,
                   hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_inForward,
                   hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    unsigned int l_blocks = p_n / t_ParEntries;
    for (unsigned int i = 0; i < l_blocks; ++i) {
        for (unsigned int j = 0; j < l_blocks; ++j) {
            for (unsigned int br = 0; br < t_ParEntries; ++br) {
#pragma HLS PIPELINE
                WideType<t_DataType, t_ParEntries> l_val;
                if (i == j) {
                    l_val = p_inSymUpTransp.read();
                } else if (i < j) {
                    l_val = p_inTransp.read();
                } else {
                    l_val = p_inForward.read();
                }
                p_out.write(l_val);
            }
        }
    }
}

template <typename t_DataType, unsigned int t_ParEntries>
void readSpmUp2Stream(unsigned int p_n,
                      t_DataType* p_a,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outSymUpTransp,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outTransp,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outForward) {
    uint16_t l_blocks = p_n / t_ParEntries;
    t_DataType* l_aAddr = p_a;
    for (unsigned int i = 0; i < l_blocks; ++i) {
        for (unsigned int j = 0; j < l_blocks; ++j) {
            t_DataType* l_aAddrRowJ = p_a + (j * l_blocks - (j * (j - 1)) / 2) * t_ParEntries * t_ParEntries;
            t_DataType* l_aBlockAddr =
                (i > j) ? (l_aAddrRowJ + (i - j) * t_ParEntries) : (l_aAddr + (j - i) * t_ParEntries);
            unsigned int l_blockId = (i > j) ? j : i;
            for (unsigned int br = 0; br < t_ParEntries; ++br) {
#pragma HLS PIPELINE
                WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
                for (unsigned int bl = 0; bl < t_ParEntries; ++bl) {
                    l_val[bl] = l_aBlockAddr[bl];
                }
                if (i == j) {
                    p_outSymUpTransp.write(l_val);
                } else if (i > j) {
                    p_outTransp.write(l_val);
                } else {
                    p_outForward.write(l_val);
                }
                l_aBlockAddr += p_n - l_blockId * t_ParEntries;
            }
        }
        l_aAddr += (l_blocks - i) * t_ParEntries * t_ParEntries;
    }
}

template <typename t_DataType, unsigned int t_ParEntries>
void readSpmLo2Stream(unsigned int p_n,
                      t_DataType* p_a,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outSymUpTransp,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outTransp,
                      hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_outForward) {
    uint16_t l_blocks = p_n / t_ParEntries;
    t_DataType* l_aAddr = p_a;
    for (unsigned int i = 0; i < l_blocks; ++i) {
        for (unsigned int j = 0; j < l_blocks; ++j) {
            t_DataType* l_aAddrRowJ = p_a + j * (j + 1) * t_ParEntries * t_ParEntries / 2;
            t_DataType* l_aBlockAddr = (i < j) ? (l_aAddrRowJ + i * t_ParEntries) : (l_aAddr + j * t_ParEntries);
            unsigned int l_blockId = (i < j) ? j : i;
            for (unsigned int br = 0; br < t_ParEntries; ++br) {
#pragma HLS PIPELINE
                WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
                for (unsigned int bl = 0; bl < t_ParEntries; ++bl) {
                    l_val[bl] = l_aBlockAddr[bl];
                }
                if (i == j) {
                    p_outSymUpTransp.write(l_val);
                } else if (i < j) {
                    p_outTransp.write(l_val);
                } else {
                    p_outForward.write(l_val);
                }
                l_aBlockAddr += (l_blockId + 1) * t_ParEntries;
            }
        }
        l_aAddr += (i + 1) * t_ParEntries * t_ParEntries;
    }
}
/**
 * @brief symUp2Stream function that moves super-symmetric matrix from memory to stream
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries number of parallel processed entries in the matrix
 *
 * @param p_n number of rows/cols in a symmetric matrix
 * @param p_a point to a p_n x p_n symmetric matrix
 * @param p_out output stream
 */
template <typename t_DataType, unsigned int t_ParEntries>
void symUp2Stream(unsigned int p_n,
                  t_DataType* p_a,
                  hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_symTransp;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_transp;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_forward;

    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_symTranspRes;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_transpRes;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_forwardRes;

    unsigned int l_symBlocks = p_n / t_ParEntries;
    unsigned int l_transpBlocks = (l_symBlocks - 1) * l_symBlocks / 2;

#pragma HLS DATAFLOW
    readSymUp2Stream<t_DataType, t_ParEntries>(p_n, p_a, l_symTransp, l_transp, l_forward);
    transpSymUpMatBlocks<t_DataType, t_ParEntries>(l_symBlocks, l_symTransp, l_symTranspRes);
    transpMatBlocks<t_DataType, t_ParEntries>(l_transpBlocks, l_transp, l_transpRes);
    fwdMatBlocks<t_DataType, t_ParEntries>(l_transpBlocks, l_forward, l_forwardRes);
    mergeSymUpMat<t_DataType, t_ParEntries>(p_n, l_symTranspRes, l_transpRes, l_forwardRes, p_out);
}

/**
 * @brief symLo2Stream function that moves sub-symmetric matrix from memory to stream
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries number of parallel processed entries in the matrix
 *
 * @param p_n number of rows/cols in a symmetric matrix
 * @param p_a point to a p_n x p_n symmetric matrix
 * @param p_out output stream
 */
template <typename t_DataType, unsigned int t_ParEntries>
void symLo2Stream(unsigned int p_n,
                  t_DataType* p_a,
                  hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_symTransp;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_transp;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_forward;

    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_symTranspRes;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_transpRes;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_forwardRes;

    unsigned int l_symBlocks = p_n / t_ParEntries;
    unsigned int l_transpBlocks = (l_symBlocks - 1) * l_symBlocks / 2;

#pragma HLS DATAFLOW
    readSymLo2Stream<t_DataType, t_ParEntries>(p_n, p_a, l_symTransp, l_transp, l_forward);
    transpSymLoMatBlocks<t_DataType, t_ParEntries>(l_symBlocks, l_symTransp, l_symTranspRes);
    transpMatBlocks<t_DataType, t_ParEntries>(l_transpBlocks, l_transp, l_transpRes);
    fwdMatBlocks<t_DataType, t_ParEntries>(l_transpBlocks, l_forward, l_forwardRes);
    mergeSymLoMat<t_DataType, t_ParEntries>(p_n, l_symTranspRes, l_transpRes, l_forwardRes, p_out);
}

/**
 * @brief spmUp2Stream function that moves packed super-symmetric matrix from memory to stream
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries number of parallel processed entries in the matrix
 *
 * @param p_n number of rows/cols in a symmetric matrix
 * @param p_a point to a p_n x p_n symmetric matrix
 * @param p_out output stream
 */
template <typename t_DataType, unsigned int t_ParEntries>
void spmUp2Stream(unsigned int p_n,
                  t_DataType* p_a,
                  hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_symTransp;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_transp;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_forward;

    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_symTranspRes;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_transpRes;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_forwardRes;

    unsigned int l_symBlocks = p_n / t_ParEntries;
    unsigned int l_transpBlocks = (l_symBlocks - 1) * l_symBlocks / 2;

#pragma HLS DATAFLOW
    readSpmUp2Stream<t_DataType, t_ParEntries>(p_n, p_a, l_symTransp, l_transp, l_forward);
    transpSymUpMatBlocks<t_DataType, t_ParEntries>(l_symBlocks, l_symTransp, l_symTranspRes);
    transpMatBlocks<t_DataType, t_ParEntries>(l_transpBlocks, l_transp, l_transpRes);
    fwdMatBlocks<t_DataType, t_ParEntries>(l_transpBlocks, l_forward, l_forwardRes);
    mergeSymUpMat<t_DataType, t_ParEntries>(p_n, l_symTranspRes, l_transpRes, l_forwardRes, p_out);
}

/**
 * @brief spmLo2Stream function that moves packed sub-symmetric matrix from memory to stream
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries number of parallel processed entries in the matrix
 *
 * @param p_n number of rows/cols in a symmetric matrix
 * @param p_a point to a p_n x p_n symmetric matrix
 * @param p_out output stream
 */
template <typename t_DataType, unsigned int t_ParEntries>
void spmLo2Stream(unsigned int p_n,
                  t_DataType* p_a,
                  hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_symTransp;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_transp;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_forward;

    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_symTranspRes;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_transpRes;
    hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt> l_forwardRes;

    unsigned int l_symBlocks = p_n / t_ParEntries;
    unsigned int l_transpBlocks = (l_symBlocks - 1) * l_symBlocks / 2;

#pragma HLS DATAFLOW
    readSpmLo2Stream<t_DataType, t_ParEntries>(p_n, p_a, l_symTransp, l_transp, l_forward);
    transpSymLoMatBlocks<t_DataType, t_ParEntries>(l_symBlocks, l_symTransp, l_symTranspRes);
    transpMatBlocks<t_DataType, t_ParEntries>(l_transpBlocks, l_transp, l_transpRes);
    fwdMatBlocks<t_DataType, t_ParEntries>(l_transpBlocks, l_forward, l_forwardRes);
    mergeSymLoMat<t_DataType, t_ParEntries>(p_n, l_symTranspRes, l_transpRes, l_forwardRes, p_out);
}

/**
 * @brief vec2SymStream function that moves vector from memory to stream that matches the symatrix matrix data mover
 * outputs
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries number of parallelly processed entries in the matrix
 *
 * @param p_n number of rows/cols in a square matrix
 * @param p_x vector input
 * @param p_out output stream
 */
template <typename t_DataType, unsigned int t_ParEntries>
void vec2SymStream(unsigned int p_n,
                   t_DataType* p_x,
                   hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    unsigned int l_blocks = p_n / t_ParEntries;
    for (unsigned int i = 0; i < l_blocks; ++i) {
        for (unsigned int j = 0; j < l_blocks; ++j) {
            for (unsigned int br = 0; br < t_ParEntries; ++br) {
#pragma HLS PIPELINE
                WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
                for (unsigned int bl = 0; bl < t_ParEntries; ++bl) {
                    l_val[bl] = p_x[j * t_ParEntries + bl];
                }
                p_out.write(l_val);
            }
        }
    }
}

} // namespace blas

} // namespace xf
#endif
