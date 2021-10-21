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
 * @file hls_ssr_fft_2d.hpp
 * @brief XF FFT 2D kernel top level module
 *
 * This file is part of XF FFT Library.
 */

// File Name : hls_ssr_fft_2d_modeling_utilities.h

#ifndef __HLS_SSR_FFT_2D_MODELING_UTILITIES_H__
#define __HLS_SSR_FFT_2D_MODELING_UTILITIES_H__

#include <math.h>
#include <string>
#include <assert.h>
#include <stdio.h>
#include "hls_stream.h"

enum StimDataType {
    COUNTING,
    COUNTING_ROW,
    COUNTING_COL,
    IMPULSE,
    ZEROS,
    ONES,
    ROW_OF_ONES,
    COL_OF_ONES,
    C_COUNTING,
    C_COUNTING_ROW,
    C_COUNTING_COL,
    C_IMPULSE,
    C_ZEROS,
    C_ONES,
    C_ROW_OF_ONES,
    C_COL_OF_ONES
};

template <typename T_data, unsigned int t_rows, unsigned int t_cols>
void genStimulusfor2DFFT(T_data p_data[t_rows][t_cols], bool p_print = false, StimDataType p_dataType = IMPULSE) {
    int count = 0;
    std::cout << "Generating Input data for 2d FFT matrix form...." << std::endl;

    // Generate counting matrix
    // stored in row major order
    if (p_dataType == COUNTING) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                p_data[r][c] = count++;
            }
        }
    }

    // Generate counting row at row=0
    if (p_dataType == COUNTING_ROW) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                if (r == 0)
                    p_data[r][c] = count++;
                else
                    p_data[r][c] = 0;
            }
        }
    }

    // Generate counting column at col=0
    if (p_dataType == COUNTING_COL) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                if (c == 0)
                    p_data[r][c] = count++;
                else
                    p_data[r][c] = 0;
            }
        }
    }

    // Generate Impulse
    if (p_dataType == IMPULSE) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                if (c == 0 && r == 0)
                    p_data[r][c] = 1;
                else
                    p_data[r][c] = 0;
            }
        }
    }

    // Generate ZEROS
    if (p_dataType == ZEROS) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                p_data[r][c] = 0;
            }
        }
    }

    // Generate ONES
    if (p_dataType == ONES) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                p_data[r][c] = 1;
            }
        }
    }

    // Generate ROW of ONES
    if (p_dataType == ROW_OF_ONES) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                if (r == 0)
                    p_data[r][c] = 1;
                else
                    p_data[r][c] = 0;
            }
        }
    }
    // Generate counting matrix
    // stored in row major order
    if (p_dataType == C_COUNTING) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                p_data[r][c] = T_data(count, count);
                count++;
            }
        }
    }

    // Generate counting row at row=0
    if (p_dataType == C_COUNTING_ROW) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                if (r == 0) {
                    p_data[r][c] = T_data(count, count);
                    count++;
                } else
                    p_data[r][c] = 0;
            }
        }
    }

    // Generate counting column at col=0
    if (p_dataType == C_COUNTING_COL) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                if (c == 0) {
                    p_data[r][c] = T_data(count, count);
                    count++;
                } else
                    p_data[r][c] = 0;
            }
        }
    }

    // Generate Impulse
    if (p_dataType == C_IMPULSE) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                if (c == 0)
                    p_data[r][c] = T_data(1, 1);
                else
                    p_data[r][c] = 0;
            }
        }
    }

    // Generate ZEROS
    if (p_dataType == C_ZEROS) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                p_data[r][c] = 0;
            }
        }
    }

    // Generate ONES
    if (p_dataType == C_ONES) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                p_data[r][c] = T_data(1, 1);
            }
        }
    }

    // Generate ROW of ONES
    if (p_dataType == C_ROW_OF_ONES) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                if (r == 0)
                    p_data[r][c] = T_data(1, 1);
                else
                    p_data[r][c] = 0;
            }
        }
    }

    // Generate COL of ONES
    if (p_dataType == C_COL_OF_ONES) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                if (c == 0)
                    p_data[r][c] = T_data(1, 1);
                else
                    p_data[r][c] = 0;
            }
        }
    }

    // Generate COL of ONES
    if (p_dataType == C_COL_OF_ONES) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                if (c == 0)
                    p_data[r][c] = T_data(1, 1);
                else
                    p_data[r][c] = 0;
            }
        }
    }

    if (p_print) {
        int width = int(ceil(log10(t_rows * t_cols)));
        std::cout << "The Stimulus Data generated ..." << std::endl;
        int nl = 0;
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols; ++c) {
                std::cout << std::setw(2 * width + 4) << "(" << p_data[r][c].real() << "," << p_data[r][c].imag() << ")"
                          << ", ";
                if (nl == 6) {
                    std::cout << std::endl;
                    nl = 0;
                } else
                    nl++;
            }
            nl = 0;
            std::cout << std::endl;
            std::cout
                << "-------------------------------------------------------------------------------------------------"
                << std::endl;
        }
    }
}
template <unsigned int t_rows, unsigned int t_cols, typename T_elemType>
void print2dMat(T_elemType p_data[t_rows][t_cols], std::string p_msg) {
    std::cout << "===========================================================" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << p_msg << std::endl;
    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols; ++c) {
            std::cout << std::setw(5) << p_data[r][c] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "===========================================================" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "===========================================================" << std::endl;
}
template <unsigned int t_rows, unsigned int t_cols, unsigned int t_memWidth, typename T_streamDataType>
void printMatStream(hls::stream<T_streamDataType>& p_stream, std::string p_msg) {
    std::cout << "===========================================================" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << p_msg << std::endl;
    hls::stream<T_streamDataType> buffStream;
    unsigned int l_streamSize = p_stream.size();
    std::cout << "Stream Size :" << p_stream.size() << std::endl;
    for (int r = 0; r < t_rows / t_memWidth; ++r) {
        for (int c = 0; c < t_cols; ++c) {
            T_streamDataType sample = p_stream.read();
            buffStream.write(sample);
            std::cout << sample << std::endl;
        }
    }
    for (int r = 0; r < t_rows / t_memWidth; ++r) {
        for (int c = 0; c < t_cols; ++c) {
            T_streamDataType sample = buffStream.read();
            p_stream.write(sample);
        }
    }
    assert(l_streamSize == p_stream.size());
    std::cout << "===========================================================" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "===========================================================" << std::endl;
}

// delegate for function overloading
template <typename T_data, unsigned int t_rows, unsigned int t_cols>
void genStimulusfor2DFFT(T_data p_data[t_rows][t_cols], StimDataType p_dataType = IMPULSE) {
    genStimulusfor2DFFT<T_data, t_rows, t_cols>(p_data, false, p_dataType);
}

template <unsigned int t_rows,
          unsigned int t_cols,
          unsigned int t_memWidth,
          typename T_elemType,
          typename T_streamDataType>
void stream2DMatrix(T_elemType p_matrix[t_rows][t_cols], hls::stream<T_streamDataType>& p_stream) {
    T_streamDataType l_inMemWideSample1;
    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols / t_memWidth; ++c) {
            for (int w = 0; w < t_memWidth; ++w) {
                l_inMemWideSample1[w] = p_matrix[r][w + c * t_memWidth];
            }
            p_stream.write(l_inMemWideSample1);
        }
    }
}

template <unsigned int t_rows,
          unsigned int t_cols,
          unsigned int t_memWidth,
          typename T_elemType,
          typename T_streamDataType>
void streamToMatrix(hls::stream<T_streamDataType>& p_inStream, T_outType p_matrixOut[t_rows][t_cols]) {
    T_streamDataType l_inMemWideSample1;
    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols / t_memWidth; ++c) {
            l_inMemWideSample1 = p_inStream.read();
            for (int w = 0; w < t_memWidth; ++w) {
                p_matrixOut[r][w + c * t_memWidth] = l_inMemWideSample1[w];
            }
        }
    }
}

template <unsigned int t_rows,
          unsigned int t_cols,
          unsigned int t_memWidth,
          unsigned int t_kernels,
          typename T_elemType,
          typename T_streamDataType>
void doBlkTransposeAndStream(hls::stream<T_streamDataType>& p_inStream, hls::stream<T_streamDataType>& p_stream) {
    T_streamDataType l_inMemWideSample1;
    T_elemType p_matrix[t_rows][t_cols];
    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols / t_memWidth; ++c) {
            l_inMemWideSample1 = p_inStream.read();
            for (int w = 0; w < t_memWidth; ++w) {
                p_matrix[r][w + c * t_memWidth] = l_inMemWideSample1[w];
            }
        }
    }
    T_streamDataType inMemWideSample;
    for (int blk = 0; blk < t_rows / t_kernels; ++blk) // blocks
    {
        for (int c = 0; c < t_cols / k_memWidth; ++c) // wide columns
        {
            for (int r = 0; r < t_kernels; ++r) // rows in block
            {
                for (int elemIndex = 0; elemIndex < t_memWidth; elemIndex++) {
                    int row_index = blk * t_kernels + r;
                    int col_index = c * t_memWidth + elemIndex;
                    inMemWideSample[elemIndex] = p_matrix[row_index][col_index];
                }
                p_stream.write(inMemWideSample);
            }
        }
    }
}

template <unsigned int t_rows,
          unsigned int t_cols,
          unsigned int t_memWidth,
          unsigned int t_kernels,
          typename T_elemType,
          typename T_streamDataType>
void doBlkInverseTranspose(hls::stream<T_streamDataType>& p_stream, hls::stream<T_streamDataType>& p_outStream) {
    T_elemType p_matrix[t_rows][t_cols];
    T_streamDataType inMemWideSample;
    for (int blk = 0; blk < t_rows / t_kernels; ++blk) // blocks
    {
        for (int c = 0; c < t_cols / k_memWidth; ++c) // wide columns
        {
            for (int r = 0; r < t_kernels; ++r) // rows in block
            {
                inMemWideSample = p_stream.read();
                for (int elemIndex = 0; elemIndex < t_memWidth; elemIndex++) {
                    int row_index = blk * t_kernels + r;
                    int col_index = c * t_memWidth + elemIndex;
                    p_matrix[row_index][col_index] = inMemWideSample[elemIndex];
                }
            }
        }
    }

    MemWideIFTypeIn l_inMemWideSample1;
    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols / t_memWidth; ++c) {
            for (int w = 0; w < t_memWidth; ++w) {
                l_inMemWideSample1[w] = p_matrix[r][w + c * t_memWidth];
            }
            p_outStream.write(l_inMemWideSample1);
        }
    }
}

template <unsigned int t_rows,
          unsigned int t_cols,
          unsigned int t_memWidth,
          typename T_elemType,
          typename T_streamDataType>
void matrixTransElmntWise(hls::stream<T_streamDataType>& p_inStream, hls::stream<T_streamDataType>& p_outStream) {
    T_elemType p_matrixIn[t_rows][t_cols];
    T_elemType p_matrixOut[t_rows][t_cols];

    T_streamDataType l_inMemWideSample1;
    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols / t_memWidth; ++c) {
            l_inMemWideSample1 = p_inStream.read();
            for (int w = 0; w < t_memWidth; ++w) {
                p_matrixIn[r][w + c * t_memWidth] = l_inMemWideSample1[w];
            }
        }
    }

    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols; ++c) {
            p_matrixOut[r][c] = p_matrixIn[c][r];
        }
    }

    T_streamDataType l_inMemWideSample2;
    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols / t_memWidth; ++c) {
            for (int w = 0; w < t_memWidth; ++w) {
                l_inMemWideSample2[w] = p_matrixOut[r][w + c * t_memWidth];
            }
            p_outStream.write(l_inMemWideSample2);
        }
    }
}

template <unsigned int t_rows, unsigned int t_cols, typename T_elemType>
bool checkCountingMatrix(T_elemType p_data[t_rows][t_cols]) {
    T_elemType sample;
    int count = 0;
    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols; ++c) {
            sample = count++;
            if (sample != p_data[r][c]) return false;
        }
    }
    return true;
}

template <unsigned int t_rows, unsigned int t_cols, typename T_elemType>
bool compare2dMats(T_elemType p_data2[t_rows][t_cols], T_elemType p_data1[t_rows][t_cols]) {
    T_elemType sample;
    int count = 0;
    for (int r = 0; r < t_rows; ++r) {
        for (int c = 0; c < t_cols; ++c) {
            sample = count++;
            if (p_data1[r][c] != p_data2[r][c]) return false;
        }
    }
    return true;
}

#endif //__HLS_SSR_FFT_2D_MODELING_UTILITIES_H__
