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
//================================== End Lic =================================================
#define TEST_2D_FFT_
#ifdef TEST_2D_FFT_
#ifndef __SYNTHESIS__
#define _DEBUG_TYPES
#endif
#include <math.h>
#include <string>
#include <assert.h>
#include <stdio.h>
#include "top_2d_fft_test.hpp"
#include "mVerificationUtlityFunctions.hpp"
#include "vitis_fft/hls_ssr_fft_2d_modeling_utilities.hpp"
#include "vt_fft.hpp"

int main(int argc, char** argv) {
    // 2d input matrix
    T_elemType l_inMat[k_fftKernelSize][k_fftKernelSize];
    T_outType l_outMat[k_fftKernelSize][k_fftKernelSize];
    T_outType l_data2d_golden[k_fftKernelSize][k_fftKernelSize];

    // init input matrix with imag part only impulse
    for (int r = 0; r < k_fftKernelSize; ++r) {
        for (int c = 0; c < k_fftKernelSize; ++c) {
            if (r == 0 && c == 0)
                l_inMat[r][c] = T_compleFloat(0, 1);
            else
                l_inMat[r][c] = T_compleFloat(0, 0);
        }
    }
    // Wide Stream for reading and streaming a 2-d matrix
    MemWideIFStreamTypeIn l_matToStream("matrixToStreaming");
    MemWideIFStreamTypeOut fftOutputStream("fftOutputStream");
    // Pass same data stream multiple times to measure the II correctly
    for (int runs = 0; runs < 5; ++runs) {
        stream2DMatrix<k_fftKernelSize, k_fftKernelSize, k_memWidth, T_elemType, MemWideIFTypeIn>(l_inMat,
                                                                                                  l_matToStream);
        top_fft2d(l_matToStream, fftOutputStream);

        printMatStream<k_fftKernelSize, k_fftKernelSize, k_memWidth, MemWideIFTypeOut>(
            fftOutputStream, "2D FFT Output Natural Order...");
        streamToMatrix<k_fftKernelSize, k_fftKernelSize, k_memWidth, T_outType>(fftOutputStream, l_outMat);
    } // runs loop

    T_outType golden_result = T_elemType(0, 1);
    for (int r = 0; r < k_fftKernelSize; ++r) {
        for (int c = 0; c < k_fftKernelSize; ++c) {
            if (golden_result != l_outMat[r][c]) return 1;
        }
    }

    std::cout << "================================================================" << std::endl;
    std::cout << "---------------------Impulse test Passed Successfully." << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
#endif
