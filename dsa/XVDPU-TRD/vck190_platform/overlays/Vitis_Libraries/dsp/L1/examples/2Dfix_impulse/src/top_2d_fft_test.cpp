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
#include "top_2d_fft_test.hpp"
void top_fft2d(MemWideIFStreamTypeIn& p_inStream, MemWideIFStreamTypeOut& p_outStream) {
#ifdef _DEBUG_TYPES
    T_outType_row T_outType_row_temp;
    T_outType T_outType_temp;
    MemWideIFTypeIn MemWideIFTypeIn_temp;
    MemWideIFTypeOut MemWideIFTypeOut_temp;
#endif

#pragma HLS INLINE
#pragma HLS DATA_PACK variable = p_inStream
#pragma HLS DATA_PACK variable = p_outStream
#pragma HLS interface ap_ctrl_none port = return

#ifndef __SYNTHESIS__
    std::cout << "================================================================================" << std::endl;
    std::cout << "---------------------Calling 2D FFT Kernel with Parameters----------------------" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "    The Main Memory Width (no. complex<float>)   : " << k_memWidth << std::endl;
    std::cout << "    The Size of 1D Row Kernel                    : " << FFTParams::N << std::endl;
    std::cout << "    The SSR for 1D Row Kernel                    : " << FFTParams::R << std::endl;
    std::cout << "    The Transform Direction for Row Kernel       : "
              << ((FFTParams::transform_direction == FORWARD_TRANSFORM) ? "Forward" : "Reverse");
    std::cout << std::endl;

    std::cout << "    The Size of 1D Column Kernel                 : " << FFTParams2::N << std::endl;
    std::cout << "    The SSR for 1D Row Kernel                    : " << FFTParams2::R << std::endl;
    std::cout << "    The Transform Direction for Row Kernel       : "
              << ((FFTParams2::transform_direction == FORWARD_TRANSFORM) ? "Forward" : "Reverse");
    std::cout << std::endl;

    std::cout << "    The Row Instance ID Offset                   : " << k_rowInstanceIDOffset << std::endl;
    std::cout << "    The Column Instance ID Offset                : " << k_colInstanceIDOffset << std::endl;

    std::cout << "    Number of 1D Kernels Used Row/Col wise       : " << k_numOfKernels << std::endl;
    std::cout << "    The Total Number of 1D Kernels Used(row+col) : " << 2 * k_numOfKernels << std::endl;
    std::cout << "================================================================================" << std::endl;

#endif
    fft2d<k_memWidth, k_fftKernelSize, k_fftKernelSize, k_numOfKernels, FFTParams, FFTParams2, k_rowInstanceIDOffset,
          k_colInstanceIDOffset, T_elemType>(p_inStream, p_outStream);
}
