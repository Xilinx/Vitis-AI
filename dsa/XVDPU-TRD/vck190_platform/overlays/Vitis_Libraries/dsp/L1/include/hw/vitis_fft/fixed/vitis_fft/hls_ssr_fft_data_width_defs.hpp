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
 * @file hls_ssr_fft_data_width_defs.hpp
 * @brief header, describes OCM interface block for data demuxing
 *
 */

#ifndef __HLS_SSR_FFT_DATA_WIDTH_DEFS_H__
#define __HLS_SSR_FFT_DATA_WIDTH_DEFS_H__

#ifndef __SYNTHESIS__
#include <iostream>
#endif

#include "vitis_fft/hls_ssr_fft_enums.hpp"
#include "vitis_fft/hls_ssr_fft_utilities.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <typename t_elemType, unsigned int t_memWidthBits, unsigned int t_R, DataTypeEnum t_dataType = COMPLEX>
struct DataWidthDefs {
    static const unsigned int k_sizeElemTypeBytes = sizeof(t_elemType);
    static const unsigned int k_sizeRealElemType = sizeof(t_elemType) / 2;
    static const unsigned int k_sizeImagElemType = sizeof(t_elemType) / 2;
    static const unsigned int k_sizeRadixWideTypeBytes = k_sizeElemTypeBytes * t_R;
    static const unsigned int k_memWidthBytes = t_memWidthBits / 8;
    static const DataTypeEnum k_DataType = COMPLEX;
    static const unsigned int k_sizeMemInElements = k_memWidthBytes / k_sizeElemTypeBytes;
    static const unsigned int k_defaultRowKernels = k_sizeMemInElements / t_R;
    static const unsigned int k_defaultColKernels = k_sizeMemInElements / t_R;
#ifndef __SYNTHESIS__

    friend std::ostream& operator<<(std::ostream& out, const DataWidthDefs& o) {
        out << "Memory Width in Complex Words         : " << k_sizeMemInElements << std::endl;
        out << "Number of Default Row 1-D FFT Kernels : " << k_defaultRowKernels << std::endl;
        out << "Number of Default Col 1-D FFT Kernels : " << k_defaultColKernels << std::endl;
        out << "Memory Width in Bytes                 : " << k_memWidthBytes << std::endl;
        out << "Size of Complex Element Type in Bytes : " << k_sizeElemTypeBytes << std::endl;
        out << "Size of Real Element Type in Bytes    : " << k_sizeRealElemType << std::endl;
        out << "Size of Imag Element Type in Bytes    : " << k_sizeImagElemType << std::endl;
        out << "Size of Radix Wide Type in Bytes      : " << k_sizeRadixWideTypeBytes << std::endl;
        out << "Element Data Type                     : "
            << "COMPLEX" << std::endl;

        return out;
    }
#endif
};
template <typename t_elemType, unsigned int t_memWidthBits, unsigned int t_R>
struct DataWidthDefs<t_elemType, t_memWidthBits, t_R, REAL> {
    static const unsigned int k_sizeElemTypeBytes = sizeof(t_elemType);
    static const unsigned int k_sizeRealElemType = sizeof(t_elemType);
    static const unsigned int k_sizeRadixWideTypeBytes = k_sizeElemTypeBytes * t_R;
    static const unsigned int k_memWidthBytes = t_memWidthBits / 8;
    static const DataTypeEnum k_DataType = REAL;
    static const unsigned int k_sizeMemInElements = k_memWidthBytes / k_sizeElemTypeBytes;
    static const unsigned int k_defaultRowKernels = k_sizeMemInElements / t_R;
    static const unsigned int k_defaultColKernels = k_sizeMemInElements / t_R;
#ifndef __SYNTHESIS__
    friend std::ostream& operator<<(std::ostream& out, const DataWidthDefs& o) {
        out << "Memory Width in Complex Words         : " << k_sizeMemInElements << std::endl;
        out << "Number of Default Row 1-D FFT Kernels : " << k_defaultRowKernels << std::endl;
        out << "Number of Default Col 1-D FFT Kernels : " << k_defaultColKernels << std::endl;
        out << "Memory Width in Bytes                 : " << k_memWidthBytes << std::endl;
        out << "Size of Complex Element Type in Bytes : " << k_sizeElemTypeBytes << std::endl;
        out << "Size of Real Element Type in Bytes    : " << k_sizeRealElemType << std::endl;
        out << "Size of Radix Wide Type in Bytes      : " << k_sizeRadixWideTypeBytes << std::endl;
        out << "Element Data Type                     : "
            << "COMPLEX" << std::endl;
        return out;
    }
#endif
};

template <typename t_elemType, unsigned int t_memWidthBits, unsigned int t_R>
struct DataWidthDefs<t_elemType, t_memWidthBits, t_R, IMAG> {
    static const unsigned int k_sizeElemTypeBytes = sizeof(t_elemType);
    static const unsigned int k_sizeImagElemType = sizeof(t_elemType);
    static const unsigned int k_sizeRadixWideTypeBytes = k_sizeElemTypeBytes * t_R;
    static const unsigned int k_memWidthBytes = t_memWidthBits / 8;
    static const DataTypeEnum k_DataType = IMAG;
    static const unsigned int k_sizeMemInElements = k_memWidthBytes / k_sizeElemTypeBytes;
    static const unsigned int k_defaultRowKernels = k_sizeMemInElements / t_R;
    static const unsigned int k_defaultColKernels = k_sizeMemInElements / t_R;
#ifndef __SYNTHESIS__
    friend std::ostream& operator<<(std::ostream& out, const DataWidthDefs& o) {
        out << "Memory Width in Complex Words         : " << k_sizeMemInElements << std::endl;
        out << "Number of Default Row 1-D FFT Kernels : " << k_defaultRowKernels << std::endl;
        out << "Number of Default Col 1-D FFT Kernels : " << k_defaultColKernels << std::endl;
        out << "Memory Width in Bytes                 : " << k_memWidthBytes << std::endl;
        out << "Size of Complex Element Type in Bytes : " << k_sizeElemTypeBytes << std::endl;
        out << "Size of Imag Element Type in Bytes    : " << k_sizeImagElemType << std::endl;
        out << "Size of Radix Wide Type in Bytes      : " << k_sizeRadixWideTypeBytes << std::endl;
        out << "Element Data Type                     : "
            << "COMPLEX" << std::endl;

        return out;
    }
#endif
};

// This class is defined just to print throughput estimates.
template <typename t_elemType,
          unsigned int t_memWidthBits,
          unsigned int t_R,
          unsigned int t_numRows,
          unsigned int t_numCols,
          DataTypeEnum t_dataType = COMPLEX>
struct DataWidthDefsWithPerf : public DataWidthDefs<t_elemType, t_memWidthBits, t_R, COMPLEX> {
#ifndef __SYNTHESIS__
    static const unsigned int k_fmax = 1; // Ghz
    static const unsigned int k_totalNumKernels =
        DataWidthDefs<t_elemType, t_memWidthBits, t_R, COMPLEX>::k_defaultRowKernels *
        DataWidthDefs<t_elemType, t_memWidthBits, t_R, COMPLEX>::k_defaultColKernels;
    static const unsigned int k_opsPer1DRowKernel = 5 * t_numCols * ssrFFTLog2<t_numCols>::val;
    static const unsigned int k_opsPer1DColKernel = 5 * t_numRows * ssrFFTLog2<t_numRows>::val;

    friend std::ostream& operator<<(std::ostream& out, const DataWidthDefsWithPerf& o) {
        double f_factor = 0.3;
        double k_gsps1dRowKernel = k_fmax * t_R * f_factor;
        double k_1dRowKernelsPerSec = k_gsps1dRowKernel / t_numCols;
        double k_gops1DRowKernel = k_1dRowKernelsPerSec * k_opsPer1DRowKernel;

        double k_gsps1dColKernel = k_fmax * t_R * f_factor;
        double k_1dColKernelsPerSec = k_gsps1dColKernel / t_numRows;
        double k_gops1DColKernel = k_1dColKernelsPerSec * k_opsPer1DColKernel;

        double k_gopsRowAllRowKernels =
            k_gops1DRowKernel * DataWidthDefs<t_elemType, t_memWidthBits, t_R, COMPLEX>::k_defaultRowKernels;

        double k_gopsRowAllColKernels =
            k_gops1DColKernel * DataWidthDefs<t_elemType, t_memWidthBits, t_R, COMPLEX>::k_defaultColKernels;

        double k_gops = k_gopsRowAllRowKernels + k_gopsRowAllColKernels;
        out << static_cast<const DataWidthDefs<t_elemType, t_memWidthBits, t_R, COMPLEX>&>(o);
        out << "***********Expected total GOPs for 2-D Kernel(" << t_numRows << "x" << t_numCols << ") : " << k_gops
            << " GOPs" << std::endl;
        out << "   ********Expected total GOPs per 1-D Row Kernel(" << t_numCols << ") : " << k_gops1DRowKernel
            << " GOPs" << std::endl;
        out << "      *****Expected total GOPs per 1-D Column Kernel(" << t_numRows << "): " << k_gops1DColKernel
            << " GOPs" << std::endl;
        return out;
    }
#endif
};
/*
 *   static const double k_fmax=0.3; // Ghz
  static const int k_totalNumKernels=k_defaultRowKernels*k_defaultColKernels;
  static const int ssrFFTLog2<>
 */

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif //__HLS_SSR_FFT_DATA_WIDTH_DEFS_H__
