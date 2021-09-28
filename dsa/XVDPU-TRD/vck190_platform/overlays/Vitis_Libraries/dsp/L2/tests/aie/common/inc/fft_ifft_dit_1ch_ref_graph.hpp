/*
 * Copyright 2021 Xilinx, Inc.
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
#ifndef _DSPLIB_FFT_IFFT_DIT_1CH_REF_GRAPH_HPP_
#define _DSPLIB_FFT_IFFT_DIT_1CH_REF_GRAPH_HPP_

#include <adf.h>
#include <vector>
#include "fft_ifft_dit_1ch_ref.hpp"
#include "fir_ref_utils.hpp"

#define CEIL(x, y) (((x + y - 1) / y) * y)
namespace xf {
namespace dsp {
namespace aie {
namespace fft {
namespace dit_1ch {
using namespace adf;

template <typename TT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_CASC_LEN = 1, // necessary to match UUT, but unused by ref model
          unsigned int TP_DYN_PT_SIZE = 0,
          unsigned int TP_WINDOW_VSIZE = TP_POINT_SIZE>
class fft_ifft_dit_1ch_ref_graph : public graph {
   public:
    port<input> in;
    port<output> out;

    // FIR Kernel
    kernel m_fftKernel;

    // Constructor
    fft_ifft_dit_1ch_ref_graph() {
        printf("===================================\n");
        printf("== FFT/IFFT DIT 1 Channel REF Graph\n");
        printf("===================================\n");

        // Create FIR class
        m_fftKernel = kernel::create_object<fft_ifft_dit_1ch_ref<TT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT,
                                                                 TP_SHIFT, TP_DYN_PT_SIZE, TP_WINDOW_VSIZE> >();

        // Make connections
        // Size of window in Bytes. Dynamic point size adds a 256 bit (32 byte) header. This is larger than required,
        // but keeps 256 bit alignment
        connect<window<TP_WINDOW_VSIZE * sizeof(TT_DATA) + TP_DYN_PT_SIZE * kFftDynHeadBytes> >(in, m_fftKernel.in[0]);
        connect<window<TP_WINDOW_VSIZE * sizeof(TT_DATA) + TP_DYN_PT_SIZE * kFftDynHeadBytes> >(m_fftKernel.out[0],
                                                                                                out);

        // Specify mapping constraints
        runtime<ratio>(m_fftKernel) = 0.4;

        // Source files
        source(m_fftKernel) = "fft_ifft_dit_1ch_ref.cpp";
        headers(m_fftKernel) = {"fft_ifft_dit_1ch_ref.hpp"};
    };
};
}
}
}
}
}
#endif // _DSPLIB_FFT_IFFT_DIT_1CH_REF_GRAPH_HPP_
