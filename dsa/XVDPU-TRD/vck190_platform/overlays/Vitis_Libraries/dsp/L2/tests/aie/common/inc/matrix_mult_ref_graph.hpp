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
#ifndef MATRIX_MULT_REF_HPP
#define MATRIX_MULT_REF_HPP

// This file holds the definition header of thematrix mult reference model graph class

#include <adf.h>
#include <vector>

#include "matrix_mult_ref.hpp"
#include "fir_ref_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace blas {
namespace matrix_mult {
using namespace adf;

template <typename TT_DATA_A,
          typename TT_DATA_B,
          unsigned int TP_DIM_A,
          unsigned int TP_DIM_AB,
          unsigned int TP_DIM_B,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_DIM_A_LEADING = ROW_MAJOR,
          unsigned int TP_DIM_B_LEADING = COL_MAJOR,
          unsigned int TP_DIM_OUT_LEADING = ROW_MAJOR,
          unsigned int TP_ADD_TILING_A = 1,     // not used - just to match UUT.
          unsigned int TP_ADD_TILING_B = 1,     // not used - just to match UUT.
          unsigned int TP_ADD_DETILING_OUT = 1, // not used - just to match UUT.
          unsigned int TP_INPUT_WINDOW_VSIZE_A = TP_DIM_A* TP_DIM_AB,
          unsigned int TP_INPUT_WINDOW_VSIZE_B = TP_DIM_B* TP_DIM_AB,
          unsigned int TP_CASC_LEN = 1 // not used - just to match UUT.
          >
class matrix_mult_ref_graph : public graph {
   public:
    // port<input> in[2];
    port<input> inA;
    port<input> inB;
    port<output> out;

    // FIR Kernel
    kernel m_firKernel;

    // Constructor
    matrix_mult_ref_graph() {
        printf("===========================\n");
        printf("==    MATRIX MULT REF   == \n");
        printf("===========================\n");

        // Create FIR class
        m_firKernel = kernel::create_object<
            matrix_mult_ref<TT_DATA_A, TT_DATA_B, TP_DIM_A, TP_DIM_AB, TP_DIM_B, TP_SHIFT, TP_RND, TP_DIM_A_LEADING,
                            TP_DIM_B_LEADING, TP_DIM_OUT_LEADING, TP_INPUT_WINDOW_VSIZE_A, TP_INPUT_WINDOW_VSIZE_B> >();
        printf("Created object");
        // Make connections
        // Size of window in Bytes.
        connect<window<TP_INPUT_WINDOW_VSIZE_A * sizeof(TT_DATA_A)> >(inA, m_firKernel.in[0]);
        connect<window<TP_INPUT_WINDOW_VSIZE_B * sizeof(TT_DATA_B)> >(inB, m_firKernel.in[1]);
        connect<window<(TP_INPUT_WINDOW_VSIZE_A / TP_DIM_AB) * (TP_INPUT_WINDOW_VSIZE_B / TP_DIM_AB) *
                       sizeof(outType_t<TT_DATA_A, TT_DATA_B>)> >(m_firKernel.out[0], out);
        printf("connected window");
        // Specify mapping constraints
        runtime<ratio>(m_firKernel) = 0.4;
        printf("entering source");
        // Source files
        source(m_firKernel) = "matrix_mult_ref.cpp";
        printf("finished constructing");
    };
};
}
}
}
}
}
#endif // MATRIX_MULT_REF_HPP
