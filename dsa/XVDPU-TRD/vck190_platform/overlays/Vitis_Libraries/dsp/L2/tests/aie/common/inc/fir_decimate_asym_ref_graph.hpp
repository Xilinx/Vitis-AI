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
#ifndef _DSPLIB_FIR_DECIMATE_ASYM_REF_GRAPH_HPP_
#define _DSPLIB_FIR_DECIMATE_ASYM_REF_GRAPH_HPP_

#include <adf.h>
#include <vector>
#include "fir_decimate_asym_ref.hpp"
#include "fir_ref_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace decimate_asym {
using namespace adf;

// base class description and specialization for static coeffs, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN = 1,
          unsigned int TP_USE_COEFF_RELOAD = 0,
          unsigned int TP_NUM_OUTPUTS = 1>
class fir_decimate_asym_ref_graph : public graph {
   public:
    port<input> in;
    port<output> out;

    // FIR Kernel
    kernel m_firKernel;

    // Constructor
    fir_decimate_asym_ref_graph(const std::vector<TT_COEFF>& taps) {
        printf("==============================\n");
        printf("== FIR_DECIMATE_ASYM REF Graph - Static Coefficients\n");
        printf("==============================\n");

        // Create FIR class
        m_firKernel =
            kernel::create_object<fir_decimate_asym_ref<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT,
                                                        TP_RND, TP_INPUT_WINDOW_VSIZE, USE_COEFF_RELOAD_FALSE, 1> >(
                taps);

        // Make connections
        // Size of window in Bytes.
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernel.in[0]);
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) / TP_DECIMATE_FACTOR> >(m_firKernel.out[0], out);

        // Specify mapping constraints
        runtime<ratio>(m_firKernel) = 0.4;

        // Source files
        source(m_firKernel) = "fir_decimate_asym_ref.cpp";
        headers(m_firKernel) = {"fir_decimate_asym_ref.hpp"};
    };
};

// specialization for static coeffs, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN>
class fir_decimate_asym_ref_graph<TT_DATA,
                                  TT_COEFF,
                                  TP_FIR_LEN,
                                  TP_DECIMATE_FACTOR,
                                  TP_SHIFT,
                                  TP_RND,
                                  TP_INPUT_WINDOW_VSIZE,
                                  TP_CASC_LEN,
                                  USE_COEFF_RELOAD_FALSE,
                                  2> : public graph {
   public:
    port<input> in;
    port<output> out;
    port<output> out2;

    // FIR Kernel
    kernel m_firKernel;

    // Constructor
    fir_decimate_asym_ref_graph(const std::vector<TT_COEFF>& taps) {
        printf("==============================\n");
        printf("== FIR_DECIMATE_ASYM REF Graph - Static Coefficients\n");
        printf("==============================\n");

        // Create FIR class
        m_firKernel =
            kernel::create_object<fir_decimate_asym_ref<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT,
                                                        TP_RND, TP_INPUT_WINDOW_VSIZE, USE_COEFF_RELOAD_FALSE, 2> >(
                taps);

        // Make connections
        // Size of window in Bytes.
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernel.in[0]);
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) / TP_DECIMATE_FACTOR> >(m_firKernel.out[0], out);
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) / TP_DECIMATE_FACTOR> >(m_firKernel.out[1], out2);

        // Specify mapping constraints
        runtime<ratio>(m_firKernel) = 0.4;

        // Source files
        source(m_firKernel) = "fir_decimate_asym_ref.cpp";
        headers(m_firKernel) = {"fir_decimate_asym_ref.hpp"};
    };
};

// specialization for reload coeffs, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN>
class fir_decimate_asym_ref_graph<TT_DATA,
                                  TT_COEFF,
                                  TP_FIR_LEN,
                                  TP_DECIMATE_FACTOR,
                                  TP_SHIFT,
                                  TP_RND,
                                  TP_INPUT_WINDOW_VSIZE,
                                  TP_CASC_LEN,
                                  USE_COEFF_RELOAD_TRUE,
                                  1> : public graph {
   public:
    port<input> in;
    port<output> out;
    port<input> coeff;

    // FIR Kernel
    kernel m_firKernel;

    // Constructor
    fir_decimate_asym_ref_graph() {
        printf("==============================\n");
        printf("== FIR_DECIMATE_ASYM REF Graph - Reloadable coefficients\n");
        printf("==============================\n");

        // Create FIR class
        m_firKernel =
            kernel::create_object<fir_decimate_asym_ref<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT,
                                                        TP_RND, TP_INPUT_WINDOW_VSIZE, USE_COEFF_RELOAD_TRUE, 1> >();

        // Make connections
        // Size of window in Bytes.
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernel.in[0]);
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) / TP_DECIMATE_FACTOR> >(m_firKernel.out[0], out);
        connect<parameter>(coeff, async(m_firKernel.in[1]));

        // Specify mapping constraints
        runtime<ratio>(m_firKernel) = 0.4;

        // Source files
        source(m_firKernel) = "fir_decimate_asym_ref.cpp";
        headers(m_firKernel) = {"fir_decimate_asym_ref.hpp"};
    };
};

// specialization for reload coeffs, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN>
class fir_decimate_asym_ref_graph<TT_DATA,
                                  TT_COEFF,
                                  TP_FIR_LEN,
                                  TP_DECIMATE_FACTOR,
                                  TP_SHIFT,
                                  TP_RND,
                                  TP_INPUT_WINDOW_VSIZE,
                                  TP_CASC_LEN,
                                  USE_COEFF_RELOAD_TRUE,
                                  2> : public graph {
   public:
    port<input> in;
    port<output> out;
    port<output> out2;
    port<input> coeff;

    // FIR Kernel
    kernel m_firKernel;

    // Constructor
    fir_decimate_asym_ref_graph() {
        printf("==============================\n");
        printf("== FIR_DECIMATE_ASYM REF Graph - Reloadable coefficients\n");
        printf("==============================\n");

        // Create FIR class
        m_firKernel =
            kernel::create_object<fir_decimate_asym_ref<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT,
                                                        TP_RND, TP_INPUT_WINDOW_VSIZE, USE_COEFF_RELOAD_TRUE, 2> >();

        // Make connections
        // Size of window in Bytes.
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernel.in[0]);
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) / TP_DECIMATE_FACTOR> >(m_firKernel.out[0], out);
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) / TP_DECIMATE_FACTOR> >(m_firKernel.out[1], out2);
        connect<parameter>(coeff, async(m_firKernel.in[1]));

        // Specify mapping constraints
        runtime<ratio>(m_firKernel) = 0.4;

        // Source files
        source(m_firKernel) = "fir_decimate_asym_ref.cpp";
        headers(m_firKernel) = {"fir_decimate_asym_ref.hpp"};
    };
};
}
}
}
}
}
#endif // _DSPLIB_FIR_DECIMATE_ASYM_REF_GRAPH_HPP_
