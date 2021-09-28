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
#ifndef _DSPLIB_WIDGET_API_CAST_REF_GRAPH_HPP_
#define _DSPLIB_WIDGET_API_CAST_REF_GRAPH_HPP_

/*
This file holds the definition of the Widget API Cast Reference model graph.
*/

#include <adf.h>
#include <vector>
#include "widget_api_cast_ref.hpp"
#include "fir_ref_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace widget {
namespace api_cast {
using namespace adf;

template <typename TT_DATA,
          unsigned int TP_IN_API,
          unsigned int TP_OUT_API,
          unsigned int TP_NUM_INPUTS,
          unsigned int TP_WINDOW_VSIZE,
          unsigned int TP_NUM_OUTPUT_CLONES>
class widget_api_cast_ref_graph : public graph {
   public:
    port<input> in[TP_NUM_INPUTS];
    port<output> out[TP_NUM_OUTPUT_CLONES];

    kernel m_kernel;

    // Constructor
    widget_api_cast_ref_graph() {
        printf("============================\n");
        printf("== WIDGET API CAST REF Graph\n");
        printf("============================\n");
        printf("IN_API            = %d\n", TP_IN_API);
        printf("OUT_API           = %d\n", TP_OUT_API);
        printf("NUM_INPUTS        = %d\n", TP_NUM_INPUTS);
        printf("WINDOW_VSIZE      = %d\n", TP_WINDOW_VSIZE);
        printf("NUM_OUTPUT_CLONES = %d\n", TP_NUM_OUTPUT_CLONES);

        m_kernel = kernel::create_object<widget_api_cast_ref<TT_DATA, TP_IN_API, TP_OUT_API, TP_NUM_INPUTS,
                                                             TP_WINDOW_VSIZE, TP_NUM_OUTPUT_CLONES> >();
        // Specify mapping constraints
        runtime<ratio>(m_kernel) = 0.1; // Nominal figure. The real figure requires knowledge of the sample rate.
        // Source files
        source(m_kernel) = "widget_api_cast_ref.cpp";

        // make connections
        if (TP_IN_API == kWindowAPI) {
            for (int i = 0; i < TP_NUM_INPUTS; i++) {
                connect<window<TP_WINDOW_VSIZE * sizeof(TT_DATA)> >(in[i], m_kernel.in[i]);
            }
        } else if (TP_IN_API == kStreamAPI) {
            for (int i = 0; i < TP_NUM_INPUTS; i++) {
                connect<stream>(in[i], m_kernel.in[i]);
            }
        }
        if (TP_OUT_API == kWindowAPI) {
            for (int i = 0; i < TP_NUM_OUTPUT_CLONES; i++) {
                connect<window<TP_WINDOW_VSIZE * sizeof(TT_DATA)> >(m_kernel.out[i], out[i]);
            }
        } else if (TP_OUT_API == kStreamAPI) {
            for (int i = 0; i < TP_NUM_OUTPUT_CLONES; i++) {
                connect<stream>(m_kernel.out[i], out[i]);
            }
        }
    };
};
}
}
}
}
}
#endif // _DSPLIB_WIDGET_API_CAST_REF_GRAPH_HPP_
