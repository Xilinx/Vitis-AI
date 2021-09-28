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
#ifndef _DSPLIB_WIDGET_REAL2COMPLEX_REF_GRAPH_HPP_
#define _DSPLIB_WIDGET_REAL2COMPLEX_REF_GRAPH_HPP_

/*
This file holds the definition of the Widget Real2complex Reference model graph.
*/

#include <adf.h>
#include <vector>
#include "widget_real2complex_ref.hpp"
#include "fir_ref_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace widget {
namespace real2complex {
using namespace adf;

template <typename TT_DATA, typename TT_OUT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_real2complex_ref_graph : public graph {
   public:
    port<input> in;
    port<output> out;

    kernel m_kernel;

    // Constructor
    widget_real2complex_ref_graph() {
        printf("============================\n");
        printf("== WIDGET_REAL2COMPLEX REF Graph\n");
        printf("============================\n");
        printf("WINDOW_VSIZE      = %d\n", TP_WINDOW_VSIZE);

        m_kernel = kernel::create_object<widget_real2complex_ref<TT_DATA, TT_OUT_DATA, TP_WINDOW_VSIZE> >();
        // Specify mapping constraints
        runtime<ratio>(m_kernel) = 0.1; // Nominal figure. The real figure requires knowledge of the sample rate.
        // Source files
        source(m_kernel) = "widget_real2complex_ref.cpp";

        // make connections
        connect<window<TP_WINDOW_VSIZE * sizeof(TT_DATA)> >(in, m_kernel.in[0]);
        connect<window<TP_WINDOW_VSIZE * sizeof(TT_OUT_DATA)> >(m_kernel.out[0], out);
    };
};
}
}
}
}
}
#endif // _DSPLIB_WIDGET_REAL2COMPLEX_REF_GRAPH_HPP_
