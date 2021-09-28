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
#ifndef _DSPLIB_TEST_HPP_
#define _DSPLIB_TEST_HPP_

// This file holds the definition of the test harness for the Widget real2complex graph.

#include <adf.h>
#include <vector>
#include "utils.hpp"

#include "uut_config.h"
#include "test_stim.hpp"

#define Q(x) #x
#define QUOTE(x) Q(x)

#ifndef UUT_GRAPH
#define UUT_GRAPH widget_real2complex_graph
#endif

#include QUOTE(UUT_GRAPH.hpp)

using namespace adf;

namespace xf {
namespace dsp {
namespace aie {
namespace testcase {

class test_graph : public graph {
   private:
   public:
    port<input> in;
    port<output> out;

    // Constructor
    test_graph() {
        printf("========================\n");
        printf("== Widget test.hpp parameters: ");
        printf(QUOTE(UUT_GRAPH));
        printf("\n");
        printf("========================\n");
        printf("Data type         = ");
        printf(QUOTE(DATA_TYPE));
        printf("\n");
        printf("Data type         = ");
        printf(QUOTE(DATA_OUT_TYPE));
        printf("\n");
        printf("WINDOW_VSIZE      = %d \n", WINDOW_VSIZE);
        namespace dsplib = xf::dsp::aie;

        // Widget sub-graph
        dsplib::widget::real2complex::UUT_GRAPH<DATA_TYPE, DATA_OUT_TYPE, WINDOW_VSIZE> widgetGraph;

        // Make connections
        connect<>(in, widgetGraph.in);
        connect<>(widgetGraph.out, out);

#ifdef USING_UUT
#define CASC_LEN 1
// Report out for AIE Synthesizer QoR harvest
// Nothing to report
#endif
        printf("========================\n");
    };
};
}
}
}
};

#endif // _DSPLIB_TEST_HPP_
