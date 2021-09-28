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

// This file holds the definition of the test harness for the Widget API Cast graph.

#include <adf.h>
#include <vector>
#include "utils.hpp"

#include "uut_config.h"
#include "test_stim.hpp"

#define Q(x) #x
#define QUOTE(x) Q(x)

#ifndef UUT_GRAPH
#define UUT_GRAPH widget_api_cast_graph
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
    port<input> in[NUM_INPUTS];
    port<output> out[NUM_OUTPUT_CLONES];

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
        printf("IN_API            = %d \n", IN_API);
        printf("OUT_API           = %d \n", OUT_API);
        printf("NUM_INPUTS        = %d \n", NUM_INPUTS);
        printf("WINDOW_VSIZE      = %d \n", WINDOW_VSIZE);
        printf("NUM_OUTPUT_CLONES = %d \n", NUM_OUTPUT_CLONES);
        namespace dsplib = xf::dsp::aie;

        // Widget sub-graph
        dsplib::widget::api_cast::UUT_GRAPH<DATA_TYPE, IN_API, OUT_API, NUM_INPUTS, WINDOW_VSIZE, NUM_OUTPUT_CLONES>
            widgetGraph;

        // Make connections
        for (int i = 0; i < NUM_INPUTS; i++) {
            connect<>(in[i], widgetGraph.in[i]);
        }
        for (int i = 0; i < NUM_OUTPUT_CLONES; i++) {
            connect<>(widgetGraph.out[i], out[i]);
        }

#ifdef USING_UUT
        // For cases which use 3 or 4 output windows, contention and poor QoR is seen if the processor is at the edge of
        // the array since
        // the windows then share banks and contention occurs.
        location<kernel>(*widgetGraph.getKernels()) = tile(1, 1);
#endif

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
