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
/*
This file holds the body of the test harness for the widger api cast
reference model graph
*/

#include <stdio.h>
#include "test.hpp"

#if (NUM_INPUTS == 1)
#if (NUM_OUTPUT_CLONES == 1)
simulation::platform<1, 1> platform(QUOTE(INPUT_FILE), QUOTE(OUTPUT_FILE));
#elif (NUM_OUTPUT_CLONES == 2)
simulation::platform<1, 2> platform(QUOTE(INPUT_FILE), QUOTE(OUTPUT_FILE), QUOTE(OUTPUT_FILE2));
#elif (NUM_OUTPUT_CLONES == 3)
simulation::platform<1, 3> platform(QUOTE(INPUT_FILE), QUOTE(OUTPUT_FILE), QUOTE(OUTPUT_FILE2), QUOTE(OUTPUT_FILE3));
#elif (NUM_OUTPUT_CLONES == 4)
simulation::platform<1, 4> platform(
    QUOTE(INPUT_FILE), QUOTE(OUTPUT_FILE), QUOTE(OUTPUT_FILE2), QUOTE(OUTPUT_FILE3), QUOTE(OUTPUT_FILE4));
#endif
#elif (NUM_INPUTS == 2)
#if (NUM_OUTPUT_CLONES == 1)
simulation::platform<2, 1> platform(QUOTE(INPUT_FILE), QUOTE(INPUT_FILE2), QUOTE(OUTPUT_FILE));
#elif (NUM_OUTPUT_CLONES == 2)
simulation::platform<2, 2> platform(QUOTE(INPUT_FILE), QUOTE(INPUT_FILE2), QUOTE(OUTPUT_FILE), QUOTE(OUTPUT_FILE2));
#elif (NUM_OUTPUT_CLONES == 3)
simulation::platform<2, 3> platform(
    QUOTE(INPUT_FILE), QUOTE(INPUT_FILE2), QUOTE(OUTPUT_FILE), QUOTE(OUTPUT_FILE2), QUOTE(OUTPUT_FILE3));
#elif (NUM_OUTPUT_CLONES == 4)
simulation::platform<2, 4> platform(QUOTE(INPUT_FILE),
                                    QUOTE(INPUT_FILE2),
                                    QUOTE(OUTPUT_FILE),
                                    QUOTE(OUTPUT_FILE2),
                                    QUOTE(OUTPUT_FILE3),
                                    QUOTE(OUTPUT_FILE4));
#endif
#endif

xf::dsp::aie::testcase::test_graph widgetTestHarness;

// Connect platform to uut
connect<> net_in0(platform.src[0], widgetTestHarness.in[0]);
#if (NUM_INPUTS > 1)
connect<> net_in1(platform.src[1], widgetTestHarness.in[1]);
#endif

connect<> net_out0(widgetTestHarness.out[0], platform.sink[0]);
#if (NUM_OUTPUT_CLONES > 1)
connect<> net_out2(widgetTestHarness.out[1], platform.sink[1]);
#endif
#if (NUM_OUTPUT_CLONES > 2)
connect<> net_out3(widgetTestHarness.out[2], platform.sink[2]);
#endif
#if (NUM_OUTPUT_CLONES > 3)
connect<> net_out4(widgetTestHarness.out[3], platform.sink[3]);
#endif

int main(void) {
    printf("\n");
    printf("========================\n");
    printf("UUT: ");
    printf(QUOTE(UUT_GRAPH));
    printf("\n");
    printf("========================\n");
    printf("Number of samples   = %d \n", WINDOW_VSIZE);
    switch (IN_API) {
        case 0:
            printf("Input API   = window\n");
            break;
        case 1:
            printf("Input API   = stream\n");
            break;
        default:
            printf("Input API unrecognised = %d\n", IN_API);
            break;
    };
    switch (OUT_API) {
        case 0:
            printf("Output API   = window\n");
            break;
        case 1:
            printf("Output API   = stream\n");
            break;
        default:
            printf("Output API unrecognised = %d\n", OUT_API);
            break;
    };
    printf("Data type       = ");
    printf(QUOTE(DATA_TYPE));
    printf("\n");
    printf("NUM_OUTPUT_CLONES     = %d \n", NUM_OUTPUT_CLONES);
    printf("\n");

    widgetTestHarness.init();
    widgetTestHarness.run(NITER);
    widgetTestHarness.end();

    return 0;
}
