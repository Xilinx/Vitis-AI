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

simulation::platform<1, 1> platform(QUOTE(INPUT_FILE), QUOTE(OUTPUT_FILE));

xf::dsp::aie::testcase::test_graph widgetTestHarness;

// Connect platform to uut
connect<> net_in0(platform.src[0], widgetTestHarness.in);

connect<> net_out0(widgetTestHarness.out, platform.sink[0]);

int main(void) {
    printf("\n");
    printf("========================\n");
    printf("UUT: ");
    printf(QUOTE(UUT_GRAPH));
    printf("\n");
    printf("========================\n");
    printf("Number of samples   = %d \n", WINDOW_VSIZE);
    printf("Data type       = ");
    printf(QUOTE(DATA_TYPE));
    printf("\n");
    printf("Data out type       = ");
    printf(QUOTE(DATA_OUT_TYPE));
    printf("\n");
    printf("\n");

    widgetTestHarness.init();
    widgetTestHarness.run(NITER);
    widgetTestHarness.end();

    return 0;
}
