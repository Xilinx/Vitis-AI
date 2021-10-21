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
This file holds the body of the test harness for symmetric decimator FIR filter
reference model graph
*/

#include <stdio.h>
#include "test.hpp"

#if (NUM_OUTPUTS == 1)
simulation::platform<1, 1> platform(QUOTE(INPUT_FILE), QUOTE(OUTPUT_FILE));
#else
simulation::platform<1, 2> platform(QUOTE(INPUT_FILE), QUOTE(OUTPUT_FILE), QUOTE(OUTPUT_FILE2));
#endif

xf::dsp::aie::testcase::test_graph filter;

connect<> net0(platform.src[0], filter.in);
connect<> net1(filter.out, platform.sink[0]);
#if (NUM_OUTPUTS == 2)
connect<> net2(filter.out2, platform.sink[1]);
#endif

int main(void) {
    printf("\n");
    printf("========================\n");
    printf("UUT: ");
    printf(QUOTE(UUT_GRAPH));
    printf("\n");
    printf("========================\n");
    printf("Input samples     = %d \n", INPUT_SAMPLES);
    printf("Input margin      = %lu \n", INPUT_MARGIN(FIR_LEN, DATA_TYPE));
    printf("Output samples    = %d \n", OUTPUT_SAMPLES);
    printf("FIR Length        = %d \n", FIR_LEN);
    printf("Decimation Factor = %d \n", DECIMATE_FACTOR);
    printf("Shift             = %d \n", SHIFT);
    printf("Round mode        = %d \n", ROUND_MODE);
    printf("Dual Inputs       = %d \n", DUAL_IP);
    printf("Data type         = ");
    printf(QUOTE(DATA_TYPE));
    printf("\n");
    printf("Coeff type      = ");
    printf(QUOTE(COEFF_TYPE));
    printf("\nCoeff reload  = %d \n", USE_COEFF_RELOAD);
    printf("CASC_LEN        = %d \n", CASC_LEN);
    printf("NUM_OUTPUTS     = %d \n", NUM_OUTPUTS);
    printf("\n");

    filter.init();
#if (USE_COEFF_RELOAD == 1)
    filter.update(filter.coeff, filter.m_taps[0], (FIR_LEN + 1) / 2);
    filter.run(NITER / 2);
    filter.wait();
    filter.update(filter.coeff, filter.m_taps[1], (FIR_LEN + 1) / 2);
    filter.run(NITER / 2);
#else
    filter.run(NITER);
#endif

    filter.end();

    return 0;
}
