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
This file is the test harness for the fft_ifft_dit_1ch graph class.
*/

#include <stdio.h>
#include "test.hpp"

simulation::platform<1, 1> platform(QUOTE(INPUT_FILE), QUOTE(OUTPUT_FILE));

xf::dsp::aie::testcase::test_graph fft;

connect<> net0(platform.src[0], fft.in);
connect<> net1(fft.out, platform.sink[0]);

int main(void) {
    printf("\n");
    printf("========================\n");
    printf("UUT: ");
    printf(QUOTE(UUT_GRAPH));
    printf("\n");
    printf("========================\n");
    printf("Input samples      = %d \n", INPUT_SAMPLES);
    printf("Output samples     = %d \n", OUTPUT_SAMPLES);
    printf("Point Size         = %d \n", POINT_SIZE);
    printf("FFT/nIFFT          = %d \n", FFT_NIFFT);
    printf("Shift              = %d \n", SHIFT);
    printf("Kernels            = %d \n", CASC_LEN);
    printf("Dynamic point size = %d \n", DYN_PT_SIZE);
    printf("Window Size        = %d \n", WINDOW_VSIZE);
    printf("Data type          = ");
    printf(QUOTE(DATA_TYPE));
    printf("\n");
    printf("Twiddle type       = ");
    printf(QUOTE(TWIDDLE_TYPE));
    printf("\n");

    fft.init();
    fft.run(NITER);
    fft.end();

    return 0;
}
