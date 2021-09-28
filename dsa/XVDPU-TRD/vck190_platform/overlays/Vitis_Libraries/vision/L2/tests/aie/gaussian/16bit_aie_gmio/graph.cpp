/*
 * Copyright 2021 Xilinx, Inc.
 *
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

#include "graph.h"

GMIO gmioIn[1] = {
    GMIO("gmioIn1", 256, 1000),
};
GMIO gmioOut[1] = {GMIO("gmioOut1", 256, 1000)};

// connect dataflow graph to simulation platform
simulation::platform<1, 1> platform(&gmioIn[0], &gmioOut[0]);

// Graph object
myGraph gaussian_graph;

// Virtual platform connectivity
connect<> net0(platform.src[0], gaussian_graph.inptr);
connect<> net1(gaussian_graph.outptr, platform.sink[0]);

#define SRS_SHIFT 10
float kData[9] = {0.01134, 0.08382, 0.01134, 0.08382, 0.61932, 0.08382, 0.01134, 0.08382, 0.01134};

template <int SHIFT, int VECTOR_SIZE>
auto float2fixed_coeff(float data[9]) {
    // 3x3 kernel positions
    //
    // k0 k1 0 k2 0
    // k3 k4 0 k5 0
    // k6 k7 0 k8 0
    std::array<int16_t, VECTOR_SIZE> ret;
    ret.fill(0);
    for (int i = 0; i < 3; i++) {
        ret[5 * i + 0] = data[3 * i + 0] * (1 << SHIFT);
        ret[5 * i + 1] = data[3 * i + 1] * (1 << SHIFT);
        ret[5 * i + 3] = data[3 * i + 2] * (1 << SHIFT);
    }
    return ret;
}

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(int argc, char** argv) {
    gaussian_graph.init();
    gaussian_graph.update(gaussian_graph.kernelCoefficients, float2fixed_coeff<10, 16>(kData).data(), 16);
    gaussian_graph.run(1);
    gaussian_graph.end();
    return 0;
}
#endif
