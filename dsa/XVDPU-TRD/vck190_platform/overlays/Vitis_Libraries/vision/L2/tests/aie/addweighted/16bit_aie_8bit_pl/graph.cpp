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

PLIO* in1 = new PLIO("DataIn1", adf::plio_64_bits, "data/input1.txt");
PLIO* in2 = new PLIO("DataIn2", adf::plio_64_bits, "data/input2.txt");
PLIO* out1 = new PLIO("DataOut1", adf::plio_64_bits, "data/output.txt");

// connect dataflow graph to simulation platform
simulation::platform<2, 1> platform(in1, in2, out1);

// instantiate adf dataflow graph to compute weighted moving average
addweightedGraph mygraph;

connect<> net0(platform.src[0], mygraph.in1);
connect<> net1(platform.src[1], mygraph.in2);
connect<> net2(mygraph.out, platform.sink[0]);

// initialize and run the dataflow graph
#if defined(__AIESIM__) || defined(__X86SIM__)
int main(int argc, char** argv) {
    float alpha = 2.0f;
    float beta = 3.0f;
    float gamma = 4.0f;
    mygraph.init();
    mygraph.update(mygraph.alpha, alpha);
    mygraph.update(mygraph.beta, beta);
    mygraph.update(mygraph.gamma, gamma);
    mygraph.run(1);
    mygraph.end();

    return 0;
}
#endif
