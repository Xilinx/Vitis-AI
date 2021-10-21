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

PLIO* in_0 = new PLIO("DataIn1", adf::plio_64_bits, "data/input.txt");
PLIO* out_0 = new PLIO("DataOut1", adf::plio_64_bits, "data/output.txt");

// connect dataflow graph to simulation platform
simulation::platform<1, 1> platform(in_0, out_0);

// instantiate adf dataflow graph
thresholdGraph mygraph;

connect<> net0(platform.src[0], mygraph.in1);

connect<> net1(mygraph.out1, platform.sink[0]);

// initialize and run the dataflow graph
#if defined(__AIESIM__) || defined(__X86SIM__)

int main(int argc, char** argv) {
    int16_t thresh_val = 100;
    int16_t max_val = 50;
    mygraph.init();
    mygraph.update(mygraph.threshVal, thresh_val);
    mygraph.update(mygraph.maxVal, max_val);
    mygraph.run(1);
    mygraph.end();
    return 0;
}
#endif
