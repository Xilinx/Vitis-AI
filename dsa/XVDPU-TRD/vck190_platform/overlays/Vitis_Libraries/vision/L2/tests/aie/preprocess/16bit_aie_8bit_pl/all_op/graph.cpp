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

PLIO* in1 = new PLIO("DataIn1", adf::plio_64_bits, "data/input.txt");
PLIO* out1 = new PLIO("DataOut1", adf::plio_64_bits, "data/output_all_op.txt");

simulation::platform<1, 1> platform(in1, out1);

preprocessgraph ppgraph;

connect<> net0(platform.src[0], ppgraph.in1);
connect<> net1(ppgraph.out, platform.sink[0]);

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(int argc, char** argv) {
    float alp = 72.25;
    float scale = 0.5;
    float gam = 4.25;
    ppgraph.init();
    ppgraph.update(ppgraph.alpha, alp);
#if OP_MODE == 1 || OP_MODE == 2
    ppgraph.update(ppgraph.beta, scale);
#endif
#if OP_MODE == 2
    ppgraph.update(ppgraph.gamma, gam);
#endif
    ppgraph.run(1);

    ppgraph.end();

    return 0;
}

#endif
