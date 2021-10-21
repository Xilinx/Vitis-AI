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

GMIO gmioIn[2] = {GMIO("gmioIn1", 256, 1000), GMIO("gmioIn2", 256, 1000)};
GMIO gmioOut[1] = {GMIO("gmioOut1", 256, 1000)};

// connect dataflow graph to simulation platform
simulation::platform<2, 1> platform(&gmioIn[0], &gmioIn[1], &gmioOut[0]);

myGraph accumw_graph;

connect<> net0(platform.src[0], accumw_graph.inprt1);
connect<> net1(platform.src[1], accumw_graph.inprt2);

connect<> net2(accumw_graph.outprt, platform.sink[0]);

#if defined(__AIESIM__) || defined(__NEW_X86Sim__)
int main(int argc, char** argv) {
    float alpha = 0.5f;
    accumw_graph.init();
    accumw_graph.update(accumw_graph.alpha, alpha);
    accumw_graph.run(1);
    accumw_graph.end();
    return 0;
}

#endif
