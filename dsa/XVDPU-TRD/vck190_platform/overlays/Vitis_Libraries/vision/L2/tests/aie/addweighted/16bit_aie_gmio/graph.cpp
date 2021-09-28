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

addweightedGraph mygraph;

connect<> net0(platform.src[0], mygraph.in1);
connect<> net1(platform.src[1], mygraph.in2);
connect<> net2(mygraph.out, platform.sink[0]);

#if defined(__AIESIM__) || defined(__NEW_X86Sim__)
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
