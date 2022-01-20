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

PLIO* in1 = new PLIO("DataIn1", adf::plio_32_bits, "data/input.txt");
PLIO* out1 = new PLIO("DataOut1", adf::plio_32_bits, "data/output.txt");

simulation::platform<1, 1> platform(in1, out1);

two_node_pipeline filter_graph;

connect<> net0(platform.src[0], filter_graph.in);
connect<> net1(filter_graph.out, platform.sink[0]);

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(int argc, char** argv) {
    filter_graph.init();
    filter_graph.run(1);
    filter_graph.end();
    return 0;
}

#endif
