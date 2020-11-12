/*
 * Copyright 2019 Xilinx Inc.
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

#include "demo_graph.hpp"
#include "resnet.hpp"

using namespace std;
using namespace xir;

int main(int args, char** argv) {
  // BuildGraphDemo bgd;

  // auto g = bgd.build_graph_demo();

  Resnet res;

  auto g = res.Build();

  g->get_root_subgraph();
  g->serialize("graph.serial.pb");

  return 0;
}
