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

#include <iostream>
#include <string>
#include <vector>

#include "UniLog/UniLog.hpp"
#include "xir/graph/graph_template.hpp"

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);

  auto g = xir::GraphTemplate::create("test");
  UNI_LOG_INFO << "Graph name: " << g->get_name();
  auto w = g->add_op("weights", {"const"}, {});
  UNI_LOG_INFO << "Add Op " << w->get_name();
  auto i = g->add_op("input_data", {"data"}, {});
  UNI_LOG_INFO << "Add Op " << i->get_name();
  auto c =
      g->add_op("calc", {"conv2d", "depthwise-conv2d"}, {{w, ""}, {i, ""}});
  UNI_LOG_INFO << "Add Op " << c->get_name();

  UNI_LOG_INFO << "C in " << c->get_input_num() << " out "
               << c->get_fanout_num();

  g->visualize("test", "png");

  return 0;
}
