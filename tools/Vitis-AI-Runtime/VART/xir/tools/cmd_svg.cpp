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

#include "cmd_svg.hpp"

#include <iostream>
#include <sstream>
using namespace std;

CmdSvg::CmdSvg(const std::string& name) : Cmd(name) {}

int CmdSvg::main(int argc, char* argv[]) {
  if (argc < 3) {
    cout << help() << endl;
    return 1;
  }
  auto xmodel = std::string(argv[1]);
  auto svg = std::string(argv[2]);
  auto graph = xir::Graph::deserialize(xmodel);
  CHECK(graph != nullptr) << "cannot read graph" << xmodel;
  graph->visualize(svg, "svg");
  return 0;
}

std::string CmdSvg::help() const {
  std::ostringstream str;
  str << "xir " << get_name() << " <xmodel> <svg>\n\t"
      << "convert a xmodel to svg file.";
  return str.str();
}
