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
#include "xir/attrs/attrs.hpp"
#include "xir/graph/graph.hpp"
#include "xir/util/cxxopts.hpp"

int main(int argc, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);

  cxxopts::Options parser_def(argv[0], "Dump image");
  parser_def.add_options()                                             //
      ("i,input", "input xmodel", cxxopts::value<std::string>(),       //
       "FILE")                                                         //
      ("o,output", "output image",                                     //
       cxxopts::value<std::string>(),                                  //
       "FILE")                                                         //
      ("f,format", "image format [svg, png]",                          //
       cxxopts::value<std::string>()->default_value("svg"), "FORMAT")  //
      ("h,help", "show help");
  parser_def.parse_positional({"input"});

  // parse the input
  cxxopts::ParseResult parse_results = parser_def.parse(argc, argv);
  if ((argc == 0) || parse_results.count("help")) {
    std::cout << parser_def.help();
    return 0;
  }

  auto xmodel = parse_results["input"].as<std::string>();
  auto format = parse_results["format"].as<std::string>();
  UNI_LOG_INFO << "Read xmodel file " << xmodel << " ...";
  auto graph = xir::Graph::deserialize(xmodel);
  UNI_LOG_INFO << "Graph name: " << graph->get_name();
  UNI_LOG_INFO << "Output format: " << format;

  auto image = std::string{};
  if (parse_results.count("output") == 0) {
    image = graph->get_name() + "." + format;
  } else {
    image = parse_results["output"].as<std::string>();
  }

  UNI_LOG_INFO << "Write image file " << image << " ...";
  graph->visualize(image, format);

  return 0;
}
