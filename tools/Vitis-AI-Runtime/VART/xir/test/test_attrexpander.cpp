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
#include "xir/attrs/attr_def.hpp"
#include "xir/attrs/attr_expander_imp.hpp"

void print_attr_defs(xir::AttrExpander::Target target);

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);

  std::cout << "Op: ";
  print_attr_defs(xir::AttrExpander::Target::Op);

  std::cout << "Tensor: ";
  print_attr_defs(xir::AttrExpander::Target::Tensor);

  std::cout << "Subgraph: ";
  print_attr_defs(xir::AttrExpander::Target::Subgraph);

  return 0;
}

void print_attr_defs(xir::AttrExpander::Target target) {
  auto attr_defs = xir::attr_expander()->get_expanded_attrs(target);
  for (auto& def : attr_defs) {
    std::cout << def.name;
  }
  std::cout << std::endl;
}
