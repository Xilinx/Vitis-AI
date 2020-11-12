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
#include "UniLog/UniLog.hpp"
#include "xir/attrs/attrs.hpp"
#include "xir/util/tool_function.hpp"

int main(int argc, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);
  auto attr = xir::Attrs::create();
  attr->set_attr<std::string>("name", "xir")
      ->set_attr<std::uint32_t>("num", 11);
  UNI_LOG_INFO << attr->debug_info();
  UNI_LOG_INFO << xir::to_string(attr->get_keys());
  UNI_LOG_INFO << "name= " << attr->get_attr<std::string>("name") << " and "
               << "num=" << attr->get_attr<std::uint32_t>("num");
  // test cmp
  CHECK_EQ(xir::Attrs::cmp(xir::any(1), xir::any(1)), 1);
  CHECK_EQ(xir::Attrs::cmp(xir::any(1), xir::any(2)), 0);
  CHECK_EQ(xir::Attrs::cmp(xir::any(1), xir::any(nullptr)), 0);
  CHECK_EQ(xir::Attrs::cmp(xir::any(nullptr), xir::any(nullptr)), 1);
  return 0;
}
