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
#include "xir/util/tool_function.hpp"

int main(int argc, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);
  std::string name("__file_name");
  UNI_LOG_INFO << name;
  name = xir::add_prefix(name, "xir", "test");
  UNI_LOG_INFO << name;
  name = xir::add_suffix(name, "fix", "blob");
  UNI_LOG_INFO << name;
  name = xir::add_suffix(name, name);
  UNI_LOG_INFO << name;
  UNI_LOG_INFO << xir::to_string(xir::extract_xfix(name));
  name = xir::remove_xfix(name);
  UNI_LOG_INFO << name;
  return 0;
}
