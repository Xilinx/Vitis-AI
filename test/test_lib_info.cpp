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
  UNI_LOG_INFO << "lib_name = " << xir::get_lib_name();
  UNI_LOG_INFO << "lib_id = " << xir::get_lib_id();
  auto lib_info = xir::get_lib_name() + " : " + xir::get_lib_id();
  UNI_LOG_INFO << "lib_info = " << lib_info;
  return 0;
}
