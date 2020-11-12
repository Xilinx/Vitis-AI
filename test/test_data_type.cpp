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
#include "xir/util/data_type.hpp"
#include "xir/util/tool_function.hpp"

int main(int argc, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);
  UNI_LOG_INFO << xir::create_data_type<float>().to_string();
  UNI_LOG_INFO << xir::create_data_type<double>().to_string();
  UNI_LOG_INFO << xir::create_data_type<int>().to_string();
  UNI_LOG_INFO << xir::create_data_type<std::int32_t>().to_string();
  UNI_LOG_INFO << xir::create_data_type<std::int64_t>().to_string();
  UNI_LOG_INFO << xir::create_data_type<std::uint32_t>().to_string();
  UNI_LOG_INFO << xir::create_data_type<std::uint64_t>().to_string();
  UNI_LOG_INFO << xir::create_data_type<long>().to_string();

  UNI_LOG_INFO << xir::DataType{"INT32"}.to_string();
  // UNI_LOG_INFO << xir::DataType{"INT3 "}.to_string();
  UNI_LOG_INFO << xir::DataType{"INT33"}.to_string();
  // UNI_LOG_INFO << xir::DataType{"FLOAT"}.to_string();

  return 0;
}
