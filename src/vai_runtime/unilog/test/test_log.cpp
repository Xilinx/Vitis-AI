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
#include <memory>
#include <string>
using namespace std;

#include "UniLog/UniLog.hpp"

REGISTER_ERROR_CODE(UNILOG_EXAMPLE_ERRORCODE, "unilog example error code.",
                    "unilog example debug info.");

int main(int argc, char *argv[]) {
  int a = 6;
  double b = 3.55;
  std::string c{"example"};
  int a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7, a8 = 8, a9 = 9,
      a10 = 10, a11 = 11, a12 = 12, a13 = 13, a14 = 14, a15 = 15, a16 = 16;
  std::string e1{"example1"}, e2{"example2"}, e3{"example3"}, e4 = {"example4"};
//    UNI_LOG_CHECK_EQ(a, b, UNILOG_EXAMPLE_ERRORCODE) << UNI_LOG_VALUES(a, b);
  UNI_LOG_CHECK_NE(a, b, UNILOG_EXAMPLE_ERRORCODE) << UNI_LOG_VALUES(a, b);

  UNI_LOG_CHECK_NE(a1 + a16 + a2 + a15 + a3 + a14,
                   a4 + a13 + a5 + a12 + a6 + a11, UNILOG_EXAMPLE_ERRORCODE)
      << "[values:]"
      << UNI_LOG_VALUES(a1, e1, a2, e2, a3, e3, a4, a5, a6, a7, a8, e4, a9, a10,
                        a11, a12);
//  UNI_LOG_CHECK_EQ(a1 + a16 + a2 + a15 + a3 + a14,
//                   a4 + a13 + a5 + a12 + a6 + a11, UNILOG_EXAMPLE_ERRORCODE)
//      << "equal values: "
//      << UNI_LOG_VALUES(a1, e1, a2, e2, a3, e3, a4, a5, a6, a7, a8, e4, a9, a10,
//                        a11, a12);
}
