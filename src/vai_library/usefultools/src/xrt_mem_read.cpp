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
#include <glog/logging.h>
#include <xrt.h>

#include <iostream>
using namespace std;

#include "parse_value.hpp"

int main(int argc, char* argv[]) {
  auto arg_addr = string(argv[1]);
  auto arg_size = string(argv[2]);
  unsigned long addr = 0;
  unsigned long size = 0;
  parse_value(arg_addr, addr);
  parse_value(arg_size, size);
  auto deviceIndex = 0;
  auto handle = xclOpen(deviceIndex, NULL, XCL_INFO);
  auto buf = std::vector<char>(size);
  CHECK_EQ(buf.size(), size);
  auto flags = 0;
  auto ok = xclUnmgdPread(handle, flags, &buf[0], size, addr);
  PCHECK(ok == 0) << "";
  CHECK(cout.write(&buf[0], size).good());
  xclClose(handle);
  return 0;
}
