/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#include <iostream>
#include <xir/device_memory.hpp>
#include "parse_value.hpp"

using namespace std;
int main(int argc, char* argv[]) {
  auto arg_addr = std::string(argv[1]);
  auto arg_size = std::string(argv[2]);
  auto filename = std::string(argv[3]);
  unsigned long addr = 0;
  unsigned long size = 0;
  parse_value(arg_addr, addr);
  parse_value(arg_size, size);
  // auto buf = std::vector<char>(size);
  // CHECK_EQ(buf.size(), size);
  //  CHECK(cin.read(&buf[0], size).good());
  auto device_memory = xir::DeviceMemory::create((size_t)0ull);
  device_memory->save(filename, addr, size);
  return 0;
}
