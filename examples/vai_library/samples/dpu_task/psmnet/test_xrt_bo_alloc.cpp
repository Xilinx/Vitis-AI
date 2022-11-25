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
using namespace std;
#include "vai_aie_task_handler.hpp"

int main(int argc, char** argv) {
  auto input_bytes = 1024 * 1024 * 100;
  auto h = vai_aie_task_handler(argv[1]);
  auto bo = xrtBOAlloc(h.dhdl, input_bytes, 0, 0);
  cout << std::hex << "0x" << xrtBOAddress(bo) << "}";
  xrtBOFree(bo);
}
