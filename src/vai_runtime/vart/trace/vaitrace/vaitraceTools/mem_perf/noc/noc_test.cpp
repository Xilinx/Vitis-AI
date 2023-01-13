/**
 * Copyright 2022 Xilinx Inc.
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

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <thread>
#include <inttypes.h>
#include "noc.hpp"

int main(int argc, char* argv[]) {
  int addr[4] = {0xf6070000, 0xf6210000, 0xf6380000, 0xf64f0000};

  MEM_PERF* noc = new NOC(1000, addr);

  auto t1 = std::chrono::steady_clock::now().time_since_epoch();
  double t2 = std::chrono::duration_cast<
                  std::chrono::duration<double, std::ratio<1, 1>>>(t1)
                  .count();
  printf("%.7f\n", t2);

  noc->start_collect(0.01);
  std::this_thread::sleep_for(std::chrono::seconds(5));
  noc->stop_collect();

  printf("noc record counter = %d\n", noc->record_counter);

  for (int j = 0; j < 8; j++) {
    printf("the value of noc.data = %d\n", noc->data[0].data[j]);
  }

  return EXIT_SUCCESS;
}
