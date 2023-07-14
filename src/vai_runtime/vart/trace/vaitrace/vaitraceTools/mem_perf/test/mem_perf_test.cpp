/**
 * Copyright 2022-2023 Advanced Micro Devices Inc..
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
#include "apm.hpp"

int main(int argc, char* argv[]) {
  MEM_PERF* mem_perf;
  std::string noc_type = "noc";
  std::string apm_type = "apm";

  int addr[] = {};
  int len = 4;

  if (argc < 2) {
    std::cout << "please input parameter: noc or apm" << std::endl;
    return 0;
  }

  std::string mem_type = std::string(argv[1]);
  std::cout << "will do sample: " << mem_type << std::endl;

  if (mem_type == noc_type) {
    mem_perf = new NOC(1000, addr, len);

  } else if (mem_type == apm_type) {
    mem_perf = new APM(1);

  } else {
    std::cout << "not support type: " << mem_type << std::endl;
    return 0;
  }

  auto t1 = std::chrono::steady_clock::now().time_since_epoch();
  double t2 = std::chrono::duration_cast<
                  std::chrono::duration<double, std::ratio<1, 1>>>(t1)
                  .count();
  printf("%.7f\n", t2);

  mem_perf->start_collect(0.01);
  std::this_thread::sleep_for(std::chrono::seconds(5));
  mem_perf->stop_collect();

  printf("noc record counter = %d\n", mem_perf->record_counter);

  for (int j = 0; j < 8; j++) {
    printf("the value of noc.data = %d\n", mem_perf->data[0].data[j]);
  }

  return EXIT_SUCCESS;
}
