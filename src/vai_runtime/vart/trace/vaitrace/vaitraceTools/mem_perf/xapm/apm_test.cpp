
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

#include "apm.hpp"

#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <thread>

#ifdef BOARD_ULTRA96
#  define APM_CLK_FREQ 266666656
#else
#  define APM_CLK_FREQ 533333333
#endif

int main(int argc, char* argv[]) {
  MEM_PERF* apm = new APM(1);

  auto t1 = std::chrono::steady_clock::now().time_since_epoch();
  double t2 = std::chrono::duration_cast<
                  std::chrono::duration<double, std::ratio<1, 1>>>(t1)
                  .count();
  printf("%.7f\n", t2);

  apm->start_collect(0.01);
  std::this_thread::sleep_for(std::chrono::seconds(5));
  apm->stop_collect();

  printf("the record counter = %d\n", apm->record_counter);
  std::cout << "the data size = " << sizeof(apm->data) / sizeof(apm->data[0])
            << std::endl;
  //    std::cout << "the time = " << apm->data[3].time << std::endl;
  /*
      for(int i = 0; i < 10; i++) {
              std::cout << apm->data[apm->record_counter].data[i] << std::endl;
      }
  */
  return EXIT_SUCCESS;
}
