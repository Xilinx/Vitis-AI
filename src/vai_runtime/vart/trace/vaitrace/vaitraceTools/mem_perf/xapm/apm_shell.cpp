
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

#ifdef BOARD_ULTRA96
#  define APM_CLK_FREQ 266666656
#else
#  define APM_CLK_FREQ 533333333
#endif
extern "C" {
APM apm(1);

int apm_start(double interval_in_sec) {
  apm.start_collect(interval_in_sec);

  return EXIT_SUCCESS;
}

int apm_stop() {
  apm.stop_collect();

  return EXIT_SUCCESS;
}

int apm_pop_data(struct record* d) { return apm.pop_data(d); }
}
