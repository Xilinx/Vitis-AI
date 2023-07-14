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
#include <iostream>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(SESSION_ID, "0")
DEF_ENV_PARAM_2(SESSION_NAME, "hello there", std::string)
#include <vitis/ai/profiling.hpp>

int main(int argc, char* argv[]) {
  __TIC__(B);
  std::cout << "SESSION_ID " << ENV_PARAM(SESSION_ID)++ << " "    //
            << "SESSION_NAME " << ENV_PARAM(SESSION_NAME) << " "  //
            << std::endl;
  auto sum = 0ll;
  for (int i = 0; i < 10000000; ++i) {
    sum = sum + ENV_PARAM(SESSION_ID);
  }

  std::cout << "sum = " << sum << " "  //
            << std::endl;

  __TOC__(B);
  return 0;
}
