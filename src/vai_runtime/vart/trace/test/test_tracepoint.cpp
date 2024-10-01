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

#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vitis/ai/tracepoint.hpp>

auto help = std::string(
    "			 	\r\n\
Must set enviroment before test_tracepoint run: 	\r\n\
# export VAI_TRACE_ENABLE=1	    		 	\r\n\
For x86 linux kernel-3.10:			 	\r\n\
# export VAI_TRACE_TS=\"x86-tsc\"		 	\r\n\
For newer x86 kernel and arm			 	\r\n\
# export VAI_TRACE_TS=\"boot\"			 	\r\n\
");

int test_tracepoint(void) {
  vitis::ai::tracepoint(VAI_EVENT_HOST_START, "TEST");

  for (int i = 0; i < 1000; i++) {
    vitis::ai::tracepoint(VAI_EVENT_INFO, "TEST", "HELLO_" + std::to_string(i));
  }

  vitis::ai::tracepoint(VAI_EVENT_HOST_END, "TEST");

  return 0;
}

int main(int argc, char* argv[]) {
  if (argc > 1) {
    std::cout << help;
    return 0;
  }
  /* Test two threads */
  fork();
  test_tracepoint();

  return 0;
}
