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

#include "event.hpp"
#include "ringbuf.hpp"

using namespace vitis::ai;
using namespace std;

int main(void) {
  RingBuf buf(8);

  for (auto i = 0; i < 1000000; i++) {
    // auto e1 = new traceEvent<uint64_t>(VAI_EVENT_PY_FUNC_END, "XXX", i,
    // string("hels"), 123123123);
    auto e1 = new traceEvent<uint64_t>(VAI_EVENT_PY_FUNC_END, "XXX", i,
                                       "run_time", 123123123);
    buf.push(e1);

    if (i % 100000 == 0) {
      std::cout << "input test " << i << " times" << std::endl;
    }
  }

  buf.dump();

  return 0;
}
