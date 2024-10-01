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

#include "payload.hpp"

using namespace vitis::ai::trace;
using namespace std;

int main(void) {
  auto cu_device_id = 1;
  auto cu_core_id = 0;
  auto cu_addr = 0x1234567890;
  auto cu_fingerprint = 0x158161909596;

  auto x = tracePayload<int>("workload", 1234567);

  // Method 2
  auto z2 = trace("dev_id", cu_device_id, "core_id", cu_core_id);
  auto z4 = trace("dev_id", cu_device_id, "core_id", cu_core_id, "cu_addr",
                  cu_addr, "cu_fig", cu_fingerprint);

  auto e = trace(pair("dev_id", cu_device_id), pair("core_id", cu_core_id),
                 pair("cu_addr", cu_addr), pair("cu_fig", cu_fingerprint));

  cout << "payload size: " << sizeof(x) << endl;
  cout << "auto z2 size: " << sizeof(z2) << endl;
  cout << "auto z4 size: " << sizeof(z4) << endl;

  cout << "----------------------" << endl;
  // Method 3

  auto iii = make_pl_inf(99.9, 8, 'c');
  auto iiii = make_pl_inf(99.9, 8, 'c', 33);
  iiii.dump();
  // cout << "i3 sizeof: " << sizeof(iii) << ", i4 sizeof: " << sizeof(iiii) <<
  // endl; cout << "i3 size: " << iii.size() << ", i4 size: " << iiii.size() <<
  // endl;
  return 0;
}
