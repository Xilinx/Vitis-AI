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

// This example requires an empty, local redis-server instance running with
// default settings.
#include <glog/logging.h>

#include <iostream>
#include <mutex>
#include <sstream>
#include <vitis/ai/lock.hpp>

using namespace std;
int main(int argc, char* argv[]) {
  auto device_name = string(argv[1]);
  {
    cout << "trying to lock..." << endl;
    auto mtx = vitis::ai::Lock::create(device_name);
    auto lock =
        std::unique_lock<vitis::ai::Lock>(*(mtx.get()), std::try_to_lock_t());
    if (!lock.owns_lock()) {
      cout << "waiting for other process to release the resource:"
           << device_name << endl;
      lock.lock();
    }
    cout << device_name << " is lock. presss any key to release the lock..."
         << endl;
    char c;
    cin >> c;
    cout << "lock is released" << endl;
  }
  return 0;
}
