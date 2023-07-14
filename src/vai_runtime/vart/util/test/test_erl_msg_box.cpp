
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
#include <vitis/ai/erl_msg_box.hpp>
using namespace std;

int main1(int argc, char* argv[]) {
  auto m = make_unique<vitis::ai::ErlMsgBox<int>>(3);
  for (auto i = 0; i < 4; i++) {
    auto x =
        m->send_ptr(std::make_unique<int>(i), std::chrono::milliseconds(1000));
    cout << "x  = "
         << (x == nullptr ? std::string("null") : std::to_string(*x.get()))
         << endl;
  }
  auto cur = std::make_unique<int>(0);
  while ((cur = m->recv(std::chrono::milliseconds(1000))) != nullptr) {
    cout << "cur  = " << std::to_string(*cur.get()) << endl;
  }
  return 0;
}
int main(int argc, char* argv[]) {
  cout << "start test" << endl;
  return main1(argc, argv);
  return 0;
}
