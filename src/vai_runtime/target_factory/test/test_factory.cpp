/*
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "UniLog/UniLog.hpp"
#include "vitis/ai/target_factory.hpp"

#include <iomanip>
#include <iostream>
#include <string>
using namespace vitis::ai;
using namespace std;
int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);
  auto name = std::string{argv[1]};
  auto target = target_factory()->create(name);
  UNI_LOG_INFO << "Create target " << target.name();
  UNI_LOG_INFO << "  type         " << target.type();
  UNI_LOG_INFO << "  isa version  0x" << std::hex << std::setw(2)
               << setfill('0') << target.isa_version();
  UNI_LOG_INFO << "  feature code 0x" << std::hex << std::setw(12)
               << setfill('0') << target.feature_code();
  UNI_LOG_INFO << "  fingerprint  0x" << std::hex << std::setw(16)
               << setfill('0')
               << target_factory()->get_fingerprint(target.name());
  target_factory()->dump(target, "tmp.txt");
  std::cout << "Dump target " << name << " to tmp.txt" << std::endl;

  return 0;
}
