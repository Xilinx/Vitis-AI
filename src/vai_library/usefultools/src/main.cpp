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

#include <iomanip>
#include <iostream>
#include <memory>

#include "./dpregconfig.hpp"

using namespace std;
using namespace xilinx::dpregconfig;

int main(int argc, char *argv[]) {
  // string base_addr_str = argv[2];
  // uint32_t base_addr = (uint32_t)stoul(base_addr_str.substr(2), 0, 16);
  /*std::cout << "config file " << argv[1] << " "                           //
            << "base_addr " << argv[2] << "  to int " << base_addr << " " //
            << std::endl;*/
  auto config_file = (char *)"/usr/share/zu_veg_config";
  if (argc == 2) config_file = argv[1];
  auto regconfig = xilinx::dpregconfig::RegConfig::create(config_file);

  regconfig->run();
  std::vector<Config> configs = regconfig->getConfigs();

  for (Config c : configs) {
    // std::cout << c.offset << "\t" << c.bit_from << "\t" << c.bit_to << "\t"
    //         << c.name << "\t" << c.function << "\t";
    std::string value = std::to_string(c.value);
    for (auto d : c.dicts) {
      if ((unsigned)d.first == c.value) {
        value = d.second;
        break;
      }
    }

    std::cout << left << setw(20) << c.name << ":    " << value << " " << c.desc
              << std::endl;
    // std::cout << value <<  "\t" << c.function << std::endl;
    // std::cout << " " <<  std::endl;
  }

  // regconfig->run();

  return 0;
}
