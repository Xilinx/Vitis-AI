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
#include "./dpregconfig_imp.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
//#include <chrono>
using namespace std;
constexpr auto MAP_SIZE = 0x1000;

namespace xilinx {
namespace dpregconfig {

void RegConfigImp::loadConfigs() {
  std::ifstream in(config_file_);
  if (!in) {
    std::cout << "Error opening " << config_file_ << " for input" << std::endl;
    exit(-1);
  }
  std::string line;
  const char split = '\t';
  // frist line read base_addr_
  // std::vector<Config> configs;
  int line_num = 0;
  while (getline(in, line)) {
    line_num++;
    std::string token;
    std::istringstream tokenStream(line);
    int j = 0;
    Config config;
    while (std::getline(tokenStream, token, split)) {
      j++;
      switch (j) {
        case 1: {
          // std::cout << token << std::endl;
          if (token[0] == '0' && token[1] == 'x') {
            config.offset = (uint32_t)stoul(token.substr(2), 0, 16);
          }
          break;
        }
        case 2: {
          if (line_num == 1 && token[0] == '0' && token[1] == 'x') {
            base_addr_ = (uint32_t)stoul(token.substr(2), 0, 16);
            break;
          }
          if (token[0] == '[') {
            std::size_t pos = token.find(":");
            if (pos == string::npos) {
              config.bit_from = std::atoi(token.substr(1, token.size()).data());
              config.bit_to = config.bit_from;
            } else {
              config.bit_from = std::atoi(token.substr(1, pos).data());
              config.bit_to =
                  std::atoi(token.substr((pos + 1), token.size()).data());
            }
          }
          // std::cout << "from " << config.bit_from << " to "
          //           << config.bit_to << " " << std::endl;
          break;
        }
        case 3: {
          // std::cout << token << std::endl;
          config.name = token;
          break;
        }
        case 4: {
          config.privilege = token;
          break;
        }
        case 5: {
          break;
        }
        case 6: {
          config.function = token;
          break;
        }
        case 7: {
          // std::cout << token << std::endl;
          std::vector<std::pair<int, std::string> > dicts;
          std::string k;
          std::istringstream stream(token);
          while (std::getline(stream, k, ';')) {
            std::size_t pos1 = k.find(":");
            if (pos1 != string::npos) {
              std::pair<int, std::string> dict;
              dict.first = std::atoi(k.substr(0, pos1).data());
              dict.second = k.substr((pos1 + 1), k.size()).data();
              /*    std::cout << "dict.first "  << dict.first << " " //
                        << "dict.second "  << dict.second << " " //
                        << std::endl; */

              dicts.push_back(dict);
            }
          }
          config.dicts = dicts;
          break;
        }
        case 8: {
          config.desc = token;
          break;
        }
      }
    }

    if ((config.privilege.find("r") != string::npos ||
         config.privilege.find("R") != string::npos) &&
        config.name != "Name" && config.name != "Reserved") {
      configs_.push_back(config);
    }
    // configs.push_back(&config);
    // }
  }

  // return configs;
}

RegConfigImp::RegConfigImp(const char *config_file)
    : config_file_(config_file) {
  loadConfigs();
}

RegConfigImp::~RegConfigImp() {}

void RegConfigImp::run() {
  /* for (Config c : configs_) {
   std::cout << c.offset << " " << c.bit_from << " " << c.bit_to << " "
             << c.name << " " << std::endl;
             }*/
  //    loadConfigs();
  auto size = (size_t)(configs_[configs_.size() - 1].offset + 4);
  auto dev_fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (dev_fd < 0) {
    printf("open(/dev/mem) failed.");
    return;
  }
  //    std::cout << "base_addr_ " << base_addr_ << " " //
  //          << "size " << size  << " "           //
  //          << std::endl;
  unsigned char *map_base = (unsigned char *)mmap(
      NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd, base_addr_);

  if (map_base == MAP_FAILED) {
    printf("mmap failed\n");
    return;
  }

  /*for(int i=0;i<(int)(size / 4) ;i++){
      auto base1 = (volatile uint32_t *)(map_base + i * 4);
      std::cout << "base1 "  << base1 << " " << std::endl; //
      }*/
  for (Config &c : configs_) {
    auto base1 = (volatile uint32_t *)(map_base + c.offset);
    auto value = *base1;
    value = value << (31 - c.bit_from);
    value = value >> (31 - (c.bit_from - c.bit_to));
    /*std::cout << value << "\t " //
              << c.name << "\t\t"   //
              << c.function << "\t"
              << std::endl; */
    c.value = value;
  }

  // auto base1 = (volatile uint32_t *)(map_base + offset);

  munmap(map_base, MAP_SIZE);
  if (dev_fd) {
    close(dev_fd);
  }
}

std::vector<Config> RegConfigImp::getConfigs() { return configs_; }

}  // namespace dpregconfig
}  // namespace xilinx
