/*
 * Copyright 2019 Xilinx Inc.
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
#include <glog/logging.h>

#include <iomanip>
#include <iostream>
#include <xir/xrt_device_handle.hpp>

#include "parse_value.hpp"
#include "xrt_xcl_read.hpp"
using namespace std;
void set_reg(xclDeviceHandle xcl_handle, uint32_t ip_index,
             uint64_t cu_base_addr, uint32_t offset, uint32_t value) {
  auto read_result =
      xrtXclWrite(xcl_handle, ip_index, offset, cu_base_addr, value);

  CHECK_EQ(read_result, 0) << "xrtXclWrite has error!";
}

uint32_t get_reg(xclDeviceHandle xcl_handle, uint32_t ip_index,
                 uint64_t cu_base_addr, uint32_t offset) {
  uint32_t value = 0;
  auto read_result =
      xrtXclRead(xcl_handle, ip_index, offset, cu_base_addr, &value);

  CHECK_EQ(read_result, 0) << "xrtXclRead has error!";
  return value;
}

void xdpu_set_reg(xclDeviceHandle xcl_handle, uint32_t ip_index,
                  uint64_t cu_base_addr, string set_reg_conf) {
  struct reg {
    uint32_t addr;
    uint32_t value;
    string name;
  };
  vector<reg> regs;
  ifstream stream(set_reg_conf);
  string name;
  string offset_add;
  string offset_val;
  while ((stream >> name >> offset_add >> offset_val).good()) {
    uint64_t offset_add2;
    uint64_t offset_val2;
    // LOG(INFO) << "name=" << name << " offset_add=" << offset_add << "
    // offset_val=" << offset_val;
    parse_value(offset_add, offset_add2);
    parse_value(offset_val, offset_val2);
    regs.emplace_back(reg{(uint32_t)offset_add2, (uint32_t)offset_val2, name});
  }
  stream.close();
  for (const auto& reg : regs) {
    auto befor_val = get_reg(xcl_handle, ip_index, cu_base_addr, reg.addr);
    set_reg(xcl_handle, ip_index, cu_base_addr, reg.addr, reg.value);
    auto after_val = get_reg(xcl_handle, ip_index, cu_base_addr, reg.addr);
    LOG_IF(INFO, true) << "addr 0x" << hex << reg.addr << "\t" << setfill(' ')
                       << "before 0x" << hex << setw(6) << left << befor_val
                       << " " << dec << setw(6) << right << befor_val << "\t"
                       << setfill(' ') << "set 0x" << hex << setw(6) << left
                       << reg.value << " " << dec << setw(6) << right
                       << reg.value << "\t" << setfill(' ') << "after 0x" << hex
                       << setw(6) << left << after_val << " " << dec << setw(6)
                       << right << after_val << "\t" << reg.name << endl;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    cout << "usage: " << argv[0] << " <reg_conf> <cu_name> <index> " << endl;
    cout << "eg: " << argv[0] << " /usr/share/vart/reg.conf dpu 0" << endl;
    return 1;
  }
  string set_reg_conf = argv[1];
  auto cu_name = std::string(argv[2]);
  auto index = std::stoi(std::string(argv[3]));
  auto h = xir::XrtDeviceHandle::get_instance();
  LOG(INFO) << "h->get_handle() " << h->get_handle(cu_name, index) << " "  //
      ;
  xdpu_set_reg(h->get_handle(cu_name, index), h->get_cu_index(cu_name, index),
               h->get_cu_addr(cu_name, index), set_reg_conf);
  return 0;
}
