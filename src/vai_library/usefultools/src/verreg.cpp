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
#include "verreg.hpp"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
#include <sstream>

using namespace std;

static const char *yes_or_no(int a) { return a ? "yes" : "no"; }
static string hp_width(int n) {
  string ret = "INVALID";
  if (n > 0) {
    ret = to_string(32 * (1 << (n - 1)));
  }
  return ret;
}

static string data_width(int n) {
  string ret = "INVALID";
  switch (n) {
    case 1:
      ret = "8bit";
      break;
    case 2:
      ret = "16bit";
      break;
  }
  return ret;
}
static string bank_group(int n) {
  string ret = "INVALID";
  if (n > 0) ret = to_string(n);
  return ret;
}
static string arch_type(int n) {
  string ret = "INVALID";
  switch (n) {
    case 1:
      ret = "B1024F(2*8*8^2)";
      break;
    case 2:
      ret = "B1152F(2*4*12^2)";
      break;
    case 3:
      ret = "B4096F(2*8*16^2)";
      break;
    case 4:
      ret = "B256F(2*2*8^2)";
      break;
    case 5:
      ret = "B512F(2*4*8^2)";
      break;
    case 6:
      ret = "B800F(2*4*10^2)";
      break;
    case 7:
      ret = "B1600F(2*8*10^2)";
      break;
    case 8:
      ret = "B2048F(2*4*16^2)";
      break;
    case 9:
      ret = "B2304F(2*8*12^2)";
      break;
    case 10:
      ret = "B8192F(2*16*16^2)";
      break;
    case 11:
      ret = "B3136F(2*8*14^2)";
      break;
    case 12:
      ret = "B288F(2*4*6^2)";
      break;
  }
  return ret;
}

static string dpu_target_version(int n) {
  auto ret = "INVALID";
  switch (n) {
    case 1:
      ret = "1.1.3";
      break;
    case 2:
      ret = "1.3.0";
      break;
    case 3:
      ret = "1.3.1";
      break;
    case 4:
      ret = "1.3.2";
      break;
    case 5:
      ret = "1.3.3";
      break;
    case 6:
      ret = "1.3.4";
      break;
    case 7:
      ret = "1.3.5";
      break;
    case 8:
      ret = "1.4.0";
      break;
    case 9:
      ret = "1.4.1";
      break;
    case 10:
      ret = "1.4.2";
      break;
    case 11:
      ret = "1.3.6";
      break;
    case 12:
      ret = "1.3.7";
      break;
  }
  return ret;
}

static string dpu_hp_interact(int n) {
  auto ret = string{"INVALID"};
  if (n > 0) {
    ret = to_string(n);
  }
  return ret;
}

static int interrupt_number(struct VerReg *reg, unsigned int n) {
  auto base =
      ((n & 0x8) == 0) ? reg->x8.interrupt_base0 : reg->x8.interrupt_base1;
  return base + (n & 0x7);
}

static string average_pool(int n) {
  stringstream ret;
  unsigned mask = 1;
  int x = 2;
  int first = true;
  for (int i = 0; i < 8; ++i) {
    if (mask & n) {
      if (!first) {
        ret << ",";
      }
      ret << x << "x" << x;
      first = false;
    }
    x++;
    mask = mask << 1;
  }
  return ret.str();
}

static string depthwise_conv(int n) {
  string ret = "N/A";
  if (n > 0) {
    ret = to_string(n);
  }
  return ret;
}

static string relu_addon(int n) {
  string ret = "N/A";
  switch (n) {
    case 1:
      ret = "Leakyrelu";
      break;
    case 2:
      ret = "Relu6";
      break;
    case 3:
      ret = "LeakyRelu+Relu6";
      break;
  }
  return ret;
}

static string pre_relu(int n) {
  string ret = "N/A";
  if (n > 0) {
    ret = "PreRelu";
  }
  return ret;
}

static string pool_parallel(int n) { return to_string(n); }

static string chip_part(int n) {
  string ret = "N/A";
  switch (n) {
    case 1:
      ret = "Z7020-2";
      break;
    case 2:
      ret = "ZU2EG-2";
      break;
    case 3:
      ret = "ZU9EG-2";
      break;
    case 4:
      ret = "ZU2CG-L1";
      break;
    case 5:
      ret = "ZU5EV-2";
      break;
    case 6:
      ret = "ZU7EV-2";
      break;
    case 7:
      ret = "ZU3EG-1";
      break;
  }
  return ret;
}
int main(int argc, char *argv[]) {
  auto capacity = 0x1000u;
#if defined __aarch64__
  auto phy = 0x8ff00000;
#elif defined __arm__
  auto phy = 0x4ff00000;
#elif defined __x86_64__
  auto phy = 0x4ff00000;
#else
#error "Platform not support!"
#endif
  auto pagesize = getpagesize();
#ifdef __QNX__
  auto data = (void *)mmap_device_io(capacity, phy);
#else
  auto fd = open("/dev/mem", O_RDWR | O_SYNC);
  CHECK_GT(fd, 0);
  auto data = mmap(NULL, capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, phy);
#endif
  cout << "PAGE SIZE: " << pagesize << endl;
  CHECK(data != MAP_FAILED);
  auto reg = (struct VerReg *)data;
  cout             // 0x0
      << std::hex  //
      << "0x0: "
      << "0x" << reg->x0.u32 << "\n"       //
      << "magic: "                         //
      << "0x" << reg->x0.m << "\n"         //
      << "n_of_reg: "                      //
      << "0x" << reg->x0.n_of_reg << "\n"  //
      << "ver_fmt: "
      << "0x" << reg->x0.ver_fmt  //
      << "\n"                     //
      << endl;
  cout                                     // 0x4
      << "0x4: 0x" << reg->x4.u32 << "\n"  //
      << "bit_ver: "
      << "0x" << reg->x4.bit_ver << "\n"  //
      << "date: " << std::dec             //
      << " 20" << reg->x4.year            //
      << "/" << reg->x4.month             //
      << "/" << reg->x4.day               //
      << " " << reg->x4.hour              //
      << ":00"
      << "\n"                                            //
      << "Frequency: " << std::dec << reg->x4.frequency  //
      << " MHz"
      << "\n"                                                        //
      << "Encrption: " << std::dec << yes_or_no(reg->x4.encryption)  //
      << "\n"                                                        //
      << std::hex << endl;
  cout                                     // 0x8
      << "0x8: 0x" << reg->x8.u32 << "\n"  //
      << "reserved: "
      << "0x" << reg->x8.reserved << "\n"                         //
      << std::dec                                                 //
      << "interrupt base 0: " << reg->x8.interrupt_base0 << "\n"  //
      << "interrupt base 1: " << reg->x8.interrupt_base1 << "\n"  //
      << std::hex << endl;

  cout                                                             // 0xc
      << "0xc: 0x" << reg->xc.u32 << "\n"                          //
      << std::dec                                                  //
      << "HP_width: " << hp_width(reg->xc.hp_width) << "\n"        //
      << "Data width: " << data_width(reg->xc.data_width) << "\n"  //
      << "Bank group: " << bank_group(reg->xc.bank_group) << "\n"  //
      << "Arch type: " << arch_type(reg->xc.arch_type) << "\n"
      << "DPU target version: "
      << dpu_target_version(reg->xc.dpu_target_version) << "\n"  //
      << "DPU HP interact: " << dpu_hp_interact(reg->xc.dpu_hp_interact)
      << "\n"  //
      << "DPU core number: " << reg->xc.dpu_core_number << "\n"
      << std::hex << endl;

  cout                                       // 0xc
      << "0x10: 0x" << reg->x10.u32 << "\n"  //
      << std::dec << "DPU0 interrupt number : "
      << interrupt_number(reg, reg->x10.dpu0_interrupt_number) << "\n"
      << "DPU1 interrupt number : "
      << interrupt_number(reg, reg->x10.dpu1_interrupt_number) << "\n"
      << "DPU2 interrupt number : "
      << interrupt_number(reg, reg->x10.dpu2_interrupt_number) << "\n"
      << "DPU3 interrupt number : "
      << interrupt_number(reg, reg->x10.dpu3_interrupt_number) << "\n"
      << "DPU4 interrupt number : "
      << interrupt_number(reg, reg->x10.dpu4_interrupt_number) << "\n"
      << "DPU5 interrupt number : "
      << interrupt_number(reg, reg->x10.dpu5_interrupt_number) << "\n"
      << "DPU6 interrupt number : "
      << interrupt_number(reg, reg->x10.dpu6_interrupt_number) << "\n"
      << "DPU7 interrupt number : "
      << interrupt_number(reg, reg->x10.dpu7_interrupt_number) << "\n"
      << std::hex << endl;

  cout                                       // 0xc
      << "0x14: 0x" << reg->x14.u32 << "\n"  //
      << std::dec << "DPU8 interrupt number : "
      << interrupt_number(reg, reg->x14.dpu8_interrupt_number) << "\n"
      << "DPU9 interrupt number : "
      << interrupt_number(reg, reg->x14.dpu9_interrupt_number) << "\n"
      << "DPU10 interrupt number : "
      << interrupt_number(reg, reg->x14.dpu10_interrupt_number) << "\n"
      << "DPU11 interrupt number : "
      << interrupt_number(reg, reg->x14.dpu11_interrupt_number) << "\n"
      << "DPU12 interrupt number : "
      << interrupt_number(reg, reg->x14.dpu12_interrupt_number) << "\n"
      << "DPU13 interrupt number : "
      << interrupt_number(reg, reg->x14.dpu13_interrupt_number) << "\n"
      << "DPU14 interrupt number : "
      << interrupt_number(reg, reg->x14.dpu14_interrupt_number) << "\n"
      << "DPU15 interrupt number : "
      << interrupt_number(reg, reg->x14.dpu15_interrupt_number) << "\n"
      << std::hex << endl;

  cout << "0x18: 0x" << reg->x18.u32 << "\n"                               //
       << std::dec                                                         //
       << "average pool: " << average_pool(reg->x18.average_pool) << "\n"  //
       << "depthwise conv: " << depthwise_conv(reg->x18.depthwise_conv)
       << "\n"                                                       //
       << "relu addon: " << relu_addon(reg->x18.relu_addon) << "\n"  //
       << "PreRelu: " << pre_relu(reg->x18.pre_relu) << "\n"         //
       << "\n"                                                       //
       << std::hex << endl;

  cout << "0x1c: 0x" << reg->x1c.u32 << "\n"                                  //
       << "reserved: 0x" << reg->x1c.reserved << "\n"                         //
       << std::dec                                                            //
       << "pool_parallel: " << pool_parallel(reg->x1c.pool_parallel) << "\n"  //
       << "load_augm: " << yes_or_no(reg->x1c.load_augm) << "\n"              //
       << "load_img_mean: " << yes_or_no(reg->x1c.load_img_mean) << "\n"      //
       << "nl_ratio_index: " << reg->x1c.nl_ratio_index << "\n"               //
       << std::hex << endl;

  cout << "0x20: 0x" << reg->x20.u32 << "\n"  //
       << std::dec                            //
       << "f0: " << yes_or_no(reg->x20.f0) << "\n"
       << "f1: " << yes_or_no(reg->x20.f1) << "\n"
       << "f2: " << yes_or_no(reg->x20.f2) << "\n"
       << "f3: " << yes_or_no(reg->x20.f3) << "\n"
       << "f4: " << yes_or_no(reg->x20.f4) << "\n"
       << "f5: " << yes_or_no(reg->x20.f5) << "\n"
       << "f6: " << yes_or_no(reg->x20.f6) << "\n"
       << "f7: " << yes_or_no(reg->x20.f7) << "\n"
       << std::hex << endl;

  cout << "0x24: 0x" << reg->x24.u32 << "\n"                                  //
       << std::dec                                                            //
       << "softmax: " << yes_or_no(reg->x24.softmax_valid) << "\n"            //
       << "softmax_version: " << yes_or_no(reg->x24.softmax_version) << "\n"  //
       << "softmax_interrupt_number: "
       << interrupt_number(reg, reg->x24.softmax_interrupt_number) << "\n"  //
       << "fc: " << yes_or_no(reg->x24.fc_valid) << "\n"                    //
       << "fc_version: " << yes_or_no(reg->x24.fc_version) << "\n"          //
       << "fc_interrupt_number: "
       << interrupt_number(reg, reg->x24.fc_interrupt_number) << "\n"       //
       << "bt1120: " << yes_or_no(reg->x24.bt1120_valid) << "\n"            //
       << "bt1120_version: " << yes_or_no(reg->x24.bt1120_version) << "\n"  //
       << "bt1120_interrupt_number: "
       << interrupt_number(reg, reg->x24.bt1120_interrupt_number) << "\n"  //
       << "hdmi: " << yes_or_no(reg->x24.hdmi_valid) << "\n"               //
       << "hdmi_version: " << yes_or_no(reg->x24.hdmi_version) << "\n"     //
       << "hdmi_interrupt_number: "
       << interrupt_number(reg, reg->x24.hdmi_interrupt_number) << "\n"  //
       << std::hex << endl;

  cout << "0x28: 0x" << reg->x28.u32 << "\n"                                  //
       << std::dec                                                            //
       << "bgr2yuv: " << yes_or_no(reg->x28.bgr2yuv_valid) << "\n"            //
       << "bgr2yuv_version: " << yes_or_no(reg->x28.bgr2yuv_version) << "\n"  //
       << "bgr2yuv_interrupt_number: "
       << interrupt_number(reg, reg->x28.bgr2yuv_interrupt_number) << "\n"    //
       << "yuv2rgb: " << yes_or_no(reg->x28.yuv2rgb_valid) << "\n"            //
       << "yuv2rgb_version: " << yes_or_no(reg->x28.yuv2rgb_version) << "\n"  //
       << "yuv2rgb_interrupt_number: "
       << interrupt_number(reg, reg->x28.yuv2rgb_interrupt_number) << "\n"    //
       << "sigmoid: " << yes_or_no(reg->x28.sigmoid_valid) << "\n"            //
       << "sigmoid_version: " << yes_or_no(reg->x28.sigmoid_version) << "\n"  //
       << "sigmoid_interrupt_number: "
       << interrupt_number(reg, reg->x28.sigmoid_interrupt_number) << "\n"  //
       << "resize: " << yes_or_no(reg->x28.resize_valid) << "\n"            //
       << "resize_version: " << yes_or_no(reg->x28.resize_version) << "\n"  //
       << "resize_interrupt_number: "
       << interrupt_number(reg, reg->x28.resize_interrupt_number) << "\n"  //
       << std::hex << endl;

  cout << "0x2c: 0x" << reg->x2c.u32 << "\n"                         //
       << "reserved: 0x" << reg->x2c.reserved << "\n"                //
       << "auth_en: " << yes_or_no(reg->x2c.auth_en) << "\n"         //
       << "chip part: " << chip_part(reg->x2c.chip_part) << "\n"     //
       << std::dec                                                   //
       << "board_hw_version: " << reg->x2c.board_hw_version << "\n"  //
       << "board_type: " << reg->x2c.board_type << "\n"              //
       << "board_number: " << reg->x2c.board_number << "\n"          //
       << std::hex << endl;
  return 0;
}
