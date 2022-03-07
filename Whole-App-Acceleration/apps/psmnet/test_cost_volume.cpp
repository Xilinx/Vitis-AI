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
#include <arm_neon.h>
#include <glog/logging.h>
#include <xrt.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/profiling.hpp>
DEF_ENV_PARAM(DUMP_PSMNET, "0");

DEF_ENV_PARAM_2(PSMNET_MODEL_DIR, "./PSMnet", std::string);
DEF_ENV_PARAM_2(PSMNET_MODEL_0, "PSMnet_0.xmodel", std::string);
DEF_ENV_PARAM_2(PSMNET_MODEL_1, "PSMnet_1.xmodel", std::string);
DEF_ENV_PARAM_2(PSMNET_MODEL_2, "PSMnet_2.xmodel", std::string);

using namespace std;

static uint64_t get_physical_address(const xclDeviceHandle& handle,
                                     const unsigned int bo) {
  xclBOProperties p;
  auto error_code = xclGetBOProperties(handle, bo, &p);
  CHECK_EQ(error_code, 0) << "cannot xclGetBOProperties !";
  auto phy = error_code == 0 ? p.paddr : -1;
  std::cout << "error_code " << error_code << " "        //
            << "handle " << handle << " "                //
            << "bo " << bo << " "                        //
            << "phy " << std::hex << "0x" << phy << " "  //
            << std::dec << std::endl;
  CHECK_NE(phy, (decltype(phy))(-1)) << "cannot xclGetBOProperties ! "  //
                                     << " error_code=" << error_code    //
                                     << " handle " << handle << " "
                                     << " bo=" << bo;
  return phy;
}

void cost_volume_1(int8_t* input_l_ptr, int8_t* input_r_ptr, int8_t* output);
int main(int argc, char* argv[]) {
  auto input_size = 144 * 240 * 32;
  auto output_size = input_size * 2 * 48;

  if (1) {
    auto input_l = std::vector<int8_t>(input_size);
    auto input_r = std::vector<int8_t>(input_size);
    for (auto i = 0; i < input_size; ++i) {
      input_l[i] = std::rand() % 100 + 1;
      input_r[i] = std::rand() % 100 + 1;
    }
    if (ENV_PARAM(DUMP_PSMNET)) {
      auto filename_l = std::string("input_l.bin");
      LOG(INFO) << "write to  " << filename_l;
      std::ofstream ofs(filename_l, ios::binary);
      ofs.write((char*)(&input_l[0]), input_size);
      ofs.close();
    }
    if (ENV_PARAM(DUMP_PSMNET)) {
      auto filename_r = std::string("input_r.bin");
      LOG(INFO) << "write to  " << filename_r;
      std::ofstream ofs(filename_r, ios::binary);
      ofs.write((char*)(&input_r[0]), input_size);
      ofs.close();
    }

    auto output = std::vector<int8_t>(output_size);
    for (auto i = 0; i < output_size; i++) {
      output[i] = 0;
    }
    cost_volume_1(&input_l[0], &input_r[0], &output[0]);
  }
  if (1) {
    auto device_id = 0;
    auto handle = xclOpen(device_id, NULL, XCL_INFO);
    //    auto flags = XCL_BO_FLAGS_CACHEABLE;
    auto flags = XCL_BO_FLAGS_CACHEABLE;
    auto bo_handle_l = xclAllocBO(handle, input_size, 0, flags);
    auto bo_addr_l = (int8_t*)xclMapBO(handle, bo_handle_l, true);
    auto phy_addr_l = get_physical_address(handle, bo_handle_l);

    auto bo_handle_r = xclAllocBO(handle, input_size, 0, flags);
    auto bo_addr_r = (int8_t*)xclMapBO(handle, bo_handle_r, true);
    auto phy_addr_r = get_physical_address(handle, bo_handle_r);

    auto bo_handle_o = xclAllocBO(handle, output_size, 0, flags);
    auto bo_addr_o = (int8_t*)xclMapBO(handle, bo_handle_o, true);
    auto phy_addr_o = get_physical_address(handle, bo_handle_o);

    std::cout  //
        << "bo_addr_l " << std::hex << "0x" << (void*)bo_addr_l << std::dec
        << " "
        << "phy_addr_l " << std::hex << "0x" << (void*)phy_addr_l << std::dec
        << " \n"  //
        << "bo_addr_r " << std::hex << "0x" << (void*)bo_addr_r << std::dec
        << " "
        << " phy_addr_r " << std::hex << "0x" << (void*)phy_addr_r << std::dec
        << " \n"  //
        << "bo_addr_o " << std::hex << "0x" << (void*)bo_addr_o << std::dec
        << " "  //
        << "phy_addr_o " << std::hex << "0x" << (void*)phy_addr_o << std::dec
        << std::endl;  //

    cost_volume_1((int8_t*)bo_addr_l, (int8_t*)bo_addr_r, (int8_t*)bo_addr_o);

    xclUnmapBO(handle, bo_handle_l, bo_addr_l);
    xclUnmapBO(handle, bo_handle_r, bo_addr_r);
    xclUnmapBO(handle, bo_handle_o, bo_addr_o);
    xclFreeBO(handle, bo_handle_l);
    xclFreeBO(handle, bo_handle_r);
    xclFreeBO(handle, bo_handle_o);
    xclClose(handle);
  }
  if (1) {
    auto tasks = std::vector<std::unique_ptr<vitis::ai::DpuTask>>();

    tasks.emplace_back(vitis::ai::DpuTask::create(
        ENV_PARAM(PSMNET_MODEL_DIR) + "/" + ENV_PARAM(PSMNET_MODEL_0)));
    tasks.emplace_back(vitis::ai::DpuTask::create(
        ENV_PARAM(PSMNET_MODEL_DIR) + "/" + ENV_PARAM(PSMNET_MODEL_1)));
    tasks.emplace_back(vitis::ai::DpuTask::create(
        ENV_PARAM(PSMNET_MODEL_DIR) + "/" + ENV_PARAM(PSMNET_MODEL_2)));
    auto output_l_ptr = (int8_t*)tasks[0]->getOutputTensor(1u)[0].get_data(0);
    auto output_r_ptr = (int8_t*)tasks[1]->getOutputTensor(1u)[0].get_data(0);
    auto input_k2_ptr = (int8_t*)tasks[2]->getInputTensor(0u)[0].get_data(0);

    std::cout  //
        << "output_l_ptr " << std::hex << "0x" << (void*)output_l_ptr
        << std::dec << " "  //
        << "output_r_ptr " << std::hex << "0x" << (void*)output_r_ptr
        << std::dec << " "  //
        << "input_k2_ptr " << std::hex << "0x" << (void*)input_k2_ptr
        << std::dec << " "  //
        << std::endl;
    ;
    cost_volume_1(output_l_ptr, output_r_ptr, input_k2_ptr);
  }
  return 0;
}

#ifdef ENABLE_NEON
void shift_and_copy_1(int8_t* input1, int8_t* input2, int8_t* output,
                      size_t size, int shift_num) {
  memcpy(output, input1, size);
  memcpy(output + size, input2, size);
}

#endif

void cost_volume_1(int8_t* input_l_ptr, int8_t* input_r_ptr,
                   int8_t* output_ptr) {
  __TIC__(COST_VOLUME_TOTAL)

  for (size_t a = 0; a < 1; ++a) {
    int act_fix = 0;
    auto height = 144u;
    auto width = 240u;
    auto channel = 32u;
    // max_disp of psmnet, its value is 192/4
    size_t disp = 48;

#ifdef ENABLE_NEON
    // compute the first channel of the cost volume
    __TIC__(COST_VOLUME_NEON_1)
    for (auto h = 0u; h < height; ++h) {
      for (auto w = 0u; w < width; ++w) {
        for (auto m = 0u; m < disp; ++m) {
          if (w < m) {
            int pos = ((h * width + w) * disp + m) * channel * 2;
            memset(output_ptr + pos, 0, channel * 2);
          } else {
            size_t i1_pos = (h * width + w) * channel;
            size_t i2_pos = (h * width + w - m) * channel;
            size_t o_pos = ((h * width + w) * disp + m) * channel * 2;
            shift_and_copy_1(input_l_ptr + i1_pos, input_r_ptr + i2_pos,
                             output_ptr + o_pos, channel, act_fix);
          }
        }
      }
    }
    __TOC__(COST_VOLUME_NEON_1)
#endif

    if (ENV_PARAM(DUMP_PSMNET)) {
      auto filename = std::string("cost_volume.bin");
      LOG(INFO) << "write to  " << filename;
      std::ofstream ofs(filename, ios::binary);
      ofs.write((char*)output_ptr, 144 * 240 * 64 * 48);
      ofs.close();
    }
  }
  __TOC__(COST_VOLUME_TOTAL)
}  // namespace ai
