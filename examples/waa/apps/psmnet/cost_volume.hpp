/*
 * Copyright 2021 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <glog/logging.h>

#include <iostream>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#define IN_DATA_BYTES  1105920  // 144 * 240 * 32
#define OUT_DATA_BYTES 106168320  // 144 * 240 * 64 * 48

#define KERNEL_NAME "cost_volume_top"

class CostVolumeAccel
{
    xrt::run runner;
    xrt::device device;
    xrt::kernel krnl;

    xrt::bo left_input;
    xrt::bo right_input;
    xrt::bo output;

    void *left_input_m;
    void *right_input_m;
    void *output_m;

public:

    CostVolumeAccel(std::string xclbin, unsigned device_index);
    void run(int8_t *left_input_data, int8_t *right_input_data, int8_t *output_data);
};
