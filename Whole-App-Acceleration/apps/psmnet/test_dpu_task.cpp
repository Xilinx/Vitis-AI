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
#include <iostream>
using namespace std;

#include <vitis/ai/dpu_task.hpp>
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

int main(int argc, char* argv[]) {
  auto kernel_name_0 =
      "/home/root/max/psmnet/"
      "PSMNet_2_int_50%_compiled_DPUCVDX8G_ISA1_C32B3_3D.xmodel";
  auto task = vitis::ai::DpuTask::create(kernel_name_0);
  auto input_tensor_left = task->getInputTensor(0u)[0];
  auto output_tensor_left = task->getOutputTensor(0u)[0];
  cout << "input_tensor_left " << input_tensor_left << endl;
  cout << "output_tensor_left " << output_tensor_left << endl;
  return 0;
}
