/*
 * Copyright 2019 Xilinx Inc.
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

#ifndef _14PT_HPP_
#define _14PT_HPP_

#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "dnndk/dnndk.h"

using namespace std;
using namespace cv;

// JointPoint
#define PT_KRENEL_CONV "pose_0"
#define PT_KRENEL_FC "pose_2"
#define PT_KERNEL_FC2 "pose_3"
#define PT_CONV_INPUT_NODE "conv1_7x7_s2"
#define PT_CONV_CONCAT_NODE "inception_5b_1x1"
#define PT_CONV_OUTPUT_NODE "inception_5b_output"
#define PT_FC_NODE "fc_coordinate"
#define PT_FC_NODE_OPT "fc_visible"

class GestureDetect {
   public:
    void Init();
    void Finalize();
    void Run(cv::Mat&);
    GestureDetect();
    ~GestureDetect();

   private:
    DPUKernel* kernel_conv_PT;
    DPUTask* task_conv_PT;
    DPUKernel* kernel_fc_PT;
    DPUTask* task_fc_PT;
};

#endif
