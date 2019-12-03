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

#include "common.h"
int getTensorShape(DpuRunner* runner, GraphInfo *shapes, int cntin, int cntout) {
    auto outputTensors = runner->get_output_tensors();
    auto inputTensors = runner->get_input_tensors();

    for (int i=0;i<cntin; i++) {
        auto in_dims = inputTensors[i]->get_dims();
        if (runner->get_tensor_format() == ai::DpuRunner::TensorFormat::NCHW) {
            shapes->inTensorList[i].channel = inputTensors[i]->get_dim_size(1);
            shapes->inTensorList[i].width   = inputTensors[i]->get_dim_size(3);
            shapes->inTensorList[i].height  = inputTensors[i]->get_dim_size(2);
            shapes->inTensorList[i].size  = inputTensors[i]->get_element_num()/inputTensors[0]->get_dim_size(0);
        } else if (runner->get_tensor_format() == ai::DpuRunner::TensorFormat::NHWC) {
            shapes->inTensorList[i].channel = inputTensors[i]->get_dim_size(3);
            shapes->inTensorList[i].width   = inputTensors[i]->get_dim_size(2);
            shapes->inTensorList[i].height  = inputTensors[i]->get_dim_size(1);
            shapes->inTensorList[i].size  = inputTensors[i]->get_element_num()/inputTensors[0]->get_dim_size(0);
        }
        else {

            return -1;
        }
    }
    for (int i=0;i<cntout; i++) {
        auto in_dims = outputTensors[i]->get_dims();
        if (runner->get_tensor_format() == ai::DpuRunner::TensorFormat::NCHW) {
            shapes->outTensorList[i].channel = outputTensors[i]->get_dim_size(1);
            shapes->outTensorList[i].width   = outputTensors[i]->get_dim_size(3);
            shapes->outTensorList[i].height  = outputTensors[i]->get_dim_size(2);
            shapes->outTensorList[i].size  = outputTensors[i]->get_element_num()/outputTensors[0]->get_dim_size(0);
        } else if (runner->get_tensor_format() == ai::DpuRunner::TensorFormat::NHWC) {
            shapes->outTensorList[i].channel = outputTensors[i]->get_dim_size(3);
            shapes->outTensorList[i].width   = outputTensors[i]->get_dim_size(2);
            shapes->outTensorList[i].height  = outputTensors[i]->get_dim_size(1);
            shapes->outTensorList[i].size  = outputTensors[i]->get_element_num()/outputTensors[0]->get_dim_size(0);
        }
        else {

            return -1;
        }
    }
    return 0;
}



