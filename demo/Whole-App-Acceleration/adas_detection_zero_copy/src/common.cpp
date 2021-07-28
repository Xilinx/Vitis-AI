
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

#include <cassert>
#include <numeric>
int getTensorShape(vart::Runner* runner, GraphInfo* shapes, int cntin,
                   int cntout) {
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  if (shapes->output_mapping.empty()) {
    shapes->output_mapping.resize((unsigned)cntout);
    std::iota(shapes->output_mapping.begin(), shapes->output_mapping.end(), 0);
  }
  for (int i = 0; i < cntin; i++) {
    auto dim_num = inputTensors[i]->get_shape().size();
    if (dim_num == 4) {
      shapes->inTensorList[i].channel = inputTensors[i]->get_shape().at(3);
      shapes->inTensorList[i].width = inputTensors[i]->get_shape().at(2);
      shapes->inTensorList[i].height = inputTensors[i]->get_shape().at(1);
      shapes->inTensorList[i].size =
          inputTensors[i]->get_element_num() / inputTensors[0]->get_shape().at(0);
    } else if (dim_num == 2) {
      shapes->inTensorList[i].channel = inputTensors[i]->get_shape().at(1);
      shapes->inTensorList[i].width = 1;
      shapes->inTensorList[i].height = 1;
      shapes->inTensorList[i].size =
          inputTensors[i]->get_element_num() / inputTensors[0]->get_shape().at(0);
    }
  }
  for (int i = 0; i < cntout; i++) {
    auto dim_num = outputTensors[shapes->output_mapping[i]]->get_shape().size();
    if (dim_num == 4) {
      shapes->outTensorList[i].channel =
          outputTensors[shapes->output_mapping[i]]->get_shape().at(3);
      shapes->outTensorList[i].width =
          outputTensors[shapes->output_mapping[i]]->get_shape().at(2);
      shapes->outTensorList[i].height =
          outputTensors[shapes->output_mapping[i]]->get_shape().at(1);
      shapes->outTensorList[i].size =
          outputTensors[shapes->output_mapping[i]]->get_element_num() /
          outputTensors[shapes->output_mapping[0]]->get_shape().at(0);
    } else if (dim_num == 2) {
      shapes->outTensorList[i].channel =
          outputTensors[shapes->output_mapping[i]]->get_shape().at(1);
      shapes->outTensorList[i].width = 1;
      shapes->outTensorList[i].height = 1;
      shapes->outTensorList[i].size =
          outputTensors[shapes->output_mapping[i]]->get_element_num() /
          outputTensors[shapes->output_mapping[0]]->get_shape().at(0);
    }
  }
  return 0;
}

static int find_tensor(std::vector<const xir::Tensor*> tensors,
                       const std::string& name) {
  int ret = -1;
  for (auto i = 0u; i < tensors.size(); ++i) {
    if (tensors[i]->get_name().find(name) != std::string::npos) {
      ret = (int)i;
      break;
    }
  }
  assert(ret != -1);
  return ret;
}
int getTensorShape(vart::Runner* runner, GraphInfo* shapes, int cntin,
                   std::vector<std::string> output_names) {
  for (auto i = 0u; i < output_names.size(); ++i) {
    auto idx = find_tensor(runner->get_output_tensors(), output_names[i]);
    shapes->output_mapping.push_back(idx);
  }
  getTensorShape(runner, shapes, cntin, (int)output_names.size());
  return 0;
}
