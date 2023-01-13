/*Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_DECENT_Q_UTILS_CHECK_GRAPH_H_
#define TENSORFLOW_CONTRIB_DECENT_Q_UTILS_CHECK_GRAPH_H_

#include "transform_utils.h"

namespace tensorflow {
namespace decent_q {

void PrintNodeInfo(const NodeDef* node);

Status PrintStructure(const GraphDef& graph);

std::vector<const NodeDef*> FindInputs(const GraphDef& graph);

std::vector<const NodeDef*> FindVariables(const GraphDef& graph);

std::vector<const NodeDef*> FindOutputs(const GraphDef& graph);

std::map<string, int> GetOpCount(const GraphDef& graph);

int64 GetParameterCount(const GraphDef& graph);

int64 GetVariableCount(const GraphDef& graph);

int GetControlEdgeCount(const GraphDef& graph);

std::map<string, int> GetDeviceCount(const GraphDef& graph);

}  // namespace decent_q
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_DECENT_Q_UTILS_CHECK_GRAPH_H_
