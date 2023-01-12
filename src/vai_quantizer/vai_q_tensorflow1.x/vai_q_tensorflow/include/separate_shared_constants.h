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

#ifndef TENSORFLOW_CONTRIB_DECENT_Q_UTILS_SEPARATE_SHARED_CONSTANTS_H_
#define TENSORFLOW_CONTRIB_DECENT_Q_UTILS_SEPARATE_SHARED_CONSTANTS_H_

#include "transform_utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace decent_q {

// Separate shared constants
Status SeparateSharedConstants(const GraphDef& input_graph_def,
                               GraphDef* output_graph_def, bool is_quantized=false);

}  // namespace decent_q
}  // namespace tensorflow

#endif  //  TENSORFLOW_CONTRIB_DECENT_Q_UTILS_SEPARATE_SHARED_CONSTANTS_H_
