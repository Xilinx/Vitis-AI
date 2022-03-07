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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

REGISTER_OP("FixNeuron")
    .Input("input: float32")
    .Output("quantized: float32")
    .Attr("bit_width: int = 8")
    .Attr("method: int = 1")
    .Attr("mode: int")
    .Attr("phase: int")
    .Attr("output_dir: string= ''")
    .Attr("quantize_pos: int = 0")
    .Attr("T: type = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Applies quantize for input tensor.

bit_width: bit_width for quantization, default is 8
method: 0: Non-Overflow 1: Min-Diff, default is 1
mode: 0: Activation 1: Weights
phase: 0: Calibration 1: Evaluation, 2: Training
output_dir: Directory of the temp quantize information
quantize_pos: will be updated by decent_q during calibration phase, default is 0
)doc");

}  // namespace tensorflow
