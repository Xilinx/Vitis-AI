/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/kernels/upsampling_bilinear.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class UpsamplingBilinear : public NodeShader {
 public:
  UpsamplingBilinear() {}

  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto input = ctx.graph->FindInputs(ctx.node->id)[0];
    auto output = ctx.graph->FindOutputs(ctx.node->id)[0];
    auto attr =
        absl::any_cast<Upsample2DAttributes>(ctx.node->operation.attributes);

    if (input->tensor.shape.w > output->tensor.shape.w ||
        input->tensor.shape.h > output->tensor.shape.h) {
      return InvalidArgumentError("Output size is less than input size.");
    }
    if (output->tensor.shape.w != attr.new_shape.w ||
        output->tensor.shape.h != attr.new_shape.h) {
      return InvalidArgumentError(
          "Output size does not match new_size in attributes.");
    }
    if (input->tensor.shape.c != output->tensor.shape.c) {
      return InvalidArgumentError("Input/output channels mismatch.");
    }
    if (attr.type != UpsamplingType::BILINEAR) {
      return UnimplementedError("Upsample2D supports only bilinear type.");
    }
    if (input->tensor.shape.h == 1 && input->tensor.shape.w == 1) {
      // Copy a single element from input.
      *generated_code = {
          /*parameters=*/{},
          /*objects=*/{},
          /*shared_variables=*/{},
          /*workload=*/uint3(),
          /*workgroup=*/uint3(),
          /*source_code=*/"value_0 = $input_data_0[0, 0, gid.z]$;",
          /*input=*/IOStructure::ONLY_DEFINITIONS,
          /*output=*/IOStructure::AUTO,
      };
      return OkStatus();
    }
    std::vector<Variable> parameters = {
        {"input_data_0_h", input->tensor.shape.h},
        {"input_data_0_w", input->tensor.shape.w},
        {"scale_factor",
         float2(CalculateResizeScale(input->tensor.shape.w,
                                     output->tensor.shape.w, attr),
                CalculateResizeScale(input->tensor.shape.h,
                                     output->tensor.shape.h, attr))},
    };

    std::string source = R"(
  vec2 coord = vec2(gid.xy) * $scale_factor$;

  ivec2 borders = ivec2($input_data_0_w$, $input_data_0_h$) - ivec2(1, 1);
  ivec4 st;
  st.xy = ivec2(coord);
  st.zw = min(st.xy + ivec2(1, 1), borders);

  vec2 t = coord - vec2(st.xy); //interpolating factors

  vec4 tex11 = $input_data_0[st.x, st.y, gid.z]$;
  vec4 tex21 = $input_data_0[st.z, st.y, gid.z]$;
  vec4 tex12 = $input_data_0[st.x, st.w, gid.z]$;
  vec4 tex22 = $input_data_0[st.z, st.w, gid.z]$;

  value_0 = mix(mix(tex11, tex21, t.x), mix(tex12, tex22, t.x), t.y);
)";
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewUpsamplingNodeShader() {
  return absl::make_unique<UpsamplingBilinear>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
