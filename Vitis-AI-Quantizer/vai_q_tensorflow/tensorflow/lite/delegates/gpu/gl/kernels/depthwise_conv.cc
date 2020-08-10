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

#include "tensorflow/lite/delegates/gpu/gl/kernels/depthwise_conv.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/ideal_workgroup_picker.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class DepthwiseConvolution : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto input = ctx.graph->FindInputs(ctx.node->id)[0];
    auto attr = absl::any_cast<const DepthwiseConvolution2DAttributes&>(
        ctx.node->operation.attributes);
    auto weights = attr.weights.shape;
    const int offsets_count = weights.h * weights.w;
    const bool offsets_count_too_large = offsets_count > kMaxConstArraySize;
    std::vector<Variable> parameters;
    if (offsets_count_too_large) {
      parameters = {
          {"input_data_0_h", input->tensor.shape.h},
          {"input_data_0_w", input->tensor.shape.w},
          {"padding_w", attr.padding.prepended.w},
          {"padding_h", attr.padding.prepended.h},
          {"dilation_w", attr.dilations.w},
          {"dilation_h", attr.dilations.h},
          {"kernel_w", weights.w},
          {"kernel_h", weights.h},
          {"src_depth", IntegralDivideRoundUp(weights.i, 4)},
          {"channel_multiplier", weights.o},
          {"stride", int2(attr.strides.w, attr.strides.h)},
      };
    } else {
      std::vector<int2> offsets;
      for (int h = 0; h < weights.h; ++h) {
        for (int w = 0; w < weights.w; ++w) {
          offsets.emplace_back(w * attr.dilations.w - attr.padding.prepended.w,
                               h * attr.dilations.h - attr.padding.prepended.h);
        }
      }
      parameters = {
          {"input_data_0_h", input->tensor.shape.h},
          {"input_data_0_w", input->tensor.shape.w},
          {"offsets_count", offsets_count},
          {"offsets", offsets},
          {"src_depth", IntegralDivideRoundUp(weights.i, 4)},
          {"channel_multiplier", weights.o},
          {"stride", int2(attr.strides.w, attr.strides.h)},
      };
    }
    bool non_empty_padding =
        attr.padding.appended.h != 0 || attr.padding.appended.w != 0 ||
        attr.padding.prepended.h != 0 || attr.padding.prepended.w != 0;

    std::vector<std::pair<std::string, Object>> objects = {
        {"weights", MakeReadonlyObject(ConvertToPIOHW4(attr.weights))}};

    std::string source;
    if (offsets_count_too_large) {
      source = R"(
        int offsets_count = $kernel_w$ * $kernel_h$;
        int src_layer_offset = (gid.z % $channel_multiplier$) * 4;
        int filter_offset = gid.z * $src_depth$ * offsets_count * 4;
        int i = 0;
        for (int ky = 0; ky < $kernel_h$; ky++) {
          for (int kx = 0; kx < $kernel_w$; kx++, i++) {
            ivec2 coord = gid.xy * $stride$ + ivec2(kx * $dilation_w$ - $padding_w$, ky * $dilation_h$ - $padding_h$);)";
    } else {
      source = R"(
        int offsets_count = $offsets_count$;
        int src_layer_offset = (gid.z % $channel_multiplier$) * 4;
        int filter_offset = gid.z * $src_depth$ * offsets_count * 4;
        for (int i = 0; i < offsets_count; ++i) {
          ivec2 coord = gid.xy * $stride$ + $offsets[i]$;)";
    }
    if (non_empty_padding) {
      source += R"(
        if (coord.x < 0 || coord.y < 0 ||
            coord.x >= $input_data_0_w$ || coord.y >= $input_data_0_h$) {
          continue;
        })";
    }
    source += R"(
        int src_layer = gid.z / $channel_multiplier$;
        vec4 input_ = $input_data_0[coord.x, coord.y, src_layer]$;
        highp vec4 input_shifted;
        input_shifted[0] = input_[(src_layer_offset + 0) / $channel_multiplier$];
        input_shifted[1] = input_[(src_layer_offset + 1) / $channel_multiplier$];
        input_shifted[2] = input_[(src_layer_offset + 2) / $channel_multiplier$];
        input_shifted[3] = input_[(src_layer_offset + 3) / $channel_multiplier$];
        int filter_offset = gid.z * offsets_count + i;
        value_0 += input_shifted * $weights[filter_offset]$;
      }
)";
    if (offsets_count_too_large) {
      source += R"(
      }
)";
    }
    if (!attr.bias.data.empty()) {
      source += "value_0 += $bias[gid.z]$;\n";
      objects.push_back({"bias", MakeReadonlyObject(attr.bias.data)});
    }
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/std::move(objects),
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/
        GetIdealWorkgroupIfPossible(
            ctx.gpu_info->gpu_model, OperationType::DEPTHWISE_CONVOLUTION,
            HW(attr.weights.shape.h, attr.weights.shape.w), attr.strides,
            OHWI(attr.weights.shape.o, input->tensor.shape.h,
                 input->tensor.shape.w, input->tensor.shape.c)),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewDepthwiseConvolutionNodeShader() {
  return absl::make_unique<DepthwiseConvolution>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
