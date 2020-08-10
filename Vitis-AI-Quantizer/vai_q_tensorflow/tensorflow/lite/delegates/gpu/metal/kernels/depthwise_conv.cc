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

#include "tensorflow/lite/delegates/gpu/metal/kernels/depthwise_conv.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetKernelDepthWiseConv3x3Stride1x1() {
  std::string code = R"(
#include <metal_stdlib>
using namespace metal;

struct uniforms {
  int4 src_size;
  int4 dst_size;
  int2 padding;
  int2 dummy0;  // for alignment
  int4 dummy1;  // for alignment
};
$0

kernel void ComputeFunction(
                            $1
                            uint3 ugid[[thread_position_in_grid]])
{
  int gid_x = ugid.x * 2;
  int gid_y = ugid.y * 2;
  int gid_z = ugid.z;

  if (gid_x >= params.dst_size.x || gid_y >= params.dst_size.y) {
      return;
  }

  ACCUM_FLT4 r0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 l0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 t0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 b0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);

  int x0 = gid_x + params.padding.x;
  int x1 = gid_x + params.padding.x + 1;
  int x2 = gid_x + params.padding.x + 2;
  int x3 = gid_x + params.padding.x + 3;
  int y0 = gid_y + params.padding.y;
  int y1 = gid_y + params.padding.y + 1;
  int y2 = gid_y + params.padding.y + 2;
  int y3 = gid_y + params.padding.y + 3;

  bool x0_out = x0 < 0 || x0 >= params.src_size.x;
  bool x1_out = x1 < 0 || x1 >= params.src_size.x;
  bool x2_out = x2 < 0 || x2 >= params.src_size.x;
  bool x3_out = x3 < 0 || x3 >= params.src_size.x;
  bool y0_out = y0 < 0 || y0 >= params.src_size.y;
  bool y1_out = y1 < 0 || y1 >= params.src_size.y;
  bool y2_out = y2 < 0 || y2 >= params.src_size.y;
  bool y3_out = y3 < 0 || y3 >= params.src_size.y;

  x0 = clamp(x0, 0, params.src_size.x - 1);
  x1 = clamp(x1, 0, params.src_size.x - 1);
  x2 = clamp(x2, 0, params.src_size.x - 1);
  x3 = clamp(x3, 0, params.src_size.x - 1);
  y0 = clamp(y0, 0, params.src_size.y - 1);
  y1 = clamp(y1, 0, params.src_size.y - 1);
  y2 = clamp(y2, 0, params.src_size.y - 1);
  y3 = clamp(y3, 0, params.src_size.y - 1);

  device FLT4* src_loc = src_buffer + gid_z * params.src_size.z;
  device FLT4* filters_loc = filters + gid_z * 10;

  FLT4 s0 = src_loc[y0 * params.src_size.x + x0] * FLT(!(x0_out || y0_out));
  FLT4 s1 = src_loc[y1 * params.src_size.x + x0] * FLT(!(x0_out || y1_out));
  FLT4 s2 = src_loc[y2 * params.src_size.x + x0] * FLT(!(x0_out || y2_out));
  FLT4 s3 = src_loc[y3 * params.src_size.x + x0] * FLT(!(x0_out || y3_out));

  r0 += TO_ACCUM4_TYPE(s0 * filters_loc[0]);
  r0 += TO_ACCUM4_TYPE(s1 * filters_loc[1]);
  r0 += TO_ACCUM4_TYPE(s2 * filters_loc[2]);
  l0 += TO_ACCUM4_TYPE(s1 * filters_loc[0]);
  l0 += TO_ACCUM4_TYPE(s2 * filters_loc[1]);
  l0 += TO_ACCUM4_TYPE(s3 * filters_loc[2]);

  s0 = src_loc[y0 * params.src_size.x + x1] * FLT(!(x1_out || y0_out));
  s1 = src_loc[y1 * params.src_size.x + x1] * FLT(!(x1_out || y1_out));
  s2 = src_loc[y2 * params.src_size.x + x1] * FLT(!(x1_out || y2_out));
  s3 = src_loc[y3 * params.src_size.x + x1] * FLT(!(x1_out || y3_out));

  r0 += TO_ACCUM4_TYPE(s0 * filters_loc[3]);
  r0 += TO_ACCUM4_TYPE(s1 * filters_loc[4]);
  r0 += TO_ACCUM4_TYPE(s2 * filters_loc[5]);
  l0 += TO_ACCUM4_TYPE(s1 * filters_loc[3]);
  l0 += TO_ACCUM4_TYPE(s2 * filters_loc[4]);
  l0 += TO_ACCUM4_TYPE(s3 * filters_loc[5]);
  t0 += TO_ACCUM4_TYPE(s0 * filters_loc[0]);
  t0 += TO_ACCUM4_TYPE(s1 * filters_loc[1]);
  t0 += TO_ACCUM4_TYPE(s2 * filters_loc[2]);
  b0 += TO_ACCUM4_TYPE(s1 * filters_loc[0]);
  b0 += TO_ACCUM4_TYPE(s2 * filters_loc[1]);
  b0 += TO_ACCUM4_TYPE(s3 * filters_loc[2]);

  s0 = src_loc[y0 * params.src_size.x + x2] * FLT(!(x2_out || y0_out));
  s1 = src_loc[y1 * params.src_size.x + x2] * FLT(!(x2_out || y1_out));
  s2 = src_loc[y2 * params.src_size.x + x2] * FLT(!(x2_out || y2_out));
  s3 = src_loc[y3 * params.src_size.x + x2] * FLT(!(x2_out || y3_out));

  r0 += TO_ACCUM4_TYPE(s0 * filters_loc[6]);
  r0 += TO_ACCUM4_TYPE(s1 * filters_loc[7]);
  r0 += TO_ACCUM4_TYPE(s2 * filters_loc[8]);
  l0 += TO_ACCUM4_TYPE(s1 * filters_loc[6]);
  l0 += TO_ACCUM4_TYPE(s2 * filters_loc[7]);
  l0 += TO_ACCUM4_TYPE(s3 * filters_loc[8]);
  t0 += TO_ACCUM4_TYPE(s0 * filters_loc[3]);
  t0 += TO_ACCUM4_TYPE(s1 * filters_loc[4]);
  t0 += TO_ACCUM4_TYPE(s2 * filters_loc[5]);
  b0 += TO_ACCUM4_TYPE(s1 * filters_loc[3]);
  b0 += TO_ACCUM4_TYPE(s2 * filters_loc[4]);
  b0 += TO_ACCUM4_TYPE(s3 * filters_loc[5]);

  s0 = src_loc[y0 * params.src_size.x + x3] * FLT(!(x3_out || y0_out));
  s1 = src_loc[y1 * params.src_size.x + x3] * FLT(!(x3_out || y1_out));
  s2 = src_loc[y2 * params.src_size.x + x3] * FLT(!(x3_out || y2_out));
  s3 = src_loc[y3 * params.src_size.x + x3] * FLT(!(x3_out || y3_out));

  t0 += TO_ACCUM4_TYPE(s0 * filters_loc[6]);
  t0 += TO_ACCUM4_TYPE(s1 * filters_loc[7]);
  t0 += TO_ACCUM4_TYPE(s2 * filters_loc[8]);
  b0 += TO_ACCUM4_TYPE(s1 * filters_loc[6]);
  b0 += TO_ACCUM4_TYPE(s2 * filters_loc[7]);
  b0 += TO_ACCUM4_TYPE(s3 * filters_loc[8]);

  r0 += TO_ACCUM4_TYPE(filters_loc[9]);
  l0 += TO_ACCUM4_TYPE(filters_loc[9]);
  t0 += TO_ACCUM4_TYPE(filters_loc[9]);
  b0 += TO_ACCUM4_TYPE(filters_loc[9]);

  const int offset_0 = gid_z * params.dst_size.z + gid_y * params.dst_size.x + gid_x;
  const int offset_1 = offset_0 + params.dst_size.x;
  const int offset_2 = offset_0 + 1;
  const int offset_3 = offset_0 + params.dst_size.x + 1;
  bool x0_in = gid_x < params.dst_size.x;
  bool x1_in = gid_x + 1 < params.dst_size.x;
  bool y0_in = gid_y < params.dst_size.y;
  bool y1_in = gid_y + 1 < params.dst_size.y;

  if (y0_in && x0_in) {
      int linear_index = offset_0;
      FLT4 value = FLT4(r0);
      uint3 gid = uint3(gid_x, gid_y, gid_z);
      $2
      dst_buffer[linear_index] = value;
  }
  if (y1_in && x0_in) {
      int linear_index = offset_1;
      FLT4 value = FLT4(l0);
      uint3 gid = uint3(gid_x, gid_y + 1, gid_z);
      $2
      dst_buffer[linear_index] = value;
  }
  if (y0_in && x1_in) {
      int linear_index = offset_2;
      FLT4 value = FLT4(t0);
      uint3 gid = uint3(gid_x + 1, gid_y, gid_z);
      $2
      dst_buffer[linear_index] = value;
  }
  if (y1_in && x1_in) {
      int linear_index = offset_3;
      FLT4 value = FLT4(b0);
      uint3 gid = uint3(gid_x + 1, gid_y + 1, gid_z);
      $2
      dst_buffer[linear_index] = value;
  }
}
  )";

  return code;
}

// Reorder weights to make the weights memory access pattern cache friendly for
// DepthWiseConv3x3Stride1x1
std::vector<float> ReorderWeightsDepthWiseConv3x3Stride1x1(
    const DepthwiseConvolution2DAttributes& attr) {
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  const int kernel_x = 3;
  const int kernel_y = 3;
  std::vector<float> weights_reordered((kernel_x * kernel_y + 1) * src_depth *
                                       4);

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int x = 0; x < kernel_x; ++x) {
      for (int y = 0; y < kernel_y; ++y) {
        for (int i = 0; i < 4; ++i) {
          const int s_ch = s * 4 + i;
          if (s_ch < attr.weights.shape.i) {
            const int f_index = attr.weights.shape.LinearIndex({0, y, x, s_ch});
            weights_reordered[counter++] = attr.weights.data[f_index];
          } else {
            weights_reordered[counter++] = 0.0f;
          }
        }
      }
    }

    for (int i = 0; i < 4; ++i) {
      const int dst_ch = s * 4 + i;
      if (dst_ch < attr.bias.shape.v) {
        weights_reordered[counter++] = attr.bias.data[dst_ch];
      } else {
        weights_reordered[counter++] = 0.0f;
      }
    }
  }

  return weights_reordered;
}

static std::vector<uint8_t> GetUniformBufferDepthWiseConv3x3Stride1x1(
    const BHWC& src_size, const BHWC& dst_size,
    const DepthwiseConvolution2DAttributes& params) {
  std::vector<int> uniform_params = {
      src_size.w,
      src_size.h,
      src_size.w * src_size.h,
      IntegralDivideRoundUp(src_size.c, 4),
      dst_size.w,
      dst_size.h,
      dst_size.w * dst_size.h,
      IntegralDivideRoundUp(dst_size.c, 4),
      -params.padding.prepended.w,
      -params.padding.prepended.h,
      0,  // dummy, for alignment
      0,  // dummy, for alignment
      0,  // dummy, for alignment
      0,  // dummy, for alignment
      0,  // dummy, for alignment
      0,  // dummy, for alignment
  };
  return VectorToUint8Vector(uniform_params);
}

std::string GetKernelDepthWiseConv3x3Stride2() {
  std::string code = R"(
#include <metal_stdlib>
using namespace metal;

struct uniforms {
  int4 src_size;
  int4 dst_size;
  int2 padding;
  int2 stride;
  int2 dilation;
  int2 dummy0;  // for alignment
};
$0

kernel void ComputeFunction(
                            $1
                            uint3 ugid[[thread_position_in_grid]])
{
    int gid_x = ugid.x;
    int gid_y = ugid.y * 2;
    int gid_z = ugid.z;

    if (gid_x >= params.dst_size.x || gid_y >= params.dst_size.y) {
        return;
    }

    device FLT4* src_loc = src_buffer + gid_z * params.src_size.z;
    device FLT4* filters_loc = filters + gid_z * 10;

    ACCUM_FLT4 r0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);
    ACCUM_FLT4 l0 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);

    int x0 = gid_x * params.stride.x + params.padding.x;
    int x1 = gid_x * params.stride.x + params.padding.x + params.dilation.x;
    int x2 = gid_x * params.stride.x + params.padding.x + 2 * params.dilation.x;
    int y0 = gid_y * 2 + params.padding.y;
    int y1 = gid_y * 2 + params.padding.y + 1;
    int y2 = gid_y * 2 + params.padding.y + 2;
    int y3 = gid_y * 2 + params.padding.y + 3;
    int y4 = gid_y * 2 + params.padding.y + 4;

    bool x0_out = x0 < 0 || x0 >= params.src_size.x;
    bool x1_out = x1 < 0 || x1 >= params.src_size.x;
    bool x2_out = x2 < 0 || x2 >= params.src_size.x;
    bool y0_out = y0 < 0 || y0 >= params.src_size.y;
    bool y1_out = y1 < 0 || y1 >= params.src_size.y;
    bool y2_out = y2 < 0 || y2 >= params.src_size.y;
    bool y3_out = y3 < 0 || y3 >= params.src_size.y;
    bool y4_out = y4 < 0 || y4 >= params.src_size.y;

    x0 = clamp(x0, 0, params.src_size.x - 1);
    x1 = clamp(x1, 0, params.src_size.x - 1);
    x2 = clamp(x2, 0, params.src_size.x - 1);
    y0 = clamp(y0, 0, params.src_size.y - 1);
    y1 = clamp(y1, 0, params.src_size.y - 1);
    y2 = clamp(y2, 0, params.src_size.y - 1);
    y3 = clamp(y3, 0, params.src_size.y - 1);
    y4 = clamp(y4, 0, params.src_size.y - 1);

    FLT4 s0 = src_loc[y0 * params.src_size.x + x0] * FLT(!(x0_out || y0_out));
    FLT4 s1 = src_loc[y0 * params.src_size.x + x1] * FLT(!(x1_out || y0_out));
    FLT4 s2 = src_loc[y0 * params.src_size.x + x2] * FLT(!(x2_out || y0_out));

    r0 += TO_ACCUM4_TYPE(s0 * filters_loc[0]);
    r0 += TO_ACCUM4_TYPE(s1 * filters_loc[1]);
    r0 += TO_ACCUM4_TYPE(s2 * filters_loc[2]);

    s0 = src_loc[y1 * params.src_size.x + x0] * FLT(!(x0_out || y1_out));
    s1 = src_loc[y1 * params.src_size.x + x1] * FLT(!(x1_out || y1_out));
    s2 = src_loc[y1 * params.src_size.x + x2] * FLT(!(x2_out || y1_out));

    r0 += TO_ACCUM4_TYPE(s0 * filters_loc[3]);
    r0 += TO_ACCUM4_TYPE(s1 * filters_loc[4]);
    r0 += TO_ACCUM4_TYPE(s2 * filters_loc[5]);

    s0 = src_loc[y2 * params.src_size.x + x0] * FLT(!(x0_out || y2_out));
    s1 = src_loc[y2 * params.src_size.x + x1] * FLT(!(x1_out || y2_out));
    s2 = src_loc[y2 * params.src_size.x + x2] * FLT(!(x2_out || y2_out));

    r0 += TO_ACCUM4_TYPE(s0 * filters_loc[6]);
    r0 += TO_ACCUM4_TYPE(s1 * filters_loc[7]);
    r0 += TO_ACCUM4_TYPE(s2 * filters_loc[8]);
    l0 += TO_ACCUM4_TYPE(s0 * filters_loc[0]);
    l0 += TO_ACCUM4_TYPE(s1 * filters_loc[1]);
    l0 += TO_ACCUM4_TYPE(s2 * filters_loc[2]);

    s0 = src_loc[y3 * params.src_size.x + x0] * FLT(!(x0_out || y3_out));
    s1 = src_loc[y3 * params.src_size.x + x1] * FLT(!(x1_out || y3_out));
    s2 = src_loc[y3 * params.src_size.x + x2] * FLT(!(x2_out || y3_out));

    l0 += TO_ACCUM4_TYPE(s0 * filters_loc[3]);
    l0 += TO_ACCUM4_TYPE(s1 * filters_loc[4]);
    l0 += TO_ACCUM4_TYPE(s2 * filters_loc[5]);

    s0 = src_loc[y4 * params.src_size.x + x0] * FLT(!(x0_out || y4_out));
    s1 = src_loc[y4 * params.src_size.x + x1] * FLT(!(x1_out || y4_out));
    s2 = src_loc[y4 * params.src_size.x + x2] * FLT(!(x2_out || y4_out));

    l0 += TO_ACCUM4_TYPE(s0 * filters_loc[6]);
    l0 += TO_ACCUM4_TYPE(s1 * filters_loc[7]);
    l0 += TO_ACCUM4_TYPE(s2 * filters_loc[8]);

    r0 += TO_ACCUM4_TYPE(filters_loc[9]);
    l0 += TO_ACCUM4_TYPE(filters_loc[9]);

    const int offset_0 = gid_z * params.dst_size.z
      + gid_y * params.dst_size.x + gid_x;
    const int offset_1 = offset_0 + params.dst_size.x;
    bool y0_in = gid_y < params.dst_size.y;
    bool y1_in = gid_y + 1 < params.dst_size.y;

    if (y0_in) {
        int linear_index = offset_0;
        FLT4 value = FLT4(r0);
        uint3 gid = uint3(gid_x, gid_y, gid_z);
        $2
        dst_buffer[linear_index] = value;
    }
    if (y1_in) {
        int linear_index = offset_1;
        FLT4 value = FLT4(l0);
        uint3 gid = uint3(gid_x, gid_y, gid_z);
        $2
        dst_buffer[linear_index] = value;
    }
}
  )";

  return code;
}

// Reorder weights to make the weights memory access pattern cache friendly for
// DepthWiseConv3x3Stride2
std::vector<float> ReorderWeightsDepthWiseConv3x3Stride2(
    const DepthwiseConvolution2DAttributes& attr) {
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  const int kernel_x = 3;
  const int kernel_y = 3;
  std::vector<float> weights_reordered((kernel_x * kernel_y + 1) * src_depth *
                                       4);

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int i = 0; i < 4; ++i) {
          const int s_ch = s * 4 + i;
          if (s_ch < attr.weights.shape.i) {
            const int f_index = attr.weights.shape.LinearIndex({0, y, x, s_ch});
            weights_reordered[counter++] = attr.weights.data[f_index];
          } else {
            weights_reordered[counter++] = 0.0f;
          }
        }
      }
    }

    for (int i = 0; i < 4; ++i) {
      const int dst_ch = s * 4 + i;
      if (dst_ch < attr.bias.shape.v) {
        weights_reordered[counter++] = attr.bias.data[dst_ch];
      } else {
        weights_reordered[counter++] = 0.0f;
      }
    }
  }

  return weights_reordered;
}

static std::vector<uint8_t> GetUniformBufferDepthWiseConv3x3Stride2(
    const BHWC& src_size, const BHWC& dst_size,
    const DepthwiseConvolution2DAttributes& attr) {
  std::vector<int> uniform_params = {
      src_size.w,
      src_size.h,
      src_size.w * src_size.h,
      IntegralDivideRoundUp(src_size.c, 4),
      dst_size.w,
      dst_size.h,
      dst_size.w * dst_size.h,
      IntegralDivideRoundUp(dst_size.c, 4),
      -attr.padding.prepended.w,
      -attr.padding.prepended.h,
      attr.strides.w,
      attr.strides.h,
      attr.dilations.w,
      attr.dilations.h,
      0,  // dummy, for alignment
      0,  // dummy, for alignment
  };
  return VectorToUint8Vector(uniform_params);
}

}  // namespace

std::vector<ComputeTaskDescriptorPtr> DepthWiseConvolution(
    int id, ValueId input_id, ValueId output_id,
    const DepthwiseConvolution2DAttributes& attr,
    const RuntimeOptions& options) {
  int channels_multiplier = attr.weights.shape.o;
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  std::string shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;
    constant int kernel_x = $0;
    constant int kernel_y = $1;
    struct uniforms {
      int4 stride;
      int4 padding;
      int4 dillation;
      int4 size;
      int4 channel_multiplier;
    };
    $$0
    kernel void ComputeFunction(
                                $$1
                                uint tid[[thread_index_in_threadgroup]],
                                uint3 gid[[thread_position_in_grid]]) {
      const bool outside = static_cast<int>(gid.x) >= params.size.z ||
        static_cast<int>(gid.y) >= params.size.w;
      if (outside) {
        return;
      }
      device FLT4* temp = filters + gid.z * kernel_y * kernel_x;
      float4 sum0 = float4(0.0f, 0.0f, 0.0f, 0.0f);

      for(int ky = 0; ky < kernel_y; ++ky) {
        for(int kx = 0; kx < kernel_x; ++kx) {
          int2 coords  = int2(gid.xy) * params.stride.xy + int2(kx, ky) * params.dillation.xy -
            params.padding.xy;
          const bool outside = coords.x < 0 || coords.y < 0 ||
            coords.x >= params.size.x || coords.y >= params.size.y;
          if (outside) continue;
)";
  if (channels_multiplier == 1) {
    shader_source += R"(
        const int src_layer = gid.z;
        const int src_index = (src_layer * params.size.y + coords.y) * params.size.x + coords.x;
        const FLT4 src_modified = src_buffer[src_index];
)";
  } else if (channels_multiplier == 2) {
    shader_source += R"(
        const int src_layer = gid.z / 2;
        const int src_index = (src_layer * params.size.y + coords.y) * params.size.x + coords.x;
        const FLT4 src = src_buffer[src_index];
        const FLT2 t0 = gid.z % 2 == 0 ? src.xy : src.zw;
        const FLT4 src_modified = FLT4(t0.x, t0.x, t0.y, t0.y);
)";
  } else if (channels_multiplier == 4) {
    shader_source += R"(
        const int src_layer = gid.z / 4;
        const int src_index = (src_layer * params.size.y + coords.y) * params.size.x + coords.x;
        const FLT4 src = src_buffer[src_index];
        const FLT t0 = src[gid.z % 4];
        const FLT4 src_modified = FLT4(t0, t0, t0, t0);
)";
  } else {
    shader_source += R"(
        const int src_layer = gid.z / params.channel_multiplier.x;
        const int src_index = (src_layer * params.size.y + coords.y) * params.size.x + coords.x;
        const FLT4 src = src_buffer[src_index];
        FLT4 src_modified;
        const int src_layer_offset = (gid.z % params.channel_multiplier.x) * 4;
        src_modified.x = src[(src_layer_offset + 0) / params.channel_multiplier.x];
        src_modified.y = src[(src_layer_offset + 1) / params.channel_multiplier.x];
        src_modified.z = src[(src_layer_offset + 2) / params.channel_multiplier.x];
        src_modified.w = src[(src_layer_offset + 3) / params.channel_multiplier.x];
)";
  }
  shader_source += R"(
          sum0 += float4(src_modified * temp[ky * kernel_x + kx]);
        }
      }
      FLT4 res = FLT4(sum0 + float4(biases[gid.z]));
      const int linear_index = (gid.z * params.size.w + int(gid.y)) * params.size.z + int(gid.x);
      FLT4 value = res;
      $$2
      output_buffer[linear_index] = value;
    }
  )";
  desc->shader_source = absl::Substitute(shader_source, attr.weights.shape.w,
                                         attr.weights.shape.h);

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* output_buffer",
      [input_id, attr](const std::map<ValueId, BHWC>& buffers) {
        auto out_shape =
            CalculateOutputShape(buffers.find(input_id)->second, attr);
        return out_shape;
      }};

  std::vector<float> filters_reordered = ConvertToPIOHW4(attr.weights);
  auto filters = options.storage_precision == RuntimeOptions::Precision::FP32
                     ? VectorToUint8Vector(filters_reordered)
                     : VectorFloatToHalf(filters_reordered);
  auto biases = options.storage_precision == RuntimeOptions::Precision::FP32
                    ? VectorToUint8Vector(attr.bias.data)
                    : VectorFloatToHalf(attr.bias.data);
  desc->immutable_buffers = {
      {"device FLT4* const filters", filters},
      {"device FLT4* const biases", biases},
  };

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id, attr](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_id)->second;
         const auto& output_dimension = buffers.find(output_id)->second;
         std::vector<int> uniform_params{
             attr.strides.w,
             attr.strides.h,
             1,
             1,
             attr.padding.prepended.w,
             attr.padding.prepended.h,
             1,
             1,
             attr.dilations.w,
             attr.dilations.h,
             1,
             1,
             dimension.w,
             dimension.h,
             output_dimension.w,
             output_dimension.h,
             attr.weights.shape.o,
             0,
             0,
             0,
         };
         return VectorToUint8Vector(uniform_params);
       }},
  };

  desc->resize_function = [output_id](const std::map<ValueId, BHWC>& buffers) {
    const auto& dimension = buffers.find(output_id)->second;
    uint3 groups_size{8, 4, 1};
    uint3 groups_count{IntegralDivideRoundUp(dimension.w, groups_size.x),
                       IntegralDivideRoundUp(dimension.h, groups_size.y),
                       IntegralDivideRoundUp(dimension.c, 4)};
    return std::make_pair(groups_size, groups_count);
  };

  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> DepthWiseConv3x3Stride1x1(
    int id, ValueId input_id, ValueId output_id,
    const DepthwiseConvolution2DAttributes& attr,
    const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetKernelDepthWiseConv3x3Stride1x1();

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, attr](const std::map<ValueId, BHWC>& buffers) {
        auto out_shape =
            CalculateOutputShape(buffers.find(input_id)->second, attr);
        return out_shape;
      }};

  // For this operation we keep weights and biases in one buffer
  auto weights_reordered = ReorderWeightsDepthWiseConv3x3Stride1x1(attr);
  auto weights =
      options.storage_precision == metal::RuntimeOptions::Precision::FP32
          ? VectorToUint8Vector(weights_reordered)
          : VectorFloatToHalf(weights_reordered);
  desc->immutable_buffers = {
      {"device FLT4* const filters", weights},
  };

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id, attr](const std::map<ValueId, BHWC>& buffers) {
         const auto& input_dimensions = buffers.find(input_id)->second;
         const auto& output_dimensions = buffers.find(output_id)->second;
         return GetUniformBufferDepthWiseConv3x3Stride1x1(
             input_dimensions, output_dimensions, attr);
       }},
  };

  desc->resize_function = [output_id](const std::map<ValueId, BHWC>& buffers) {
    const auto& dimension = buffers.find(output_id)->second;
    const int grid_x = IntegralDivideRoundUp(dimension.w, 2);
    const int grid_y = IntegralDivideRoundUp(dimension.h, 2);
    const int grid_z = IntegralDivideRoundUp(dimension.c, 4);
    uint3 group_size{8, 4, 1};
    if (grid_x <= 4) {
      group_size.x = 4;
      group_size.z = grid_z % 2 == 0 ? 2 : 1;
    }
    const int groups_x = IntegralDivideRoundUp(grid_x, group_size.x);
    const int groups_y = IntegralDivideRoundUp(grid_y, group_size.y);
    const int groups_z = IntegralDivideRoundUp(grid_z, group_size.z);
    return std::make_pair(group_size, uint3(groups_x, groups_y, groups_z));
  };

  return {desc};
}

bool CheckDepthWiseConv3x3Stride1x1Support(
    const DepthwiseConvolution2DAttributes& attr) {
  return attr.weights.shape.o == 1 && attr.weights.shape.h == 3 &&
         attr.weights.shape.w == 3 && attr.strides.h == 1 &&
         attr.strides.w == 1 && attr.dilations.h == 1 && attr.dilations.w == 1;
}

std::vector<ComputeTaskDescriptorPtr> DepthWiseConv3x3Stride2(
    int id, ValueId input_id, ValueId output_id,
    const DepthwiseConvolution2DAttributes& attr,
    const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetKernelDepthWiseConv3x3Stride2();

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, attr](const std::map<ValueId, BHWC>& buffers) {
        auto out_shape =
            CalculateOutputShape(buffers.find(input_id)->second, attr);
        return out_shape;
      }};

  // For this operation we keep weights and biases in one buffer
  auto weights_reordered = ReorderWeightsDepthWiseConv3x3Stride2(attr);
  auto weights =
      options.storage_precision == metal::RuntimeOptions::Precision::FP32
          ? VectorToUint8Vector(weights_reordered)
          : VectorFloatToHalf(weights_reordered);
  desc->immutable_buffers = {
      {"device FLT4* const filters", weights},
  };

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id, attr](const std::map<ValueId, BHWC>& buffers) {
         const auto& input_dimensions = buffers.find(input_id)->second;
         const auto& output_dimensions = buffers.find(output_id)->second;
         return GetUniformBufferDepthWiseConv3x3Stride2(
             input_dimensions, output_dimensions, attr);
       }},
  };

  desc->resize_function = [output_id](const std::map<ValueId, BHWC>& buffers) {
    const auto& dimension = buffers.find(output_id)->second;
    const int grid_x = dimension.w;
    const int grid_y = IntegralDivideRoundUp(dimension.h, 2);
    const int grid_z = IntegralDivideRoundUp(dimension.c, 4);
    const uint3 group_size{8, 4, 1};
    const int groups_x = IntegralDivideRoundUp(grid_x, group_size.x);
    const int groups_y = IntegralDivideRoundUp(grid_y, group_size.y);
    const int groups_z = IntegralDivideRoundUp(grid_z, group_size.z);
    return std::make_pair(group_size, uint3(groups_x, groups_y, groups_z));
  };

  return {desc};
}

bool CheckDepthWiseConv3x3Stride2Support(
    const DepthwiseConvolution2DAttributes& attr) {
  return attr.weights.shape.o == 1 && attr.weights.shape.h == 3 &&
         attr.weights.shape.w == 3 && attr.strides.h == 2 &&
         attr.dilations.h == 1;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
