// Copyright 2019 Xilinx Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <stdio.h>

void GSTilingLayer_forward_c(const void *top_np,
                             const void *bottom_np,
                             const int input_batch,
                             const int input_channels,
                             const int input_height,
                             const int input_width,
                             const int stride) {
  int stride_sq = stride * stride;

  const float *bottom = (float *)bottom_np;
  float *top = (float *)top_np;

  int output_batch = input_batch;
  int output_channels = input_channels/stride_sq;
  int output_height = input_height*stride;
  int output_width = input_width*stride;

  int count_per_output_map = output_width * output_height * output_channels;
  int count_per_input_map = input_width * input_height * input_channels;

  const float *bottom_data = bottom;
  float *top_data = top;
  int ox,oy,oc,oi;
  int n, ic, iy, ix;

  for (n = 0; n < input_batch; ++n) {
    int ii = 0;
    for (ic = 0; ic < input_channels; ++ic) {
      for (iy = 0; iy < input_height; ++iy) {
        for (ix = 0; ix < input_width; ++ix) {
          int off = ic / output_channels;
          ox = ix * stride + off % stride;
          oy = iy * stride + off / stride;
          oc = ic % output_channels;

          oi = (oc * output_height + oy) * output_width + ox;
          top[oi] = bottom[ii];
          ++ii;
        }
      }
    }
    bottom_data += count_per_input_map;
    top_data += count_per_output_map;
  }
}


void hello() {
  printf("Hello world\n");
}
