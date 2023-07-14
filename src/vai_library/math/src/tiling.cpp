/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdint>

#include "vitis/ai/math.hpp"
namespace vitis {
namespace ai {

static void tilling_c(const int8_t *input, unsigned int width,
                      unsigned int height, unsigned int tile_dim,
                      unsigned int ch, int8_t *output) {
  unsigned int o_ch_width = ch * tile_dim * width;
  unsigned int tile_size = tile_dim * tile_dim;
  unsigned int i_ch = tile_size * ch;
  unsigned int i_ch_width = i_ch * width;

  // i(iy, ix, c*tile_size + ty*tile_dim + tx) -> o(iy, ix, ty, tx, c)
  for (unsigned int iy = 0; iy < height; ++iy) {
    for (unsigned int ix = 0; ix < width; ++ix) {
      for (unsigned int ty = 0; ty < tile_dim; ++ty) {
        for (unsigned int tx = 0; tx < tile_dim; ++tx) {
          for (unsigned int c = 0; c < ch; ++c) {
            unsigned int oy = iy * tile_dim + ty;
            unsigned int ox = ix * tile_dim + tx;
            unsigned int oi = oy * o_ch_width + ox * ch + c;
            unsigned int ic = ty * tile_dim + tx + c * tile_size;
            unsigned int ii = iy * i_ch_width + ix * i_ch + ic;
            output[oi] = input[ii];
          }
        }
      }
    }
  }
}
#ifdef ENABLE_NEON
#include <arm_neon.h>

// using vitis::dpmath::softmax2_vector_o1_neon;

// till_dim = 8
// ch = 2
static void tilling_t8_c2_neon(const int8_t *input, unsigned int width,
                               unsigned int height, int8_t *output) {
  unsigned int i_ch_width = (width << 7);
  for (unsigned int h = 0; h < height; ++h) {
    for (unsigned int t = 0; t < 8; ++t) {
      for (unsigned int w = 0; w < width; ++w) {
        // (h, w, c) = (h, w, 8t), (h, w, 8t+64)
        unsigned int i0 = h * i_ch_width + (w << 7) + (t << 3);
        unsigned int i1 = i0 + 64;

        int8x8_t d0 = vld1_s8(input + i0);
        int8x8_t d1 = vld1_s8(input + i1);
        int8x8x2_t q0 = {d0, d1};
        vst2_s8(output, q0);

        output += 16;
      }
    }
  }
}

// void tilling_t8_c2_softmax2_o1_neon(const int8_t *input, float scale,
//                                     unsigned int width, unsigned int
// height,
//                                     float *output) {
//   unsigned int i_ch_width = (width << 7);
//   for (unsigned int h = 0; h < height; ++h)
//     for (unsigned int t = 0; t < 8; ++t)
//       for (unsigned int w = 0; w < width; ++w) {
//         // (h, w, c) = (h, w, 8t), (h, w, 8t+64)
//         unsigned int i0 = h * i_ch_width + (w << 7) + (t << 3);
//         unsigned int i1 = i0 + 64;

//         int8x8_t d0 = vld1_s8(input + i0);
//         int8x8_t d1 = vld1_s8(input + i1);
//         float32x4x2_t q01 = softmax2_vector_o1_neon(d0, d1, scale);

//         vst1q_f32(output, q01.val[0]);
//         output += 4;
//         vst1q_f32(output, q01.val[1]);
//         output += 4;
//       }
// }

// till_dim = 8
// ch = 4
static void tilling_t8_c4_neon(const int8_t *input, unsigned int width,
                               unsigned int height, int8_t *output) {
  unsigned int i_ch_width = (width << 8);
  for (unsigned int h = 0; h < height; ++h) {
    for (unsigned int t = 0; t < 8; ++t) {
      for (unsigned int w = 0; w < width; ++w) {
        // (h, w, c) = (h, w, 8t), (h, w, 8t+64)
        unsigned int i0 = h * i_ch_width + (w << 8) + (t << 3);
        unsigned int i1 = i0 + 64;
        unsigned int i2 = i1 + 64;
        unsigned int i3 = i2 + 64;

        int8x8_t d0 = vld1_s8(input + i0);
        int8x8_t d1 = vld1_s8(input + i1);
        int8x8_t d2 = vld1_s8(input + i2);
        int8x8_t d3 = vld1_s8(input + i3);
        int8x8x4_t q01 = {d0, d1, d2, d3};
        vst4_s8(output, q01);

        output += 32;
      }
    }
  }
}
#endif
void tiling(const int8_t *input, unsigned int width, unsigned int height,
            unsigned int tile_dim, unsigned int ch, int8_t *output) {
#ifdef ENABLE_NEON
  if (tile_dim == 8 && ch == 4) {
    return tilling_t8_c4_neon(input, width, height, output);
  } else if (tile_dim == 8 && ch == 2) {
    return tilling_t8_c2_neon(input, width, height, output);
  } else {
    return tilling_c(input, width, height, tile_dim, ch, output);
  }
#else
  return tilling_c(input, width, height, tile_dim, ch, output);
#endif
}
}  // namespace ai
}  // namespace vitis
