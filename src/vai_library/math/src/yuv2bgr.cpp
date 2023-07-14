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
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include <assert.h>

#ifdef ENABLE_NEON
static int32x4_t CDE2R(int16x4_t C, int16x4_t D, int16x4_t E) {
  int32x4_t R = vdupq_n_s32(128);
  R = vmlal_s16(R, C, vdup_n_s16(298));
  R = vmlal_s16(R, E, vdup_n_s16(409));
  return R;
}
static int32x4_t CDE2G(int16x4_t C, int16x4_t D, int16x4_t E) {
  int32x4_t G = vdupq_n_s32(128);
  G = vmlal_s16(G, C, vdup_n_s16(298));
  G = vmlsl_s16(G, D, vdup_n_s16(100));
  G = vmlsl_s16(G, E, vdup_n_s16(208));
  return G;
}

static int32x4_t CDE2B(int16x4_t C, int16x4_t D, int16x4_t E) {
  int32x4_t B = vdupq_n_s32(128);
  B = vmlal_s16(B, C, vdup_n_s16(298));
  B = vmlal_s16(B, D, vdup_n_s16(516));
  return B;
}

#define SPLIT(name, name2, constA)                                             \
  const uint8x8_t name##8_0_8 = vget_low_u8(name##8_0_16);                     \
  const uint8x8_t name##8_8_16 = vget_high_u8(name##8_0_16);                   \
  const int16x8_t name##16_0_8 = vreinterpretq_s16_u16(vmovl_u8(name##8_0_8)); \
  const int16x8_t name2##16_0_8 =                                              \
      vqsubq_s16(name##16_0_8, vdupq_n_s16(constA));                           \
  const int16x8_t name##16_8_16 =                                              \
      vreinterpretq_s16_u16(vmovl_u8(name##8_8_16));                           \
  const int16x8_t name2##16_8_16 =                                             \
      vqsubq_s16(name##16_8_16, vdupq_n_s16(constA));                          \
  /*const int16x4_t name##16_0_4 = vget_low_s16(name##16_0_8);*/               \
  const int16x4_t name2##16_0_4 = vget_low_s16(name2##16_0_8);                 \
  const int16x4_t name##16_4_8 = vget_high_s16(name##16_0_8);                  \
  (void)name##16_4_8;                                                          \
  const int16x4_t name2##16_4_8 = vget_high_s16(name2##16_0_8);                \
  const int16x4_t name##16_8_12 = vget_low_s16(name##16_8_16);                 \
  (void)name##16_8_12;                                                         \
  const int16x4_t name2##16_8_12 = vget_low_s16(name2##16_8_16);               \
  const int16x4_t name##16_12_16 = vget_high_s16(name##16_8_16);               \
  (void)name##16_12_16;                                                        \
  const int16x4_t name2##16_12_16 = vget_high_s16(name2##16_8_16);

#define CDE2R(EorO, index)              \
  const int32x4_t R##EorO##32_##index = \
      CDE2R(C##EorO##16_##index, D16_##index, E16_##index);

#define CDE2B(EorO, index)              \
  const int32x4_t B##EorO##32_##index = \
      CDE2B(C##EorO##16_##index, D16_##index, E16_##index);

#define CDE2G(EorO, index)              \
  const int32x4_t G##EorO##32_##index = \
      CDE2G(C##EorO##16_##index, D16_##index, E16_##index);
namespace vitis {
namespace ai {
void yuv2bgr(int left, int top,                            //
             int width, int height,                        //
             unsigned char *__restrict y, int stride_y,    //
             unsigned char *__restrict uv, int stride_uv,  //
             unsigned char *bgr) {
  assert((uintptr_t)y % 16 == 0);
  assert((uintptr_t)uv % 16 == 0);
  assert((uintptr_t)bgr % 16 == 0);
  assert(stride_y % 16 == 0);
  assert(width % 16 == 0);
  assert(height % 2 == 0);
  assert(left % 16 == 0);
  assert(top % 2 == 0);
  // const uint8x8_t _0 = vdup_n_u8(0);
  for (int h = 0; h < height; h += 2) {
    for (int w = 0; w < width; w += 16) {
      unsigned char *pye = y + (top + h) * stride_y + left + w;
      unsigned char *pyo = pye + stride_y;
      unsigned char *puv = uv + (top + h) * stride_uv / 2 + left + w;
      unsigned char *pbgre = bgr + h * width * 3 + w * 3;
      unsigned char *pbgro = bgr + h * width * 3 + width * 3 + w * 3;
      const uint8x16_t YE8_0_16 = vld1q_u8(pye);    // 16 个像素 Y, even line
      const uint8x16_t YO8_0_16 = vld1q_u8(pyo);    // 16 个像素 Y, odd line
      const uint8x8x2_t UV8_0_2_16 = vld2_u8(puv);  // 32 个像素的  UV
      const uint8x8_t U8_0_2_16 =
          UV8_0_2_16.val[0];  // 32 个像素的 U, 8 个值，16 个像素，2行
      const uint8x8_t V8_0_2_16 =
          UV8_0_2_16.val[1];  // 32 个像素的 V, 8 个值，16 个像素，2行
      const uint8x8x2_t U8x2_0_16 = vzip_u8(U8_0_2_16, U8_0_2_16);
      const uint8x8x2_t V8x2_0_16 = vzip_u8(V8_0_2_16, V8_0_2_16);
      const uint8x16_t U8_0_16 =
          vcombine_u8(U8x2_0_16.val[0], U8x2_0_16.val[1]);
      const uint8x16_t V8_0_16 =
          vcombine_u8(V8x2_0_16.val[0], V8x2_0_16.val[1]);
      SPLIT(YE, CE, 16);
      SPLIT(YO, CO, 16);
      SPLIT(U, D, 128);
      SPLIT(V, E, 128);
      CDE2R(E, 0_4);
      CDE2B(E, 0_4);
      CDE2G(E, 0_4);
      CDE2R(O, 0_4);
      CDE2B(O, 0_4);
      CDE2G(O, 0_4);
      CDE2R(E, 4_8);
      CDE2B(E, 4_8);
      CDE2G(E, 4_8);
      CDE2R(O, 4_8);
      CDE2B(O, 4_8);
      CDE2G(O, 4_8);
      CDE2R(E, 8_12);
      CDE2B(E, 8_12);
      CDE2G(E, 8_12);
      CDE2R(O, 8_12);
      CDE2B(O, 8_12);
      CDE2G(O, 8_12);
      CDE2R(E, 12_16);
      CDE2B(E, 12_16);
      CDE2G(E, 12_16);
      CDE2R(O, 12_16);
      CDE2B(O, 12_16);
      CDE2G(O, 12_16);

#define COLLECT(name)                                                       \
  uint8x16_t name =                                                         \
      vcombine_u8(vqmovun_s16(vcombine_s16(vqshrn_n_s32(name##32_0_4, 8),   \
                                           vqshrn_n_s32(name##32_4_8, 8))), \
                  vqmovun_s16(vcombine_s16(vqshrn_n_s32(name##32_8_12, 8),  \
                                           vqshrn_n_s32(name##32_12_16, 8))));
      COLLECT(RE);
      COLLECT(BE);
      COLLECT(GE);
      COLLECT(RO);
      COLLECT(BO);
      COLLECT(GO);
      // 输出要求是 BGR ，但是海思的似乎 UV 分量反了。
      uint8x16x3_t output_E = {RE, GE, BE};
      uint8x16x3_t output_O = {RO, GO, BO};
      vst3q_u8(pbgre, output_E);
      vst3q_u8(pbgro, output_O);
    }
  }
}
}  // namespace ai
}  // namespace vitis
#endif
