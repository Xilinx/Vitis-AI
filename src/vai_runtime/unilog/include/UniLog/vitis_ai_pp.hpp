/*
 * Copyright 2019 Xilinx Inc.
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

#ifndef UNILOG_VITIS_AI_PP_H
#define UNILOG_VITIS_AI_PP_H

#define VITIS_AI_PP_SIZE(...)                                                  \
  VITIS_AI_PP_NARG_(__VA_ARGS__, VITIS_AI_PP_RSEQ_N())
#define VITIS_AI_PP_NARG_(...) VITIS_AI_PP_ARG_N(__VA_ARGS__)
#define VITIS_AI_PP_ARG_N(                                                     \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16,     \
    _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, \
    _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, \
    _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, \
    _62, _63, N, ...)                                                          \
  N
#define VITIS_AI_PP_RSEQ_N()                                                   \
  63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45,  \
      44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27,  \
      26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,   \
      8, 7, 6, 5, 4, 3, 2, 1, 0

#define VITIS_AI_PP_LOOP_1(func, ...) func(__VA_ARGS__)
#define VITIS_AI_PP_LOOP_2(func, _V1, ...)                                     \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_1(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_3(func, _V1, ...)                                     \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_2(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_4(func, _V1, ...)                                     \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_3(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_5(func, _V1, ...)                                     \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_4(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_6(func, _V1, ...)                                     \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_5(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_7(func, _V1, ...)                                     \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_6(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_8(func, _V1, ...)                                     \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_7(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_9(func, _V1, ...)                                     \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_8(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_10(func, _V1, ...)                                    \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_9(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_11(func, _V1, ...)                                    \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_10(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_12(func, _V1, ...)                                    \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_11(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_13(func, _V1, ...)                                    \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_12(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_14(func, _V1, ...)                                    \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_13(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_15(func, _V1, ...)                                    \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_14(func, __VA_ARGS__)
#define VITIS_AI_PP_LOOP_16(func, _V1, ...)                                    \
  VITIS_AI_PP_LOOP_1(func, _V1) VITIS_AI_PP_LOOP_15(func, __VA_ARGS__)

#define VITIS_AI_PP_CAT(_V1, _V2) VITIS_AI_PP_CAT_1(_V1, _V2)
#define VITIS_AI_PP_CAT_1(_V1, _V2) _V1##_V2

/* vitis_pp_loop max loop 16 values */
#define VITIS_AI_PP_LOOP(_func, ...)                                           \
  VITIS_AI_PP_CAT(VITIS_AI_PP_LOOP_, VITIS_AI_PP_SIZE(__VA_ARGS__))            \
  (_func, __VA_ARGS__)

#endif // UNILOG_VITIS_AI_PP_H
