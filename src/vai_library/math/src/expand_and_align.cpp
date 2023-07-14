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
#include "vitis/ai/expand_and_align.hpp"

#include <algorithm>
#include <tuple>
namespace vitis {
namespace ai {

static int div_ceil(int a, int b) {  //
  return a / b + (a % b == 0 ? 0 : 1);
}
static int div_floor(int a, int b) {  //
  return a / b;
}

// static int div_round(int a, int b) {  //
//   return a / b + (a % b < b / 2 ? 0 : 1);
// }
static int in_range(int a, int min_value, int max_value) {
  return std::min(std::max(a, min_value), max_value);
}

/// min_w : 48
/// max_w : image width
/// x : recog face x
/// w : recog face width
/// ratio :
static std::pair<int, int> expand(int min_w, int max_w, int x, int w,
                                  float ratio) {
  const int d = w * ratio;  //
  const int x1 = x - d;
  const int x2 = x + w + d;
  //扩边后的x1 应该在区间[0,width-48)内
  const int x1_c = in_range(x1, 0, max_w - min_w);
  //扩边后的x2 应该在区间[x1+48,width) 内
  const int x2_c = in_range(x2, x1_c + min_w, max_w);
  //返回扩边后的开始坐标和最终扩边后图形宽度[x1_c,x2_c-x1_c）
  return std::make_pair(x1_c, x2_c - x1_c);
}

/// min_w : 48
/// max_w : image width
/// x : expend return  x1
/// w : expend return expend_image_width
/// a : align paramter
static std::pair<int, int> align(int min_w, int max_w, int x, int w, int a) {
  // 如果总宽度不是对齐像素的整数倍，最多 a-1 个像素会被丢弃掉
  const int min_x_in_a = div_ceil(min_w, a);  // 48, 32, 2
  // 如果总宽度不是对齐像素的整数倍，最多 a-1 个像素会被丢弃掉
  const int max_x_in_a = div_floor(max_w, a);

  // 对齐的时候，左边尽量多扩一些
  const int x1_in_a = div_floor(x, a);  // 已经一定大于等于0 ，不必再判断
  // 对齐的时候，右边也尽量多扩一些。
  const int x2_in_a = div_ceil(x + w, a);
  // const int x2_in_a_c = std::min(x2_in_a, max_x_in_a);

  const int x1_c_in_a = in_range(x1_in_a, 0, max_x_in_a);
  const int x2_c_in_a = in_range(x2_in_a, x1_in_a + min_x_in_a, max_x_in_a);
  const int aligned_x = x1_c_in_a * a;
  const int aligned_w = (x2_c_in_a - x1_c_in_a) * a;
  return std::make_pair(aligned_x, aligned_w);
}

ExpandAndAlignX expand_and_align(int min_w, int max_w, int x, int w,
                                 float ratio, int a) {
  const auto xp = in_range(x, 0, max_w - 1);
  int new_x = 0;
  int new_w = 0;
  std::tie(new_x, new_w) = expand(min_w, max_w, xp, w, ratio);
  int aligned_x = 0;
  int aligned_w = 0;
  std::tie(aligned_x, aligned_w) = align(min_w, max_w, new_x, new_w, a);
  int relative_x = xp - aligned_x;
  /* DLOG(INFO) << "min_w " << min_w << " "           //
             << "max_x " << max_w << " "           //
             << "x " << x << " "                   //
             << "xp " << xp << " "                   //
             << "w " << w << " "                   //
             << "a " << a << " "                   //
             << "new_x " << new_x << " "           //
             << "new_w " << new_w << " "           //
             << "aligned_x " << aligned_x << " "   //
             << "aligned_w " << aligned_w << " "   //
             << "relative_x " << relative_x << " " //
      ;
  */
  // 注意，对齐之后，原来的 w 有可能超过对其之后的边界。如果输入的 w
  // 没有超过对齐之后的边界，？？？？？可能出现吗？
  int relative_w = in_range(w, 0, aligned_w - relative_x);
  return ExpandAndAlignX{aligned_x, aligned_w, relative_x, relative_w};
}

ExpandAndAlign expand_crop(int width, int height, int bbx, int bby, int bbw,
                           int bbh, float ratio_x, float ratio_y, int aligned_x,
                           int aligned_y, int min_w, int min_h) {
  auto result_x = expand_and_align(min_w, width, bbx, bbw, ratio_x, aligned_x);
  auto result_y = expand_and_align(min_h, height, bby, bbh, ratio_y, aligned_y);
  return ExpandAndAlign{result_x.x,          result_y.x,
                        result_x.w,          result_y.w,
                        result_x.relative_x, result_y.relative_x,
                        result_x.relative_w, result_y.relative_w};
}

}  // namespace ai
}  // namespace vitis
