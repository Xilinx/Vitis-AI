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

#include "vitis/ai/facerecog.hpp"
#include "./facerecog_imp.hpp"

/// Function to get a instance of class. Please delete the pointer if
/// you will not use it any more.\n
/// @param mean Mean value, a vector of float has 3 elements
namespace vitis {
namespace ai {
std::unique_ptr<FaceRecog> FaceRecog::create(const std::string &feature_model_name, bool need_preprocess) {
    return std::unique_ptr<FaceRecog>(new FaceRecogImp(feature_model_name, need_preprocess));
}

std::unique_ptr<FaceRecog> FaceRecog::create(const std::string &feature_model_name,
                                             xir::Attrs *attrs,
                                             bool need_preprocess) {
    return std::unique_ptr<FaceRecog>(new FaceRecogImp(feature_model_name,
                                                       attrs,
                                                       need_preprocess));
}
std::unique_ptr<FaceRecog> FaceRecog::create(const std::string &landmark_model_name, 
                                             const std::string &feature_model_name, 
                                             bool need_preprocess) {
    return std::unique_ptr<FaceRecog>(new FaceRecogImp(landmark_model_name, feature_model_name, need_preprocess));
}

std::unique_ptr<FaceRecog> FaceRecog::create(const std::string &landmark_model_name, 
                                             const std::string &feature_model_name, 
                                             xir::Attrs *attrs,
                                             bool need_preprocess) {
    return std::unique_ptr<FaceRecog>(new FaceRecogImp(landmark_model_name, 
                                                       feature_model_name, 
                                                       attrs,
                                                       need_preprocess));
}
FaceRecog::FaceRecog()
{
}
FaceRecog::~FaceRecog()
{
}

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

static std::pair<int,int> expand(int total, int x, int w, float ratio) {
  const int d = w * ratio;
  const int x1 = x - d;
  const int x2 = x + w + d;
  const int x1_c = in_range(x1, 0, total);
  const int x2_c = in_range(x2, 0, total);
  return std::make_pair(x1_c, x2_c - x1);
}

static std::pair<int, int> align(int total, int x, int w, int a) {
  // 如果总宽度不是对齐像素的整数倍，最多 a-1 个像素会被丢弃掉
  const int total_in_a = div_floor(total, a);
  // 对齐的时候，左边尽量多扩一些
  const int x1_in_a = div_floor(x, a);
  // 对齐的时候，右边也尽量多扩一些。
  const int x2_in_a = div_ceil(x + w, a);
  const int x1_c_in_a = in_range(x1_in_a, 0, total_in_a);
  const int x2_c_in_a = in_range(x2_in_a, 0, total_in_a);
  const int aligned_x = x1_c_in_a * a;
  const int aligned_w = (x2_c_in_a - x1_c_in_a) * a;
  return std::make_pair(aligned_x, aligned_w);
}

static std::tuple<int, int, int, int> expand_and_align_1(int total, int x,
                                                         int w, float ratio,
                                                         int a) {
  int new_x = 0;
  int new_w = 0;
  std::tie(new_x, new_w) = expand(total, x, w, ratio);
  int aligned_x = 0;
  int aligned_w = 0;
  std::tie(aligned_x, aligned_w) = align(total, new_x, new_w, a);
  int relative_x = x - aligned_x;
  // 注意，对其之后，原来的 w 有可能超过对其之后的边界。如果输入的 w
  // 没有超过对齐之后的边界，relative_w 应该不变。
  int relative_w = in_range(w, 0, aligned_w);
  return std::make_tuple(aligned_x, aligned_w, relative_x, relative_w);
}

std::pair<cv::Rect, cv::Rect>
FaceRecog::expand_and_align(int width, int height, int bbx, int bby, int bbw, int bbh,
                        float ratio_x, float ratio_y, int aligned_x, int aligned_y) {

  // Expanded rect in original image
  std::tuple<int, int, int, int> expanded;

  // Cropped rect in expanded image
  std::tuple<int, int, int, int> relative;

  std::tie(std::get<0>(expanded), std::get<2>(expanded), std::get<0>(relative),
           std::get<2>(relative)) =
      expand_and_align_1(width, bbx, bbw, ratio_x, aligned_x);
  std::tie(std::get<1>(expanded), std::get<3>(expanded), std::get<1>(relative),
           std::get<3>(relative)) =
      expand_and_align_1(height, bby, bbh, ratio_y, aligned_y);
  return std::make_pair(cv::Rect{std::get<0>(expanded), std::get<1>(expanded),
                                 std::get<2>(expanded), std::get<3>(expanded)},
                        cv::Rect{std::get<0>(relative), std::get<1>(relative),
                                 std::get<2>(relative), std::get<3>(relative)});
}


}}
