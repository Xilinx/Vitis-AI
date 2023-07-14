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
#ifndef EXPAND_AND_ALIGN_HPP_
#define EXPAND_AND_ALIGN_HPP_

namespace vitis {
namespace ai {
struct ExpandAndAlign {
  int x;
  int y;
  int w;
  int h;
  int relative_x;
  int relative_y;
  int relative_w;
  int relative_h;
};

struct ExpandAndAlignX {
  int x;
  int w;
  int relative_x;
  int relative_w;
};

ExpandAndAlign expand_crop(int width, int height, int bbx, int bby, int bbw,
                           int bbh, float ratio_x, float ratio_y, int aligned_x,
                           int align_y, int min_w = 0, int min_h = 0);

ExpandAndAlignX expand_and_align(int min_w, int max_w, int x, int w,
                                 float ratio, int a);
}  // namespace ai
}  // namespace vitis

#endif
