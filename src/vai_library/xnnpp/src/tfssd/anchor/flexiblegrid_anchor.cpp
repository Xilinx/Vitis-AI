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
#include "./flexiblegrid_anchor.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace vitis {
namespace ai {
namespace dptfssd {

using namespace std;

using std::copy_n;
using std::fill_n;
using std::make_pair;
using std::make_shared;
using std::sqrt;

FlexibleGridAnchor::FlexibleGridAnchor(int num_layers, int image_width,
                                       int image_height)
    : AnchorBase(num_layers, image_width, image_height) {}

}  // namespace dptfssd
}  // namespace ai
}  // namespace vitis
