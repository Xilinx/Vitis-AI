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
#ifndef DEEPHI_ANCHORBASE_HPP_
#define DEEPHI_ANCHORBASE_HPP_

#include <memory>
#include <utility>
#include <vector>

namespace vitis {
namespace ai {
namespace dptfssd {

using namespace std;
using VF = std::vector<float>;
using VVF = std::vector<VF>;
using VVVF = std::vector<VVF>;
using VVVVF = std::vector<VVVF>;

class AnchorBase {
 public:
  AnchorBase(int num_layers, int image_width, int image_height);

  const std::vector<std::shared_ptr<std::vector<float>>>& priors() const {
    return priors_;
  }

 protected:
  std::vector<std::shared_ptr<std::vector<float>>> priors_;
  int num_layers_;
  int image_width_;
  int image_height_;

  void tile_anchors(float grid_height, float grid_width, VF& _scales,
                    VF& _aspect_ratios,
                    std::tuple<float, float> base_anchor_size,
                    std::tuple<float, float> anchor_strides,
                    std::tuple<float, float> anchor_offsets);
};

/*
 *  below part is helper function ....
 */
void mywritefile(float* conf, int size1, std::string filename);

std::tuple<VVF, VVF> meshgrid(const VF& v1, const VF& v2);

std::tuple<VVVF, VVVF> meshgrid(const VF& v1, const VVF& vv2);

VVVVF tfstack3(const VVVF& vvv1, const VVVF& vvv2);

VF tfreshape(const VVF& vv1);
VVF tfreshape(const VVVVF& vvvv1);

void tfconcat_decode(const VVF& vv1_center, const VVF& vv2_size,
                     std::vector<std::shared_ptr<std::vector<float>>>& priors_);

void printv(std::string info, const VF& inv);

void printv(std::string info, const VVF& inv);

void printv(std::string info, const VVVF& inv);

void printv(std::string info, const VVVVF& inv);

}  // namespace dptfssd
}  // namespace ai
}  // namespace vitis

#endif
