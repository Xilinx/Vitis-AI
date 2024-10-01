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

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
namespace {

template <class T>
std::vector<T> vchar_to_vT(const std::vector<char>& vc) {
  std::vector<T> value(vc.size() / sizeof(T));
  memcpy(value.data(), vc.data(), vc.size());
  return value;
}

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    start_position_ = vchar_to_vT<float>(
        op->get_attr<std::vector<char>>("bev_start_position"));
    resolution_ =
        vchar_to_vT<float>(op->get_attr<std::vector<char>>("bev_resolution"));
    geom_sub_.clear();
    for (size_t i = 0; i < start_position_.size(); i++) {
      geom_sub_.push_back(start_position_[i] - resolution_[i] / 2.0);
    }
  }
  int calculate(vart::simple_tensor_buffer_t<float> output,
                std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
    CHECK_EQ(3, inputs[1].tensor->get_shape()[3])
        << "inputs[1].tensor->get_shape()[3] mast eq 3";
    auto& x = inputs[0];
    auto& g_data = inputs[1].data;
    auto Nprime = x.tensor->get_shape()[2];
    auto output_shape = output.tensor->get_shape();
    auto B = output_shape[0];
    auto H = output_shape[1];
    auto W = output_shape[2];
    auto C = output_shape[3];

    memset(output.data, 0, output.tensor->get_data_size());
    for (auto i = 0; i < Nprime; i++) {
      int w = (g_data[i * 3] - geom_sub_[0]) / resolution_[0];
      int h = (g_data[i * 3 + 1] - geom_sub_[1]) / resolution_[1];
      int b = (g_data[i * 3 + 2] - geom_sub_[2]) / resolution_[2];
      if (w >= 0 && w < W && h >= 0 && h < H && b >= 0 && b < B) {
        auto o_data = output.data + (((b * H) + h) * W + w) * C;
        auto x_data = x.data + i * C;
        std::transform(x_data, x_data + C, o_data, o_data, std::plus<float>{});
      }
    }
    return 0;
  }

 private:
  std::vector<float> start_position_;
  std::vector<float> resolution_;
  std::vector<float> geom_sub_;
};

}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
