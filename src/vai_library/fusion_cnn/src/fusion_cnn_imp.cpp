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
#include "./fusion_cnn_imp.hpp"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace vitis::ai::fusion_cnn;

namespace vitis {
namespace ai {

DEF_ENV_PARAM(DEBUG_FUSION_CNN, "0");
DEF_ENV_PARAM(DEBUG_FUSION_CNN_MT, "0");

FusionCNNImp::FusionCNNImp(const std::string& model_name,
                           bool need_preprocess) {
  auto attrs = xir::Attrs::create();
  model_ =
      ConfigurableDpuTask::create(model_name, attrs.get(), need_preprocess);
}

FusionCNNImp::FusionCNNImp(const std::string& model_name, xir::Attrs* attrs,
                           bool need_preprocess) {
  model_ = ConfigurableDpuTask::create(model_name, attrs, need_preprocess);
}

FusionCNNImp::~FusionCNNImp() {}

int FusionCNNImp::getInputWidth() const { return model_->getInputWidth(); }

int FusionCNNImp::getInputHeight() const { return model_->getInputHeight(); }

size_t FusionCNNImp::get_input_batch() const {
  return model_->get_input_batch();
}

static std::array<float, 9> rot_mat(float x) {
  return {cos(x), 0, -sin(x), 0, 1, 0, sin(x), 0, cos(x)};
}

static V1F get_box_corners(DetectResult& result_3d) {
  // result_3d.bboxes.bbox:  xyz lhw r
  __TIC__(FUSION_CNN_vector_construct)
  auto count = result_3d.bboxes.size();
  V1F result(count * 24);  // (N, 8, 3)

  std::array<float, 24> corners_norm = {
      -0.5, -1.0, -0.5, -0.5, -1.0, 0.5, -0.5, 0.0, 0.5, -0.5, 0.0, -0.5,
      0.5,  -1.0, -0.5, 0.5,  -1.0, 0.5, 0.5,  0.0, 0.5, 0.5,  0.0, -0.5};

  __TOC__(FUSION_CNN_vector_construct)
  // get corners
  __TIC__(FUSION_CNN_get_corners)
  for (auto i = 0u; i < count; ++i) {
    for (auto j = 0u; j < 8; ++j) {
      for (auto k = 0u; k < 3; ++k) {
        result[i * 24 + j * 3 + k] =
            result_3d.bboxes[i].bbox[k + 3] * corners_norm[j * 3 + k];
      }
    }
  }
  __TOC__(FUSION_CNN_get_corners)

  // rotation_3d_in_axis and add centers
  __TIC__(FUSION_CNN_rotation_3d)
  for (auto i = 0u; i < count; ++i) {
    auto mat = rot_mat(
        result_3d.bboxes[i].bbox[6]);  // generate rotation_mat from theta
    for (auto j = 0u; j < 8; ++j) {
      std::array<float, 3> tmp{result[i * 24 + j * 3],
                               result[i * 24 + j * 3 + 1],
                               result[i * 24 + j * 3 + 2]};
      for (auto k = 0u; k < 3; ++k) {
        result[i * 24 + j * 3 + k] = result_3d.bboxes[i].bbox[k];  // + centers
        for (auto l = 0u; l < 3; ++l) {
          result[i * 24 + j * 3 + k] += tmp[l] * mat[l * 3 + k];  // rotation
        }
      }
    }
  }
  __TOC__(FUSION_CNN_rotation_3d)

  return result;
}

static V1F get_8point_corners_in_image(V1F& corners, const V2F& p2) {
  auto count = corners.size() / 24;
  V1F result(count * 16);  // (N, 8, 2)

  // get corners, normalize and only get x y
  for (auto i = 0u; i < count; ++i) {
    for (auto j = 0u; j < 8; ++j) {
      float z = 0.0;
      for (auto k = 0u; k < 3; ++k) {    // only need x, y, z
        for (auto l = 0u; l < 3; ++l) {  // corners[i][j][4] = 0, so (l < 3)
          if (k < 2) {                   //  get x, y
            // p2 not transport, so multiple p2[k][l], not p2[l][k]
            result[i * 16 + j * 2 + k] +=
                corners[i * 24 + j * 3 + l] * p2[k][l];
          }
          if (k == 2) z += corners[i * 24 + j * 3 + l] * p2[k][l];  // get z
        }
      }
      result[i * 16 + j * 2 + 0] /= z;
      result[i * 16 + j * 2 + 1] /= z;
    }
  }
  return result;
}

static V1F get_4point_corners_in_image(V1F& corners_in_image, int img_width,
                                       int img_height) {
  auto count = corners_in_image.size() / 16;
  V1F result(count * 4);  // (N, 4)
  for (auto i = 0u; i < count; ++i) {
    float minx = std::numeric_limits<float>::max();
    float miny = std::numeric_limits<float>::max();
    float maxx = 0.0f;
    float maxy = 0.0f;
    for (auto j = 0u; j < 8; ++j) {
      minx = std::min(minx, corners_in_image[i * 16 + j * 2 + 0]);
      maxx = std::max(maxx, corners_in_image[i * 16 + j * 2 + 0]);
      miny = std::min(miny, corners_in_image[i * 16 + j * 2 + 1]);
      maxy = std::max(maxy, corners_in_image[i * 16 + j * 2 + 1]);
    }
    result[i * 4] = std::min(std::max(minx, 0.0f), (float)img_width);
    result[i * 4 + 1] = std::min(std::max(miny, 0.0f), (float)img_height);
    result[i * 4 + 2] = std::min(std::max(maxx, 0.0f), (float)img_width);
    result[i * 4 + 3] = std::min(std::max(maxy, 0.0f), (float)img_height);
  }

  return result;
}

static void box_lidar_to_camera(DetectResult& detect_result_3d, const V2F& rect,
                                const V2F& trv2c) {
  CHECK_EQ(rect.size(), 4);
  CHECK_EQ(trv2c.size(), 4);
  V2F matrics(4, V1F(4, 0.0));
  for (auto i = 0u; i < 4; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      for (auto k = 0u; k < 4; ++k) {
        matrics[i][j] += rect[i][k] * trv2c[k][j];
      }
    }
  }

  V1F multi_result(detect_result_3d.bboxes.size() * 3);
  for (auto i = 0u; i < detect_result_3d.bboxes.size(); ++i) {
    for (auto j = 0u; j < 3; ++j) {
      multi_result[i * 3 + j] = matrics[j][3];
      for (auto k = 0u; k < 3; ++k) {
        multi_result[i * 3 + j] +=
            detect_result_3d.bboxes[i].bbox[k] * matrics[j][k];
      }
    }
  }

  // (x, y, z, w, l, h, r) -> (x', y', z', l, h, w, r)
  for (auto i = 0u; i < detect_result_3d.bboxes.size(); ++i) {
    detect_result_3d.bboxes[i].bbox[0] = multi_result[i * 3 + 0];
    detect_result_3d.bboxes[i].bbox[1] = multi_result[i * 3 + 1];
    detect_result_3d.bboxes[i].bbox[2] = multi_result[i * 3 + 2];

    auto w = detect_result_3d.bboxes[i].bbox[3];
    detect_result_3d.bboxes[i].bbox[3] = detect_result_3d.bboxes[i].bbox[4];
    detect_result_3d.bboxes[i].bbox[4] = detect_result_3d.bboxes[i].bbox[5];
    detect_result_3d.bboxes[i].bbox[5] = w;
  }
}

DetectResult get_detect_result_2d_from_3d(DetectResult& detect_result_3d,
                                          const FusionParam& fusion_param) {
  __TIC__(FUSION_CNN_box_lidar_to_camera)
  box_lidar_to_camera(detect_result_3d, fusion_param.rect, fusion_param.trv2c);
  __TOC__(FUSION_CNN_box_lidar_to_camera)
  __TIC__(FUSION_CNN_get_box_corners)
  auto corners = get_box_corners(detect_result_3d);
  __TOC__(FUSION_CNN_get_box_corners)
  __TIC__(FUSION_CNN_get_8point_corners_in_image)
  auto corners_in_image = get_8point_corners_in_image(corners, fusion_param.p2);
  __TOC__(FUSION_CNN_get_8point_corners_in_image)
  __TIC__(FUSION_CNN_get_4point_corners_in_image)
  auto corners_4point_in_image = get_4point_corners_in_image(
      corners_in_image, fusion_param.img_width, fusion_param.img_height);
  __TOC__(FUSION_CNN_get_4point_corners_in_image)

  CHECK_EQ(corners_4point_in_image.size() / 4, detect_result_3d.bboxes.size());
  __TIC__(FUSION_CNN_transform_result)
  DetectResult result;
  for (auto i = 0u; i < detect_result_3d.bboxes.size(); ++i) {
    result.bboxes.emplace_back(
        DetectResult::BoundingBox{detect_result_3d.bboxes[i].score,
                                  {
                                      corners_4point_in_image[4 * i],
                                      corners_4point_in_image[4 * i + 1],
                                      corners_4point_in_image[4 * i + 2],
                                      corners_4point_in_image[4 * i + 3],
                                  }});
  }
  __TOC__(FUSION_CNN_transform_result)
  return result;
}

static V1F get_norm(const DetectResult& detect_result_3d) {
  V1F result;
  std::transform(detect_result_3d.bboxes.begin(), detect_result_3d.bboxes.end(),
                 std::back_inserter(result), [](const auto& box) -> float {
                   return sqrt(pow(box.bbox[0], 2) + pow(box.bbox[1], 2)) /
                          82.0;
                 });
  return result;
}
std::tuple<V1F, V1I> get_overlaps_and_indexes_one_channel(
    unsigned int k, unsigned int num_2d, unsigned int num_3d,
    const DetectResult& detect_result_2d,
    const DetectResult& detect_result_2d_from_3d, const V1F& dis_to_lidar) {
  V1F overlaps;
  V1I indexes;
  auto qbox_area = ((detect_result_2d.bboxes[k].bbox[2] -
                     detect_result_2d.bboxes[k].bbox[0]) *
                    (detect_result_2d.bboxes[k].bbox[3] -
                     detect_result_2d.bboxes[k].bbox[1]));
  for (auto n = 0u; n < num_3d; ++n) {
    // if (count >= MAX_NUM_OVERLAP) {
    //   break;
    // }
    auto iw = (std::min(detect_result_2d_from_3d.bboxes[n].bbox[2],
                        detect_result_2d.bboxes[k].bbox[2]) -
               std::max(detect_result_2d_from_3d.bboxes[n].bbox[0],
                        detect_result_2d.bboxes[k].bbox[0]));
    if (iw > 0) {
      auto ih = (std::min(detect_result_2d_from_3d.bboxes[n].bbox[3],
                          detect_result_2d.bboxes[k].bbox[3]) -
                 std::max(detect_result_2d_from_3d.bboxes[n].bbox[1],
                          detect_result_2d.bboxes[k].bbox[1]));
      if (ih > 0) {
        auto ua = ((detect_result_2d_from_3d.bboxes[n].bbox[2] -
                    detect_result_2d_from_3d.bboxes[n].bbox[0]) *
                   (detect_result_2d_from_3d.bboxes[n].bbox[3] -
                    detect_result_2d_from_3d.bboxes[n].bbox[1])) +
                  qbox_area - iw * ih;
        overlaps.emplace_back() = iw * ih / ua;
        overlaps.emplace_back() = detect_result_2d_from_3d.bboxes[n].score;
        overlaps.emplace_back() = detect_result_2d.bboxes[k].score;
        overlaps.emplace_back() = dis_to_lidar[n];
        indexes.emplace_back() = (int)k;
        indexes.emplace_back() = (int)n;
      } else if (k == num_2d - 1) {
        overlaps.emplace_back() = -10;
        overlaps.emplace_back() = detect_result_2d_from_3d.bboxes[n].score;
        overlaps.emplace_back() = -10;
        overlaps.emplace_back() = dis_to_lidar[n];
        indexes.emplace_back() = (int)k;
        indexes.emplace_back() = (int)n;
      }
    } else if (k == num_2d - 1) {
      overlaps.emplace_back() = -10;
      overlaps.emplace_back() = detect_result_2d_from_3d.bboxes[n].score;
      overlaps.emplace_back() = -10;
      overlaps.emplace_back() = dis_to_lidar[n];
      indexes.emplace_back() = (int)k;
      indexes.emplace_back() = (int)n;
    }
  }

  return {std::move(overlaps), std::move(indexes)};
}

std::tuple<V1F, V1I, unsigned int> get_overlaps_and_indexes_async(
    const DetectResult& detect_result_2d, DetectResult& detect_result_3d,
    const FusionParam& fusion_param) {
  constexpr auto MAX_NUM_OVERLAP = 800000u;
  V1F overlaps(MAX_NUM_OVERLAP / 4 * 4);
  V1I indexes(MAX_NUM_OVERLAP / 4 * 2);

  auto num_2d = detect_result_2d.bboxes.size();
  __TIC__(FUSION_CNN_get_norm)
  auto dis_to_lidar = get_norm(detect_result_3d);
  __TOC__(FUSION_CNN_get_norm)
  __TIC__(FUSION_CNN_get_detect_result_2d_from_3d)
  auto detect_result_2d_from_3d =
      get_detect_result_2d_from_3d(detect_result_3d, fusion_param);
  __TOC__(FUSION_CNN_get_detect_result_2d_from_3d)
  auto num_3d = detect_result_2d_from_3d.bboxes.size();

  __TIC__(FUSION_CNN_calculate_overlaps_and_indexes)
  auto count = 0u;
  std::vector<std::future<std::tuple<V1F, V1I>>> futures(num_2d);

  for (auto k = 0u; k < num_2d; ++k) {
    futures[k] =
        std::async(get_overlaps_and_indexes_one_channel, k, num_2d, num_3d,
                   std::ref(detect_result_2d),
                   std::ref(detect_result_2d_from_3d), std::ref(dis_to_lidar));
  }

  for (auto j = 0u; j < futures.size(); ++j) {
    auto [o, i] = futures[j].get();
    if (count + i.size() / 2 >= MAX_NUM_OVERLAP) {
      break;
    }

    if (indexes.size() < MAX_NUM_OVERLAP * 2 &&
        count * 2 + i.size() >= indexes.size()) {
      overlaps.resize(overlaps.size() * 2);
      indexes.resize(indexes.size() * 2);
    }
    std::copy(o.begin(), o.end(), overlaps.begin() + count * 4);
    std::copy(i.begin(), i.end(), indexes.begin() + count * 2);
    count += i.size() / 2;
  }
  __TOC__(FUSION_CNN_calculate_overlaps_and_indexes)

  return {std::move(overlaps), std::move(indexes), count};
}

std::tuple<V1F, V1I, unsigned int> get_overlaps_and_indexes(
    const DetectResult& detect_result_2d, DetectResult& detect_result_3d,
    const FusionParam& fusion_param) {
  constexpr auto MAX_NUM_OVERLAP = 800000u;
  V1F overlaps(MAX_NUM_OVERLAP / 2 * 4);
  V1I indexes(MAX_NUM_OVERLAP / 2 * 2);

  auto num_2d = detect_result_2d.bboxes.size();
  __TIC__(FUSION_CNN_get_norm)
  auto dis_to_lidar = get_norm(detect_result_3d);
  __TOC__(FUSION_CNN_get_norm)
  __TIC__(FUSION_CNN_get_detect_result_2d_from_3d)
  auto detect_result_2d_from_3d =
      get_detect_result_2d_from_3d(detect_result_3d, fusion_param);
  __TOC__(FUSION_CNN_get_detect_result_2d_from_3d)
  auto num_3d = detect_result_2d_from_3d.bboxes.size();

  __TIC__(FUSION_CNN_calculate_overlaps_and_indexes)
  auto count = 0u;
  bool not_resize = true;
  for (auto k = 0u; k < num_2d; ++k) {
    if (count >= MAX_NUM_OVERLAP) {
      break;
    }

    auto qbox_area = ((detect_result_2d.bboxes[k].bbox[2] -
                       detect_result_2d.bboxes[k].bbox[0]) *
                      (detect_result_2d.bboxes[k].bbox[3] -
                       detect_result_2d.bboxes[k].bbox[1]));
    for (auto n = 0u; n < num_3d; ++n) {
      if (count >= MAX_NUM_OVERLAP) {
        break;
      }
      if (not_resize) {
        if (count >= MAX_NUM_OVERLAP / 2 - 2) {
          overlaps.resize(MAX_NUM_OVERLAP * 4);
          indexes.resize(MAX_NUM_OVERLAP * 4);
          not_resize = false;
        }
      }
      auto iw = (std::min(detect_result_2d_from_3d.bboxes[n].bbox[2],
                          detect_result_2d.bboxes[k].bbox[2]) -
                 std::max(detect_result_2d_from_3d.bboxes[n].bbox[0],
                          detect_result_2d.bboxes[k].bbox[0]));
      if (iw > 0) {
        auto ih = (std::min(detect_result_2d_from_3d.bboxes[n].bbox[3],
                            detect_result_2d.bboxes[k].bbox[3]) -
                   std::max(detect_result_2d_from_3d.bboxes[n].bbox[1],
                            detect_result_2d.bboxes[k].bbox[1]));
        if (ih > 0) {
          auto ua = ((detect_result_2d_from_3d.bboxes[n].bbox[2] -
                      detect_result_2d_from_3d.bboxes[n].bbox[0]) *
                     (detect_result_2d_from_3d.bboxes[n].bbox[3] -
                      detect_result_2d_from_3d.bboxes[n].bbox[1])) +
                    qbox_area - iw * ih;
          overlaps[4 * count] = iw * ih / ua;
          overlaps[4 * count + 1] = detect_result_2d_from_3d.bboxes[n].score;
          overlaps[4 * count + 2] = detect_result_2d.bboxes[k].score;
          overlaps[4 * count + 3] = dis_to_lidar[n];
          indexes[2 * count] = (int)k;
          indexes[2 * count + 1] = (int)n;
          count++;
        } else if (k == num_2d - 1) {
          overlaps[4 * count] = -10;
          overlaps[4 * count + 1] = detect_result_2d_from_3d.bboxes[n].score;
          overlaps[4 * count + 2] = -10;
          overlaps[4 * count + 3] = dis_to_lidar[n];
          indexes[2 * count] = (int)k;
          indexes[2 * count + 1] = (int)n;
          count++;
        }
      } else if (k == num_2d - 1) {
        overlaps[4 * count] = -10;
        overlaps[4 * count + 1] = detect_result_2d_from_3d.bboxes[n].score;
        overlaps[4 * count + 2] = -10;
        overlaps[4 * count + 3] = dis_to_lidar[n];
        indexes[2 * count] = (int)k;
        indexes[2 * count + 1] = (int)n;
        count++;
      }
    }
  }
  __TOC__(FUSION_CNN_calculate_overlaps_and_indexes)
  return {std::move(overlaps), std::move(indexes), count};
}

static void print_result(DetectResult& result) {
  std::cout << "DetectResult: " << std::endl;
  for (auto& box : result.bboxes) {
    std::cout << "score: " << box.score << "\t bbox: ";
    for (auto& coordinate : box.bbox) {
      std::cout << coordinate << " ";
    }
    std::cout << std::endl;
  }
}

std::tuple<V1I, unsigned int> FusionCNNImp::preprocess(
    const DetectResult& result_2d, DetectResult& result_3d,
    const FusionParam& fusion_param, int batch_idx) {
  __TIC__(FUSION_CNN_get_overlaps_and_indexes)
  auto [overlaps, indexes, count] =
      get_overlaps_and_indexes(result_2d, result_3d, fusion_param);
  __TOC__(FUSION_CNN_get_overlaps_and_indexes)

  auto model_input_size =
      model_->getInputTensor()[0][0].size / get_input_batch();
  auto input_ptr = (int8_t*)model_->getInputTensor()[0][0].get_data(batch_idx);
  __TIC__(FUSION_CNN_memset)
  std::memset(input_ptr, 0, model_input_size);
  __TOC__(FUSION_CNN_memset)
  auto input_tensor_scale =
      vitis::ai::library::tensor_scale(model_->getInputTensor()[0][0]);
  LOG_IF(INFO, ENV_PARAM(DEBUG_FUSION_CNN))
      << "model input size: " << model_input_size
      << "\t input scale: " << input_tensor_scale;

  __TIC__(FUSION_CNN_set_input)
  std::transform(overlaps.begin(), overlaps.begin() + 4 * count, input_ptr,
                 [input_tensor_scale](auto& overlap) {
                   return (int8_t)round((overlap * input_tensor_scale));
                 });

  __TOC__(FUSION_CNN_set_input)

  return {std::move(indexes), count};
}

void FusionCNNImp::postprocess(const V1I& indexes, const unsigned int count,
                               DetectResult& result_3d, int batch_idx) {
  auto output_ptr =
      (int8_t*)model_->getOutputTensor()[0][0].get_data(batch_idx);
  auto model_out_size =
      model_->getOutputTensor()[0][0].size / get_input_batch();
  auto out_tensor_scale =
      vitis::ai::library::tensor_scale(model_->getOutputTensor()[0][0]);
  LOG_IF(INFO, ENV_PARAM(DEBUG_FUSION_CNN))
      << "model out size: " << model_out_size
      << "\t out scale: " << out_tensor_scale;

  // get output and maxpool
  V1F max_pool_result(result_3d.bboxes.size(),
                      std::numeric_limits<float>::lowest());
  for (auto i = 0u; i < count; ++i) {
    max_pool_result[indexes[2 * i + 1]] =
        std::max(max_pool_result[indexes[2 * i + 1]],
                 ((float)(*(output_ptr + i)) * out_tensor_scale));
  }

  // update box_3d_scores
  CHECK_EQ(result_3d.bboxes.size(), max_pool_result.size());
  for (auto i = 0u; i < result_3d.bboxes.size(); ++i) {
    result_3d.bboxes[i].score = max_pool_result[i];
  }
}

DetectResult FusionCNNImp::run(const DetectResult& detect_result_2d,
                               DetectResult& detect_result_3d,
                               const FusionParam& fusion_param) {
  // auto result_2d = get_result_2d();
  // auto result_3d = get_result_3d();
  __TIC__(FUSION_CNN_E2E)
  __TIC__(FUSION_CNN_PREPROCESS)

  DetectResult result(detect_result_3d);

  if (ENV_PARAM(DEBUG_FUSION_CNN)) {
    LOG(INFO) << "result: ";
    print_result(result);
  }

  auto [indexes, count] =
      preprocess(detect_result_2d, detect_result_3d, fusion_param, 0);
  __TOC__(FUSION_CNN_PREPROCESS)

  __TIC__(FUSION_CNN_DPU)
  model_->run(0);
  __TOC__(FUSION_CNN_DPU)

  __TIC__(FUSION_CNN_POSTPROCESS)
  postprocess(indexes, count, result, 0);
  __TOC__(FUSION_CNN_POSTPROCESS)
  __TOC__(FUSION_CNN_E2E)

  if (ENV_PARAM(DEBUG_FUSION_CNN)) {
    LOG(INFO) << "update result: ";
    print_result(result);
  }
  return result;
}

std::vector<DetectResult> FusionCNNImp::run(
    const std::vector<DetectResult>& batch_detect_result_2d,
    std::vector<DetectResult>& batch_detect_result_3d,
    const std::vector<FusionParam>& batch_fusion_param) {
  auto num = std::min({batch_detect_result_2d.size(),
                       batch_detect_result_3d.size(), get_input_batch()});

  std::vector<DetectResult> batch_result(batch_detect_result_3d);
  if (ENV_PARAM(DEBUG_FUSION_CNN)) {
    LOG(INFO) << "batch: " << num;
    LOG(INFO) << "results: ";
    for (auto& r : batch_result) {
      print_result(r);
    }
  }

  __TIC__(FUSION_CNN_BATCH_PREPROCESS)
  std::vector<V1I> batch_indexes(num);
  std::vector<unsigned int> batch_count(num);
  if (ENV_PARAM(DEBUG_FUSION_CNN_MT)) {
    std::vector<std::future<std::tuple<V1I, unsigned int>>> pre_futures;
    pre_futures.reserve(num);
    for (auto i = 0u; i < num; ++i) {
      pre_futures.emplace_back(std::async(std::launch::async,
                                          &FusionCNNImp::preprocess, this,
                                          std::ref(batch_detect_result_2d[i]),
                                          std::ref(batch_detect_result_3d[i]),
                                          std::ref(batch_fusion_param[i]), i));
    }
    for (auto i = 0u; i < num; ++i) {
      std::tie(batch_indexes[i], batch_count[i]) = pre_futures[i].get();
    }
  } else {
    for (auto i = 0u; i < num; ++i) {
      std::tie(batch_indexes[i], batch_count[i]) =
          preprocess(batch_detect_result_2d[i], batch_detect_result_3d[i],
                     batch_fusion_param[i], i);
    }
  }
  __TOC__(FUSION_CNN_BATCH_PREPROCESS)

  __TIC__(FUSION_CNN_BATCH_DPU)
  model_->run(0);
  __TOC__(FUSION_CNN_BATCH_DPU)

  __TIC__(FUSION_CNN_BATCH_POSTPROCESS)
  if (ENV_PARAM(DEBUG_FUSION_CNN_MT)) {
    std::vector<std::future<void>> post_futures;
    post_futures.reserve(num);
    for (auto i = 0u; i < num; ++i) {
      post_futures.emplace_back(
          std::async(std::launch::async, &FusionCNNImp::postprocess, this,
                     std::ref((batch_indexes)[i]), std::ref((batch_count)[i]),
                     std::ref((batch_result)[i]), i));
    }
    for (auto& f : post_futures) {
      f.get();
    }
  } else {
    for (auto i = 0u; i < num; ++i) {
      postprocess(batch_indexes[i], batch_count[i], batch_result[i], i);
    }
  }
  __TOC__(FUSION_CNN_BATCH_POSTPROCESS)

  if (ENV_PARAM(DEBUG_FUSION_CNN)) {
    LOG(INFO) << "update results: ";
    for (auto& r : batch_result) {
      print_result(r);
    }
  }

  return batch_result;
}  // namespace ai

}  // namespace ai
}  // namespace vitis
