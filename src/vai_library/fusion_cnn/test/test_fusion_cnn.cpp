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

#include <sys/stat.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/fusion_cnn.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
using namespace vitis::ai;
using namespace vitis::ai::fusion_cnn;

DEF_ENV_PARAM(TEST_FUSION_CNN, "0");

static V1F read_from_bin_file(std::string file) {
  auto result_size = std::filesystem::file_size(file);
  V1F result(result_size / 4);
  CHECK(std::ifstream(file).read((char*)&result[0], result_size).good())
      << "failed to read baseline from " << file;
  return result;
}

static DetectResult get_result_2d() {
  auto result_2d_detector = read_from_bin_file("box_2d_detector.bin");
  auto result_2d_scores = read_from_bin_file("box_2d_scores.bin");
  DetectResult result;
  auto result_2d_detector_len = result_2d_detector.size() / 4;
  auto result_2d_scores_len = result_2d_scores.size();
  CHECK_EQ(result_2d_detector_len, result_2d_scores_len);

  for (auto i = 0u; i < result_2d_scores_len; ++i) {
    result.bboxes.emplace_back(DetectResult::BoundingBox{
        result_2d_scores[i],
        {result_2d_detector[4 * i], result_2d_detector[4 * i + 1],
         result_2d_detector[4 * i + 2], result_2d_detector[4 * i + 3]}});
  }

  for (auto i = 0u; i < result.bboxes.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(TEST_FUSION_CNN))
        << "score: " << result.bboxes[i].score
        << "\t box: " << result.bboxes[i].bbox[0] << ", "
        << result.bboxes[i].bbox[1] << ", " << result.bboxes[i].bbox[2] << ", "
        << result.bboxes[i].bbox[3];
  }

  return result;
}

static DetectResult get_result_3d() {
  auto result_3d_detector = read_from_bin_file("box_3d_detector.bin");
  auto result_3d_scores = read_from_bin_file("box_3d_scores.bin");
  DetectResult result;
  auto result_3d_detector_len = result_3d_detector.size() / 7;
  auto result_3d_scores_len = result_3d_scores.size();
  CHECK_EQ(result_3d_detector_len, result_3d_scores_len);

  for (auto i = 0u; i < result_3d_scores_len; ++i) {
    result.bboxes.emplace_back(DetectResult::BoundingBox{
        result_3d_scores[i],
        {result_3d_detector[7 * i], result_3d_detector[7 * i + 1],
         result_3d_detector[7 * i + 2], result_3d_detector[7 * i + 3],
         result_3d_detector[7 * i + 4], result_3d_detector[7 * i + 5],
         result_3d_detector[7 * i + 6]}});
  }

  for (auto i = 0u; i < result.bboxes.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(TEST_FUSION_CNN))
        << "score: " << result.bboxes[i].score
        << "\t box: " << result.bboxes[i].bbox[0] << ", "
        << result.bboxes[i].bbox[0] << ", " << result.bboxes[i].bbox[1] << ", "
        << result.bboxes[i].bbox[2] << ", " << result.bboxes[i].bbox[3] << ", "
        << result.bboxes[i].bbox[4] << ", " << result.bboxes[i].bbox[5] << ", "
        << result.bboxes[i].bbox[6];
  }

  return result;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage :" << argv[0] << " <model_name>" << std::endl;
    abort();
  }
  string kernel = argv[1];

  auto det = vitis::ai::FusionCNN::create(kernel);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  // vitis::ai::fusion_cnn::DetectResult detect_result_2d;
  // vitis::ai::fusion_cnn::DetectResult detect_result_3d;
  vitis::ai::fusion_cnn::FusionParam fusion_param{
      {{721.5377197265625, 0.0, 609.559326171875, 44.85728073120117},
       {0.0, 721.5377197265625, 172.85400390625, 0.2163791060447693},
       {0.0, 0.0, 1.0, 0.0027458840049803257},
       {0.0, 0.0, 0.0, 1.0}},
      {{0.9999238848686218, 0.009837759658694267, -0.007445048075169325, 0.0},
       {-0.00986979529261589, 0.9999421238899231, -0.004278459120541811, 0.0},
       {0.007402527146041393, 0.0043516140431165695, 0.999963104724884, 0.0},
       {0.0, 0.0, 0.0, 1.0}},
      {{0.0075337449088692665, -0.9999713897705078, -0.00061660201754421,
        -0.004069766029715538},
       {0.01480249036103487, 0.0007280732970684767, -0.9998902082443237,
        -0.07631617784500122},
       {0.9998620748519897, 0.007523790001869202, 0.014807550236582756,
        -0.2717806100845337},
       {0.0, 0.0, 0.0, 1.0}},
      1242,
      375};

  auto result_2d = get_result_2d();
  auto result_3d = get_result_3d();
  auto res = det->run(result_2d, result_3d, fusion_param);

  return 0;
}
