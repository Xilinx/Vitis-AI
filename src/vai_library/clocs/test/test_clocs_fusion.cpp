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
#include <vitis/ai/profiling.hpp>
#include "../src/clocs_fusion.hpp"

DEF_ENV_PARAM(DEBUG_CLOCS, "0");

using namespace std;
using namespace vitis::ai;
using namespace vitis::ai::clocs;

static std::vector<float> read_from_bin_file(std::string file) {
  auto result_size = std::filesystem::file_size(file);
  std::vector<float> result(result_size / 4);
  CHECK(std::ifstream(file).read((char*)&result[0], result_size).good())
      << "failed to read baseline from " << file;
  return result;
}

static clocs::DetectResult get_result_2d() {
  auto result_2d_detector = read_from_bin_file("box_2d_detector.bin");
  auto result_2d_scores = read_from_bin_file("box_2d_scores.bin");
  clocs::DetectResult result;
  auto result_2d_detector_len = result_2d_detector.size() / 4;
  auto result_2d_scores_len = result_2d_scores.size();
  CHECK_EQ(result_2d_detector_len, result_2d_scores_len);

  for (auto i = 0lu; i < result_2d_scores_len; ++i) {
    result.bboxes.emplace_back(vitis::ai::clocs::DetectResult::BoundingBox{
        result_2d_scores[i],
        {result_2d_detector[4 * i], result_2d_detector[4 * i + 1],
         result_2d_detector[4 * i + 2], result_2d_detector[4 * i + 3]}});
  }

  for (auto i = 0lu; i < result.bboxes.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS))
        << "score: " << result.bboxes[i].score
        << "\t box: " << result.bboxes[i].bbox[0] << ", "
        << result.bboxes[i].bbox[1] << ", " << result.bboxes[i].bbox[2] << ", "
        << result.bboxes[i].bbox[3];
  }

  return result;
}

static clocs::DetectResult get_result_3d() {
  auto result_3d_detector = read_from_bin_file("box_3d_detector.bin");
  auto result_3d_scores = read_from_bin_file("box_3d_scores.bin");
  clocs::DetectResult result;
  auto result_3d_detector_len = result_3d_detector.size() / 7;
  auto result_3d_scores_len = result_3d_scores.size();
  CHECK_EQ(result_3d_detector_len, result_3d_scores_len);

  for (auto i = 0lu; i < result_3d_scores_len; ++i) {
    result.bboxes.emplace_back(vitis::ai::clocs::DetectResult::BoundingBox{
        result_3d_scores[i],
        {result_3d_detector[7 * i], result_3d_detector[7 * i + 1],
         result_3d_detector[7 * i + 2], result_3d_detector[7 * i + 3],
         result_3d_detector[7 * i + 4], result_3d_detector[7 * i + 5],
         result_3d_detector[7 * i + 6]}});
  }

  for (auto i = 0lu; i < result.bboxes.size(); ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS))
        << "score: " << result.bboxes[i].score
        << "\t box: " << result.bboxes[i].bbox[0] << ", "
        << result.bboxes[i].bbox[0] << ", " << result.bboxes[i].bbox[1] << ", "
        << result.bboxes[i].bbox[2] << ", " << result.bboxes[i].bbox[3] << ", "
        << result.bboxes[i].bbox[4] << ", " << result.bboxes[i].bbox[5] << ", "
        << result.bboxes[i].bbox[6];
  }

  return result;
}

void print_result(vitis::ai::clocs::DetectResult& result) {
  for (auto i = 0u; i < result.bboxes.size(); ++i) {
    std::cout << "score: " << result.bboxes[i].score << "\t bbox: ";
    for (auto j = 0u; j < result.bboxes[i].bbox.size(); ++j) {
      std::cout << result.bboxes[i].bbox[j] << " ";
    }
    std::cout << std::endl;
  }
}
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage :" << argv[0] << " <model_name>" << std::endl;
    abort();
  }
  string kernel = argv[1];
  auto det = vitis::ai::ClocsFusion::create(kernel);
  vitis::ai::clocs::DetectResult detect_result_2d = get_result_2d();
  vitis::ai::clocs::DetectResult detect_result_3d = get_result_3d();

  auto res = det->run(detect_result_2d, detect_result_3d);
  print_result(res);

  return 0;
}
