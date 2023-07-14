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
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/fusion_cnn.hpp>
#include <vitis/ai/profiling.hpp>
#include "../src/utils.hpp"
#include "vitis/ai/clocs.hpp"

using namespace vitis::ai;
using namespace vitis::ai::clocs;

DEF_ENV_PARAM(DEBUG_CLOCS_SELECT_RESULT, "0");
static void read_points_file(const std::string& points_file_name,
                             std::vector<float>& points) {
  struct stat file_stat;
  if (stat(points_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << points_file_name << " state error!" << std::endl;
    exit(-1);
  }
  auto file_size = file_stat.st_size;
  LOG(INFO) << "input file:" << points_file_name << " size:" << file_size;
  // points_info.points.resize(file_size / 4);
  points.resize(file_size / 4);
  // CHECK(std::ifstream(points_file_name).read(reinterpret_cast<char
  // *>(points_info.points.data()), file_size).good());
  CHECK(std::ifstream(points_file_name)
            .read(reinterpret_cast<char*>(points.data()), file_size)
            .good());
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    // std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    std::cout << "usage:" << argv[0] << " [idx]" << std::endl;
    exit(0);
  }

  auto idx = std::atoi(argv[5]);

  if (idx >= 10) {
    std::cerr << "invalid idx:" << idx << std::endl;
    exit(0);
  }

  ClocsResult result;
  std::cout << "load selected:" << std::endl;
  auto selected_size = 0u;
  vector<vector<float>> selected_bboxes;
  vector<float> selected_scores;
  vector<float> selected_labels;
  vector<float> bboxes_buf;
  read_points_file("selected_bbox.bin", bboxes_buf);
  read_points_file("selected_score.bin", selected_scores);
  selected_size = selected_scores.size();
  LOG(INFO) << "selected_size:" << selected_size;
  selected_labels.assign(selected_size, 1);
  selected_bboxes.resize(selected_size);
  for (auto i = 0u; i < selected_size; ++i) {
    LOG(INFO) << "idx:" << i;
    std::cout << "label:" << selected_labels[i];
    std::cout << ", bbox:";
    selected_bboxes[i].assign(bboxes_buf.begin() + i * 7,
                              bboxes_buf.begin() + i * 7 + 7);
    for (auto j = 0u; j < 7; ++j) {
      std::cout << selected_bboxes[i][j] << " ";
    }
    std::cout << ", score:" << selected_scores[i];
    std::cout << std::endl;
  }

  auto selected_bboxes_for_nms = transform_for_nms(selected_bboxes);

  // 4. nms
  int pre_max_size = 1000;
  int post_max_size = 300;
  vector<ScoreIndex> ordered(selected_size);  // label, score, idx
  for (auto i = 0u; i < selected_size; ++i) {
    ordered[i] =
        ScoreIndex{selected_scores[i], (int)selected_labels[i], (int)i};
  }

  size_t k = topk(ordered, pre_max_size);
  std::stable_sort(ordered.begin(), ordered.end(), ScoreIndex::compare);
  vector<vector<float>> ordered_bboxes(k);
  vector<vector<float>> ordered_bboxes_for_nms(k, vector<float>(4));
  vector<float> ordered_scores(k);
  vector<int> ordered_labels(k);
  for (auto i = 0u; i < k; ++i) {
    auto idx = ordered[i].index;
    std::cout << "top K: idx: " << idx;
    std::cout << ", bbox:";
    ordered_bboxes[i] = selected_bboxes[i];
    ordered_bboxes_for_nms[i].assign(
        selected_bboxes_for_nms.data() + i * 4,
        selected_bboxes_for_nms.data() + i * 4 + 4);
    ordered_scores[i] = selected_scores[idx];
    ordered_labels[i] = (int)selected_labels[idx];
    std::cout << ", score:" << ordered_scores[i];
    std::cout << ", label:" << ordered_labels[i];
    std::cout << std::endl;
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_SELECT_RESULT))
      << "ordered boxes_for_nms size: " << ordered_bboxes_for_nms.size();
  float iou_thresh = 0.5;
  // std::vector<size_t> nms_result;
  // applyNMS(boxes_for_nms, score_for_nms, iou_thresh, score_thresh,
  // nms_result);
  auto nms_result =
      non_max_suppression_cpu(ordered_bboxes_for_nms, ordered_scores,
                              pre_max_size, post_max_size, iou_thresh);

  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS_SELECT_RESULT))
      << "nms result size: " << nms_result.size();
  // 5. make final result
  result.bboxes.resize(nms_result.size());
  for (auto i = 0u; i < nms_result.size(); ++i) {
    auto idx = nms_result[i];
    result.bboxes[i] =
        ClocsResult::PPBbox{ordered_scores[idx], ordered_bboxes[idx],
                            (uint32_t)ordered_labels[idx]};
  }

  auto& ret = result;
  auto size = ret.bboxes.size();
  for (auto i = 0u; i < size; ++i) {
    auto r = ret.bboxes[i];
    std::cout << "label:" << r.label << " ";
    std::cout << "bbox:"
              << " ";
    for (auto j = 0u; j < r.bbox.size(); ++j) {
      std::cout << r.bbox[j] << " ";
    }
    std::cout << "score:" << r.score;
    std::cout << std::endl;
  }

  return 0;
}

