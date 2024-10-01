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
#include "./clocs_imp.hpp"
#include <sys/stat.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/nnpp/apply_nms.hpp>
#include <vitis/ai/profiling.hpp>
#include "./utils.hpp"

using namespace std;
using namespace vitis::ai::clocs;

namespace vitis {
namespace ai {

DEF_ENV_PARAM(DEBUG_CLOCS, "0");
DEF_ENV_PARAM(DEBUG_CLOCS_DUMP, "0");
DEF_ENV_PARAM(DEBUG_CLOCS_PRINT, "0");
DEF_ENV_PARAM(DEBUG_CLOCS_MT, "0");

ClocsImp::ClocsImp(const std::string& yolo, const std::string& pointpillars_0,
                   const std::string& pointpillars_1,
                   const std::string& fusionnet, bool need_preprocess) {
  auto attrs = xir::Attrs::create();
  yolo_ = vitis::ai::YOLOvX::create(yolo, attrs.get(), need_preprocess);
  // yolo_ = vitis::ai::YOLOvX::create(yolo, need_preprocess);
  pointpillars_ = vitis::ai::ClocsPointPillars::create(
      pointpillars_0, pointpillars_1, attrs.get(), need_preprocess);
  fusionnet_ =
      vitis::ai::FusionCNN::create(fusionnet, attrs.get(), need_preprocess);
}

ClocsImp::~ClocsImp() {}

int ClocsImp::getInputWidth() const { return yolo_->getInputWidth(); }

int ClocsImp::getInputHeight() const { return yolo_->getInputHeight(); }

size_t ClocsImp::get_input_batch() const { return yolo_->get_input_batch(); }

int ClocsImp::getPointsDim() const { return pointpillars_->getPointsDim(); }

void ClocsImp::run_yolo(ClocsImp* instance,
                        const std::vector<cv::Mat>& batch_input,
                        int batch_size) {
  // assert(batch_size <= batch_input.size())
  instance->batch_yolo_results_ = instance->yolo_->run(batch_input);
}

void ClocsImp::run_pointpillars(
    ClocsImp* instance, const std::vector<std::vector<float>>& batch_input,
    int batch_size) {
  // assert(batch_size <= batch_input.size())
  instance->batch_pp_results_ = instance->pointpillars_->run(batch_input);
}

void ClocsImp::run_transform(
    ClocsImp* instance, const std::vector<clocs::ClocsInfo>& batch_inputs,
    std::vector<fusion_cnn::DetectResult>& batch_detect_2d_result,
    std::vector<fusion_cnn::DetectResult>& batch_detect_3d_result,
    std::vector<fusion_cnn::FusionParam>& batch_fusion_params, int batch_idx) {
  __TIC__(CLOCS_TRANSFORM_2d)
  batch_detect_2d_result[batch_idx].bboxes.resize(
      instance->batch_yolo_results_[batch_idx].bboxes.size());
  for (auto j = 0u; j < batch_detect_2d_result[batch_idx].bboxes.size(); ++j) {
    batch_detect_2d_result[batch_idx].bboxes[j] = {
        instance->batch_yolo_results_[batch_idx].bboxes[j].score,
        instance->batch_yolo_results_[batch_idx].bboxes[j].box};
  }
  __TOC__(CLOCS_TRANSFORM_2d)
  __TIC__(CLOCS_TRANSFORM_3d)
  batch_detect_3d_result[batch_idx].bboxes.resize(
      instance->batch_pp_results_[batch_idx].bboxes.size());
  for (auto j = 0u; j < batch_detect_3d_result[batch_idx].bboxes.size(); ++j) {
    batch_detect_3d_result[batch_idx].bboxes[j] = {
        instance->batch_pp_results_[batch_idx].bboxes[j].score,
        instance->batch_pp_results_[batch_idx].bboxes[j].bbox};
  }
  __TOC__(CLOCS_TRANSFORM_3d)
  __TIC__(CLOCS_TRANSFORM_PARAM)
  vector<vector<float>> p2(4, vector<float>(4));
  vector<vector<float>> trv2c(4, vector<float>(4));
  vector<vector<float>> rect(4, vector<float>(4));
  for (auto j = 0u; j < 4; ++j) {
    p2[j].assign(batch_inputs[batch_idx].calib_P2.data() + 4 * j,
                 batch_inputs[batch_idx].calib_P2.data() + 4 * j + 4);
    trv2c[j].assign(batch_inputs[batch_idx].calib_Trv2c.data() + 4 * j,
                    batch_inputs[batch_idx].calib_Trv2c.data() + 4 * j + 4);
    rect[j].assign(batch_inputs[batch_idx].calib_rect.data() + 4 * j,
                   batch_inputs[batch_idx].calib_rect.data() + 4 * j + 4);
  }
  batch_fusion_params[batch_idx] = fusion_cnn::FusionParam{
      p2, rect, trv2c, batch_inputs[batch_idx].image.cols,
      batch_inputs[batch_idx].image.rows};
  __TOC__(CLOCS_TRANSFORM_PARAM)
}

void ClocsImp::run_transform_with2d(
    ClocsImp* instance,
    const std::vector<std::vector<float>>& batch_detect2d_ori_result,
    const std::vector<clocs::ClocsInfo>& batch_inputs,
    std::vector<fusion_cnn::DetectResult>& batch_detect_2d_result,
    std::vector<fusion_cnn::DetectResult>& batch_detect_3d_result,
    std::vector<fusion_cnn::FusionParam>& batch_fusion_params, int batch_idx) {
  __TIC__(CLOCS_TRANSFORM_2d)
  batch_detect_2d_result[batch_idx].bboxes.resize(
      // instance->batch_yolo_results_[batch_idx].bboxes.size());
      batch_detect2d_ori_result[batch_idx].size() / 5);
  for (auto j = 0u; j < batch_detect_2d_result[batch_idx].bboxes.size(); ++j) {
    batch_detect_2d_result[batch_idx].bboxes[j].bbox =
        vector<float>(batch_detect2d_ori_result[batch_idx].begin() + j * 5,
                      batch_detect2d_ori_result[batch_idx].begin() + j * 5 + 4);

    batch_detect_2d_result[batch_idx].bboxes[j].score =
        batch_detect2d_ori_result[batch_idx][j * 5 + 4];
  }
  __TOC__(CLOCS_TRANSFORM_2d)
  __TIC__(CLOCS_TRANSFORM_3d)
  batch_detect_3d_result[batch_idx].bboxes.resize(
      instance->batch_pp_results_[batch_idx].bboxes.size());
  for (auto j = 0u; j < batch_detect_3d_result[batch_idx].bboxes.size(); ++j) {
    batch_detect_3d_result[batch_idx].bboxes[j] = {
        instance->batch_pp_results_[batch_idx].bboxes[j].score,
        instance->batch_pp_results_[batch_idx].bboxes[j].bbox};
  }
  __TOC__(CLOCS_TRANSFORM_3d)
  __TIC__(CLOCS_TRANSFORM_PARAM)
  vector<vector<float>> p2(4, vector<float>(4));
  vector<vector<float>> trv2c(4, vector<float>(4));
  vector<vector<float>> rect(4, vector<float>(4));
  for (auto j = 0u; j < 4; ++j) {
    p2[j].assign(batch_inputs[batch_idx].calib_P2.data() + 4 * j,
                 batch_inputs[batch_idx].calib_P2.data() + 4 * j + 4);
    trv2c[j].assign(batch_inputs[batch_idx].calib_Trv2c.data() + 4 * j,
                    batch_inputs[batch_idx].calib_Trv2c.data() + 4 * j + 4);
    rect[j].assign(batch_inputs[batch_idx].calib_rect.data() + 4 * j,
                   batch_inputs[batch_idx].calib_rect.data() + 4 * j + 4);
  }
  batch_fusion_params[batch_idx] = fusion_cnn::FusionParam{
      p2, rect, trv2c, batch_inputs[batch_idx].image.cols,
      batch_inputs[batch_idx].image.rows};
  __TOC__(CLOCS_TRANSFORM_PARAM)
}

std::vector<ClocsResult> ClocsImp::run_clocs(
    const std::vector<vector<float>>& batch_detect2d_result,
    const std::vector<ClocsInfo>& batch_inputs, size_t num) {
  __TIC__(CLOCS_E2E)
  std::vector<std::vector<float>> batch_points(num);
  for (auto i = 0u; i < num; ++i) {
    batch_points[i] = batch_inputs[i].points;
  }
  if (ENV_PARAM(DEBUG_CLOCS_MT)) {
    __TIC__(CLOCS_MT)
    pointpillars_->setMultiThread(true);
    std::thread th_pp(&run_pointpillars, this, batch_points, num);
    th_pp.join();
    __TOC__(CLOCS_MT)
  } else {
    // 2. run pointpillars
    __TIC__(CLOCS_POINTPILLARS)
    batch_pp_results_ = pointpillars_->run(batch_points);
    __TOC__(CLOCS_POINTPILLARS)
  }

  if (ENV_PARAM(DEBUG_CLOCS_PRINT)) {
    auto size = batch_pp_results_[0].bboxes.size();
    auto& ret = batch_pp_results_[0];
    std::cout << "print 3d result" << std::endl;
    for (auto i = 0u; i < size; ++i) {
      std::cout << "label:" << ret.bboxes[i].label << " ";
      std::cout << "bbox:"
                << " ";
      for (auto j = 0u; j < ret.bboxes[i].bbox.size(); ++j) {
        std::cout << ret.bboxes[i].bbox[j] << " ";
      }
      std::cout << "score:" << ret.bboxes[i].score;
      std::cout << std::endl;
    }
  }
  // 3. transform 2d and 3d result
  std::vector<fusion_cnn::DetectResult> batch_detect_2d_result(num);
  std::vector<fusion_cnn::DetectResult> batch_detect_3d_result(num);
  std::vector<fusion_cnn::FusionParam> batch_fusion_params(num);

  __TIC__(CLOCS_TRANSFORM)
  if (ENV_PARAM(DEBUG_CLOCS_MT)) {
    std::vector<std::thread> th_trans;
    for (auto i = 0u; i < num; ++i) {
      th_trans.push_back(std::thread(
          &run_transform_with2d, this, std::cref(batch_detect2d_result),
          std::cref(batch_inputs), std::ref(batch_detect_2d_result),
          std::ref(batch_detect_3d_result), std::ref(batch_fusion_params), i));
    }
    for (auto i = 0u; i < num; ++i) {
      th_trans[i].join();
    }
  } else {
    for (auto i = 0u; i < num; ++i) {
      run_transform_with2d(this, batch_detect2d_result, batch_inputs,
                           batch_detect_2d_result, batch_detect_3d_result,
                           batch_fusion_params, i);
    }
  }
  __TOC__(CLOCS_TRANSFORM)

  // 4. fusion
  __TIC__(CLOCS_FUSION)
  auto fusion_results = fusionnet_->run(
      batch_detect_2d_result, batch_detect_3d_result, batch_fusion_params);
  __TOC__(CLOCS_FUSION)

  if (ENV_PARAM(DEBUG_CLOCS_PRINT)) {
    auto size = batch_pp_results_[0].bboxes.size();
    auto& ret = batch_pp_results_[0];
    std::cout << "print 3d result" << std::endl;
    for (auto i = 0u; i < size; ++i) {
      std::cout << "label:" << ret.bboxes[i].label << " ";
      std::cout << "bbox:"
                << " ";
      for (auto j = 0u; j < ret.bboxes[i].bbox.size(); ++j) {
        std::cout << ret.bboxes[i].bbox[j] << " ";
      }
      std::cout << "score:" << ret.bboxes[i].score;
      std::cout << std::endl;
    }
  }

  // 5. postprocess
  // std::vector<ClocsResult> results(num);
  __TIC__(CLOCS_POSTPROCESS)
  auto results = postprocess(fusion_results, batch_pp_results_, num);
  __TOC__(CLOCS_POSTPROCESS)
  __TOC__(CLOCS_E2E)
  return results;
}

std::vector<ClocsResult> ClocsImp::run_internal(
    const std::vector<ClocsInfo>& batch_inputs) {
  __TIC__(CLOCS_E2E)
  size_t batch = get_input_batch();
  auto num = std::min(batch, batch_inputs.size());

  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS)) << "batch:" << batch;
  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS)) << "num:" << num;
  // 0. prepare input
  std::vector<cv::Mat> batch_images(num);
  for (auto i = 0u; i < num; ++i) {
    batch_images[i] = batch_inputs[i].image.clone();

    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS))
        << "batch_images[" << i << "]:" << batch_images[i].rows << ", "
        << batch_images[i].cols;
  }

  std::vector<std::vector<float>> batch_points(num);
  for (auto i = 0u; i < num; ++i) {
    batch_points[i] = batch_inputs[i].points;
  }

  // std::vector<YOLOvXResult> batch_yolo_results_;
  // std::vector<ClocsPointPillarsResult> batch_pp_results_;
  // if (num > 1 && ENV_PARAM(DEBUG_CLOCS_MT)) {
  if (ENV_PARAM(DEBUG_CLOCS_MT)) {
    __TIC__(CLOCS_MT)
    pointpillars_->setMultiThread(true);
    std::thread th_yolo(&run_yolo, this, batch_images, num);
    std::thread th_pp(&run_pointpillars, this, batch_points, num);
    th_yolo.join();
    th_pp.join();
    __TOC__(CLOCS_MT)
  } else {
    // 1. run yolo
    __TIC__(CLOCS_YOLO)
    batch_yolo_results_ = yolo_->run(batch_images);
    __TOC__(CLOCS_YOLO)
    // 2. run pointpillars
    __TIC__(CLOCS_POINTPILLARS)
    batch_pp_results_ = pointpillars_->run(batch_points);
    __TOC__(CLOCS_POINTPILLARS)
  }

  __TIC__(CLOCS_TRANSFORM)
  // 3. transform 2d and 3d result
  std::vector<fusion_cnn::DetectResult> batch_detect_2d_result(num);
  std::vector<fusion_cnn::DetectResult> batch_detect_3d_result(num);
  std::vector<fusion_cnn::FusionParam> batch_fusion_params(num);

  if (ENV_PARAM(DEBUG_CLOCS_MT)) {
    std::vector<std::thread> th_trans;
    for (auto i = 0u; i < num; ++i) {
      th_trans.push_back(std::thread(
          &run_transform, this, std::cref(batch_inputs),
          std::ref(batch_detect_2d_result), std::ref(batch_detect_3d_result),
          std::ref(batch_fusion_params), i));
    }
    for (auto i = 0u; i < num; ++i) {
      th_trans[i].join();
    }
  } else {
    for (auto i = 0u; i < num; ++i) {
      run_transform(this, batch_inputs, batch_detect_2d_result,
                    batch_detect_3d_result, batch_fusion_params, i);
    }
    __TIC__(CLOCS_TRANSFORM_2d)
    for (auto i = 0u; i < num; ++i) {
      batch_detect_2d_result[i].bboxes.resize(
          batch_yolo_results_[i].bboxes.size());
      for (auto j = 0u; j < batch_detect_2d_result[i].bboxes.size(); ++j) {
        batch_detect_2d_result[i].bboxes[j] = {
            batch_yolo_results_[i].bboxes[j].score,
            batch_yolo_results_[i].bboxes[j].box};
      }
    }
    __TOC__(CLOCS_TRANSFORM_2d)

    __TIC__(CLOCS_TRANSFORM_3d)
    for (auto i = 0u; i < num; ++i) {
      // batch_detect_3d_result[i].bboxes.resize(batch_pp_results_[i].bboxes.size());
      // for (auto j = 0u; j < batch_detect_3d_result[i].bboxes.size(); ++j) {
      //  batch_detect_3d_result[i].bboxes[j] =
      //  {batch_pp_results_[i].scores[j],
      //                                         batch_pp_results_[i].bboxes[j]};
      //}
      batch_detect_3d_result[i].bboxes.resize(
          batch_pp_results_[i].bboxes.size());
      for (auto j = 0u; j < batch_detect_3d_result[i].bboxes.size(); ++j) {
        batch_detect_3d_result[i].bboxes[j] = {
            batch_pp_results_[i].bboxes[j].score,
            batch_pp_results_[i].bboxes[j].bbox};
      }
    }
    __TOC__(CLOCS_TRANSFORM_3d)
    __TIC__(CLOCS_TRANSFORM_PARAM)
    for (auto i = 0u; i < num; ++i) {
      vector<vector<float>> p2(4, vector<float>(4));
      vector<vector<float>> trv2c(4, vector<float>(4));
      vector<vector<float>> rect(4, vector<float>(4));
      for (auto j = 0u; j < 4; ++j) {
        p2[j].assign(batch_inputs[i].calib_P2.data() + 4 * j,
                     batch_inputs[i].calib_P2.data() + 4 * j + 4);
        trv2c[j].assign(batch_inputs[i].calib_Trv2c.data() + 4 * j,
                        batch_inputs[i].calib_Trv2c.data() + 4 * j + 4);
        rect[j].assign(batch_inputs[i].calib_rect.data() + 4 * j,
                       batch_inputs[i].calib_rect.data() + 4 * j + 4);
      }
      batch_fusion_params[i] = fusion_cnn::FusionParam{
          p2, rect, trv2c, batch_images[i].cols, batch_images[i].rows};
    }
    __TOC__(CLOCS_TRANSFORM_PARAM)
  }
  __TOC__(CLOCS_TRANSFORM)

  // 4. fusion
  __TIC__(CLOCS_FUSION)
  auto fusion_results = fusionnet_->run(
      batch_detect_2d_result, batch_detect_3d_result, batch_fusion_params);
  __TOC__(CLOCS_FUSION)
  // 5. postprocess
  // std::vector<ClocsResult> results(num);
  __TIC__(CLOCS_POSTPROCESS)
  auto results = postprocess(fusion_results, batch_pp_results_, num);
  __TOC__(CLOCS_POSTPROCESS)
  __TOC__(CLOCS_E2E)
  return results;
}

std::vector<ClocsResult> ClocsImp::run_internal(
    const std::vector<std::vector<float>>& batch_detect2d_result,
    const std::vector<ClocsInfo>& batch_inputs) {
  size_t batch = get_input_batch();
  auto num = std::min(batch, batch_inputs.size());
  return run_clocs(batch_detect2d_result, batch_inputs, num);
}

ClocsResult ClocsImp::postprocess_kernel(
    fusion_cnn::DetectResult& fusion_result,
    ClocsPointPillarsResult& pp_results, size_t batch_idx) {
  ClocsResult result;
  int size = fusion_result.bboxes.size();

  // 1. score sigmoid
  std::vector<float> scores(size);
  for (auto i = 0; i < size; ++i) {
    scores[i] = fusion_result.bboxes[i].score;
  }
  sigmoid_n(scores);
  if (ENV_PARAM(DEBUG_CLOCS_PRINT)) {
    std::cout << "print sigmoid result" << std::endl;
    for (auto i = 0u; i < scores.size(); ++i) {
      std::cout << scores[i] << std::endl;
    }
  }

  // 2. select scores over thresh
  float score_thresh = 0.05;
  std::vector<int> selected;
  for (auto i = 0; i < size; ++i) {
    if (scores[i] >= score_thresh) {
      selected.push_back(i);
    }
  }

  vector<vector<float>> selected_bboxes(selected.size());
  vector<float> selected_scores(selected.size());
  vector<float> selected_labels(selected.size());
  for (auto i = 0u; i < selected.size(); ++i) {
    auto idx = selected[i];
    selected_bboxes[i] = pp_results.bboxes[idx].bbox;
    selected_scores[i] = scores[idx];
    selected_labels[i] = pp_results.bboxes[idx].label;
  }
  if (ENV_PARAM(DEBUG_CLOCS)) {
    std::cout << "selected:" << std::endl;
    for (auto i = 0u; i < selected.size(); ++i) {
      std::cout << "label:" << selected_labels[i];
      std::cout << ", bbox:";
      for (auto j = 0u; j < 7; ++j) {
        std::cout << selected_bboxes[i][j] << " ";
      }
      std::cout << ", score:" << selected_scores[i];
      std::cout << std::endl;
    }
  }

  auto selected_size = selected.size();
  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS)) << "selected size:" << selected.size();
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
    ordered_bboxes[i] = selected_bboxes[idx];
    ordered_bboxes_for_nms[i].assign(
        selected_bboxes_for_nms.data() + idx * 4,
        selected_bboxes_for_nms.data() + idx * 4 + 4);
    ordered_scores[i] = selected_scores[idx];
    ordered_labels[i] = (int)selected_labels[idx];
    if (ENV_PARAM(DEBUG_CLOCS)) {
      std::cout << "top K: idx: " << idx;
      std::cout << ", bbox:";
      std::cout << ", score:" << ordered_scores[i];
      std::cout << ", label:" << ordered_labels[i];
      std::cout << std::endl;
    }
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS))
      << "ordered boxes_for_nms size: " << ordered_bboxes_for_nms.size();
  float iou_thresh = 0.5;
  auto nms_result =
      non_max_suppression_cpu(ordered_bboxes_for_nms, ordered_scores,
                              pre_max_size, post_max_size, iou_thresh);

  LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS))
      << "nms result size: " << nms_result.size();

  // 5. make final result
  result.bboxes.resize(nms_result.size());
  for (auto i = 0u; i < nms_result.size(); ++i) {
    auto idx = nms_result[i];
    bool opp_lables =
        (ordered_bboxes[idx][6] > 0) ^ ((bool)(ordered_labels[idx]));
    LOG_IF(INFO, ENV_PARAM(DEBUG_CLOCS)) << "opp_lables: " << opp_lables;
    ordered_bboxes[idx][6] += ((int)opp_lables) * clocs::PI;
    uint32_t final_label = 0u;
    result.bboxes[i] = ClocsResult::PPBbox{ordered_scores[idx],
                                           ordered_bboxes[idx], final_label};
  }
  return result;
}

std::vector<ClocsResult> ClocsImp::postprocess(
    std::vector<fusion_cnn::DetectResult>& batch_fusion_results,
    std::vector<ClocsPointPillarsResult>& batch_pp_results, size_t batch_size) {
  std::vector<ClocsResult> results(batch_size);
  for (auto i = 0u; i < batch_size; ++i) {
    results[i] =
        postprocess_kernel(batch_fusion_results[i], batch_pp_results[i], i);
  }
  return results;
}

ClocsResult ClocsImp::run(const ClocsInfo& input) {
  std::vector<ClocsInfo> batch_inputs(1, input);
  return (this->run_internal(batch_inputs))[0];
}

ClocsResult ClocsImp::run(const std::vector<float>& detect2d_result,
                          const ClocsInfo& input) {
  std::vector<ClocsInfo> batch_inputs(1, input);
  std::vector<std::vector<float>> batch_detect2d_result(1, detect2d_result);
  return (this->run_internal(batch_detect2d_result, batch_inputs))[0];
}

std::vector<ClocsResult> ClocsImp::run(
    const std::vector<ClocsInfo>& batch_inputs) {
  return this->run_internal(batch_inputs);
}

std::vector<ClocsResult> ClocsImp::run(
    const std::vector<std::vector<float>>& batch_detect2d_result,
    const std::vector<ClocsInfo>& batch_inputs) {
  return this->run_internal(batch_detect2d_result, batch_inputs);
}

}  // namespace ai
}  // namespace vitis

