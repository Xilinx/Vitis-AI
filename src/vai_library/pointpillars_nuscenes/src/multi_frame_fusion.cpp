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

#include "./multi_frame_fusion.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./utils.hpp"

DEF_ENV_PARAM(DEBUG_SWEEPS_FUSION, "0");

namespace vitis {
namespace ai {
namespace pointpillars_nus {

static constexpr int SWEEPS_NUM = 10;

std::shared_ptr<std::vector<float>> points_filter(
    const std::shared_ptr<std::vector<float>>& points, int dim,
    const std::vector<float>& range) {
  auto len = points->size();
  assert(len % dim == 0);
  assert(range.size() >= 6);
  auto num = len / dim;
  std::shared_ptr<std::vector<float>> result{new std::vector<float>(len, 0)};

  auto cnt = 0;
  auto& ps = *points;
  for (auto i = 0u; i < num; ++i) {
    bool remove = false;
    auto index = i * dim;
    auto result_index = cnt * dim;
    if (ps[index] < range[0] || ps[index] > range[3] ||
        ps[index + 1] < range[1] || ps[index + 1] > range[4] ||
        ps[index + 2] < range[2] || ps[index + 2] > range[5]) {
      remove = true;
    }

    if (!remove) {
      std::copy(ps.data() + index, ps.data() + index + dim,
                result->data() + result_index);
      cnt++;
    }
  }
  result->resize(cnt * dim);

  return result;
}

std::vector<float> points_filter(const std::vector<float>& points, int dim,
                                 const std::vector<float>& range) {
  auto len = points.size();
  assert(len % dim == 0);
  assert(range.size() >= 6);
  auto num = len / dim;
  std::vector<float> result(len, 0);

  auto cnt = 0;
  auto& ps = points;
  for (auto i = 0u; i < num; ++i) {
    bool remove = false;
    auto index = i * dim;
    auto result_index = cnt * dim;
    if (ps[index] < range[0] || ps[index] > range[3] ||
        ps[index + 1] < range[1] || ps[index + 1] > range[4] ||
        ps[index + 2] < range[2] || ps[index + 2] > range[5]) {
      remove = true;
    }

    if (!remove) {
      std::copy(ps.data() + index, ps.data() + index + dim,
                result.data() + result_index);
      cnt++;
    }
  }
  result.resize(cnt * dim);

  return result;
}

PointsInfo points_filter(const PointsInfo& points_info,
                         std::vector<float>& range) {
  auto dim = points_info.points.dim;
  assert(range.size() >= 6);
  PointsInfo result{points_info};
  result.points.points = points_filter(points_info.points.points, dim, range);
  return result;
}

PointsInfo remove_useless_dim(const PointsInfo& points_info,
                              int invalid_ch_id) {
  auto dim = points_info.points.dim;
  auto valid_dim = dim - 1;
  auto num = points_info.points.points->size() / dim;
  PointsInfo result{points_info};
  result.points.points.reset(new std::vector<float>(num * valid_dim, 0));
  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
      << "num: " << num << " valid_dim: " << valid_dim;

  auto points_mat =
      cv::Mat(num, dim, CV_32FC1, (float*)points_info.points.points->data());
  auto result_mat =
      cv::Mat(num, valid_dim, CV_32FC1, result.points.points->data());
  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
      << "points mat size: " << points_mat.rows << " * " << points_mat.cols;
  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
      << "result mat size: " << result_mat.rows << " * " << result_mat.cols;
  auto front_rect = cv::Rect(0, 0, invalid_ch_id, num);
  auto points_end_rect =
      cv::Rect(invalid_ch_id + 1, 0, valid_dim - invalid_ch_id, num);
  auto result_end_rect =
      cv::Rect(invalid_ch_id, 0, valid_dim - invalid_ch_id, num);
  points_mat(front_rect).copyTo(result_mat(front_rect));
  result_mat(front_rect) = points_mat(front_rect);
  points_mat(points_end_rect).copyTo(result_mat(result_end_rect));
  // if (ENV_PARAM(DEBUG_SWEEPS_FUSION)) {
  //  debug_mat(points_mat, "points_mat");
  //  debug_mat(result_mat, "result_mat");
  //}
  return result;
}

SweepInfo transform_sweeps2(const SweepInfo& sweep_info, double ts) {
  auto& points_data = *(sweep_info.points.points);
  auto dim = sweep_info.points.dim;
  auto& cam_info = sweep_info.cam_info;

  auto points_num = points_data.size() / dim;

  // dim  =4 set to zero
  // PointsInfo result{points_info.cam_info, points_info.dim - 1,
  //                  std::vector<float>(points_num * result_dim, 0)};
  SweepInfo result{sweep_info};
  result.points.points.reset(new std::vector<float>(points_data.size(), 0));
  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION)) << "points_num : " << points_num;
  auto points_ptr = points_data.data();
  auto result_ptr = result.points.points->data();

  float real_ts = ts - ((double)cam_info.timestamp) / 1000000.0;
  // result[:, 0:3] = points[:, 0:3] * s2l_r.t() + s2l_t.t();
  // result[:, ts_dim] = current_frame_ts - ts

  auto cur_index = 0;
  for (auto i = 0u; i < points_num; ++i) {
    for (auto j = 0u; j < 3; ++j) {
      result_ptr[cur_index + j] +=
          points_ptr[cur_index + 0] * cam_info.s2l_r[j * 3 + 0];
      result_ptr[cur_index + j] +=
          points_ptr[cur_index + 1] * cam_info.s2l_r[j * 3 + 1];
      result_ptr[cur_index + j] +=
          points_ptr[cur_index + 2] * cam_info.s2l_r[j * 3 + 2];
      result_ptr[cur_index + j] += cam_info.s2l_t[j];
    }
    result_ptr[cur_index + 3] = points_ptr[cur_index + 3];
    result_ptr[cur_index + 4] = real_ts;
    cur_index += dim;
  }

  if (dim > 5) {
    for (auto i = 0u; i < points_num; ++i) {
      std::memcpy(result_ptr + i * dim + 5, points_ptr + i * dim + 5,
                  (dim - 5) * sizeof(float));
    }
  }

  return result;
}

SweepInfo transform_sweeps(const SweepInfo& sweep_info, double ts) {
  auto& points_data = *(sweep_info.points.points);
  auto dim = sweep_info.points.dim;
  auto& cam_info = sweep_info.cam_info;

  auto points_num = points_data.size() / dim;

  // dim  =4 set to zero
  // PointsInfo result{points_info.cam_info, points_info.dim - 1,
  //                  std::vector<float>(points_num * result_dim, 0)};
  SweepInfo result{sweep_info};
  result.points.points.reset(new std::vector<float>(points_data.size(), 0));
  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION)) << "points_num : " << points_num;

  auto points_mat =
      cv::Mat(points_num, dim, CV_32FC1, (float*)points_data.data());
  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
      << "points mat size: " << points_mat.rows << " * " << points_mat.cols;
  // if (ENV_PARAM(DEBUG_SWEEPS_FUSION)) {
  //  debug_mat(points_mat, "points_mat");
  //}
  // auto result_mat = cv::Mat(points_num, result_dim, CV_32FC1,
  // result.points.data());
  auto result_mat =
      cv::Mat(points_num, dim, CV_32FC1, result.points.points->data());
  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
      << "result mat size: " << result_mat.rows << " * " << result_mat.cols;
  cv::Rect pt_rect(0, 0, 3, points_mat.rows);
  cv::Rect ts_rect(4, 0, 1, points_mat.rows);
  // 1. result[:, 0:3] = points[:, 0:3] * s2l_r.t() + s2l_t.t();
  auto points_mat_p_rect = points_mat(pt_rect);
  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
      << "points mat p rect size: " << points_mat_p_rect.rows << " * "
      << points_mat_p_rect.cols;
  // if (ENV_PARAM(DEBUG_SWEEPS_FUSION)) {
  //  debug_mat(points_mat_p_rect, "points_mat_p_rect");
  //}
  auto result_mat_p_rect = result_mat(pt_rect);
  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
      << "result mat p rect size: " << result_mat_p_rect.rows << " * "
      << result_mat_p_rect.cols;

  cv::Mat s2l_r(3, 3, CV_32FC1, (float*)cam_info.s2l_r.data());
  cv::Mat s2l_t(3, 1, CV_32FC1, (float*)cam_info.s2l_t.data());
  // if (ENV_PARAM(DEBUG_SWEEPS_FUSION)) {
  //  debug_mat(s2l_r, "s2l_r");
  //  debug_mat(s2l_t, "s2l_t");
  //}
  result_mat_p_rect = points_mat_p_rect * s2l_r.t();
  // if (ENV_PARAM(DEBUG_SWEEPS_FUSION)) {
  //  debug_mat(result_mat_p_rect, "result_mat_p_rect");
  //}

  for (auto i = 0; i < result_mat_p_rect.rows; ++i) {
    for (auto j = 0; j < result_mat_p_rect.cols; ++j) {
      result_mat_p_rect.at<float>(i, j) += s2l_t.at<float>(j, 0);
    }
    result_mat.at<float>(i, 3) = points_mat.at<float>(i, 3);
  }

  // if (ENV_PARAM(DEBUG_SWEEPS_FUSION)) {
  //  debug_mat(result_mat_p_rect, "result_mat_p_rect");
  //}

  // 2. result[:, ts_dim] = current_frame_ts - ts
  auto real_ts = ts - ((double)cam_info.timestamp) / 1000000.0;
  result_mat(ts_rect).setTo(real_ts);
  // if (ENV_PARAM(DEBUG_SWEEPS_FUSION)) {
  //  LOG(INFO) << "real_ts:" << real_ts;
  //  debug_mat(result_mat(ts_rect), "result_mat_ts_rect");
  //}

  // 3. result[:, (one_hot_dims)] = points[:, (one_hot_dims)]

  if (dim > 5) {
    cv::Rect one_hot_rect(5, 0, dim - 5, points_mat.rows);
    auto points_mat_one_hot_rect = points_mat(one_hot_rect);
    auto result_mat_one_hot_rect = result_mat(one_hot_rect);
    LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
        << "points mat one hot rect size: " << points_mat_one_hot_rect.rows
        << " * " << points_mat_one_hot_rect.cols;
    LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
        << "result mat one hot rect size: " << result_mat_one_hot_rect.rows
        << " * " << result_mat_one_hot_rect.cols;

    // if (ENV_PARAM(DEBUG_SWEEPS_FUSION)) {
    //  debug_mat(points_mat_one_hot_rect, "points_mat_one_hot_rect");
    //}

    points_mat_one_hot_rect.copyTo(result_mat_one_hot_rect);
  }
  // if (ENV_PARAM(DEBUG_SWEEPS_FUSION)) {
  //  debug_mat(result_mat, "result_mat");
  //}

  return result;
}

std::vector<float> multi_frame_fusion(
    const PointsInfo& frame_info, const std::vector<SweepInfo>& sweeps_infos) {
  __TIC__(MULTI_FRAME_FUSION)
  auto& points_data = *(frame_info.points.points);
  auto dim = frame_info.points.dim;
  // auto &cam_info = frame_info.cam_info;
  auto timestamp = frame_info.timestamp;
  auto ts_f = ((double)timestamp) / 1000000.0;
  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
      << "points size:" << points_data.size();
  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
      << "points shape:" << points_data.size() / dim << " * " << dim;

  LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION)) << "timestamp: " << timestamp;

  assert(points_data.size() % dim == 0);

  auto frame_points_len = points_data.size();
  auto total = frame_points_len;
  auto sweeps_num = std::min(SWEEPS_NUM, (int)sweeps_infos.size());
  for (auto i = 0; i < sweeps_num; ++i) {
    total += sweeps_infos[i].points.points->size();
  }

  std::vector<float> result(total, 0);

  __TIC__(COPY_CURRENT_FRAME)
  std::copy(points_data.data(), points_data.data() + points_data.size(),
            result.data());
  __TOC__(COPY_CURRENT_FRAME)
  auto all_points_num = total / dim;
  auto result_mat = cv::Mat(all_points_num, dim, CV_32FC1, result.data());

  cv::Rect ts_rect(4, 0, 1, frame_points_len / dim);
  result_mat(ts_rect).setTo(0);

  auto len = frame_points_len;
  for (auto i = 0; i < sweeps_num; ++i) {
    // auto sweep_ts = ((double)sweeps_infos[i].cam_info.timestamp) /
    // 1000000.0; LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
    //    << "sweep[" << i
    //    << "] timestamp: " << sweeps_infos[i].cam_info.timestamp;
    // LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
    //    << "sweep_ts[" << i << "]: " << sweep_ts;
    // LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION)) << "ts_f: " << ts_f;
    // sweep_ts = ts_f - sweep_ts;
    // LOG_IF(INFO, ENV_PARAM(DEBUG_SWEEPS_FUSION))
    //    << "sweep_ts[" << i << "]: " << sweep_ts;
    // auto s = transform_points(sweeps_infos[i], ts_f);
    // auto s = transform_sweeps(sweeps_infos[i], ts_f);
    auto s = transform_sweeps2(sweeps_infos[i], ts_f);

    std::copy(s.points.points->data(),
              s.points.points->data() + s.points.points->size(),
              result.data() + len);
    len = len + sweeps_infos[i].points.points->size();
  }

  __TOC__(MULTI_FRAME_FUSION)

  return result;
}

}  // namespace pointpillars_nus
}  // namespace ai
}  // namespace vitis
