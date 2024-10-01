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
#include <limits>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "vitis/ai/xmodel_postprocessor.hpp"

namespace {

class MyPostProcessor {
 public:
  static xir::OpDef get_op_def() {
    return xir::OpDef("open_pose")  //
        .add_input_arg(xir::OpArgDef{"L1", xir::OpArgDef::REQUIRED,
                                     xir::DataType::Type::FLOAT,
                                     "pose detection"})
        .add_input_arg(xir::OpArgDef{"L2", xir::OpArgDef::REQUIRED,
                                     xir::DataType::Type::FLOAT,
                                     "pose detection"})
        .set_annotation("postprocessor for open pose.");
  }

  explicit MyPostProcessor(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) {
    auto input_shape = args.graph_input_tensor->get_shape();
    CHECK_EQ(input_shape.size(), 4u);
    height_ = input_shape[1];
    width_ = input_shape[2];
  }

  vitis::ai::proto::DpuModelResult process(
      const vart::simple_tensor_buffer_t<float>& L1,
      const vart::simple_tensor_buffer_t<float>& L2);

 private:
  int width_;
  int height_;
};

using Peak = std::tuple<int, float, cv::Point2f>;
using Peaks = std::vector<Peak>;
using AllPeaks = std::vector<Peaks>;
using Candidate = std::tuple<int, int, float, float>;
using Connection = std::tuple<int, int, float, int, int>;
using AllConnection = std::vector<Connection>;

static std::vector<std::vector<int>> limbSeq = {
    {0, 1}, {1, 2}, {2, 3},  {3, 4},  {1, 5},   {5, 6},  {6, 7},
    {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13}};
static std::vector<std::vector<int>> mapIdx = {
    {15, 16}, {17, 18}, {19, 20}, {21, 22}, {23, 24}, {25, 26}, {27, 28},
    {29, 30}, {31, 32}, {33, 34}, {35, 36}, {37, 38}, {39, 40}};
bool isThreeInConnection(const std::vector<Connection>& connections, int index) {
  for (size_t i = 0; i < connections.size(); ++i) {
    if (index == std::get<3>(connections[i])) return true;
  }
  return false;
}

bool isFourInConnection(const std::vector<Connection>& connections, int index) {
  for (size_t i = 0; i < connections.size(); ++i) {
    if (index == std::get<4>(connections[i])) return true;
  }
  return false;
}

void find_peak(cv::Mat ori_img, Peaks& peaks, int& idx) {
  cv::Mat gas_img;
  GaussianBlur(ori_img, gas_img, cv::Size(3, 3), 3);
  for (int x = 1; x < gas_img.cols - 1; ++x)
    for (int y = 1; y < gas_img.rows - 1; ++y) {
      {
        if (gas_img.at<float>(y, x) <= 0.1) continue;
        if (gas_img.at<float>(y, x) >= gas_img.at<float>(y, x - 1) &&
            gas_img.at<float>(y, x) >= gas_img.at<float>(y - 1, x) &&
            gas_img.at<float>(y, x) >= gas_img.at<float>(y, x + 1) &&
            gas_img.at<float>(y, x) >= gas_img.at<float>(y + 1, x)) {
          peaks.emplace_back(++idx, ori_img.at<float>(y, x), cv::Point(x, y));
        }
      }
    }
}

void findLines(int width, const std::vector<cv::Mat>& pafs,
               const AllPeaks& all_peaks, std::vector<AllConnection>& connection_all,
               std::vector<int>& special_k) {
  std::vector<Connection> connection;
  int mid_num = 10;
  for (size_t k = 0; k < mapIdx.size(); ++k) {
    cv::Mat score_midx = pafs[mapIdx[k][0] - 15];
    cv::Mat score_midy = pafs[mapIdx[k][1] - 15];
    Peaks candA = all_peaks[limbSeq[k][0]];
    Peaks candB = all_peaks[limbSeq[k][1]];
    size_t nA = candA.size();
    size_t nB = candB.size();
    std::vector<float> vec;
    vec.reserve(2);
    if (!candA.empty() && !candB.empty()) {
      std::vector<Candidate> connection_candidate;
      for (size_t i = 0; i < candA.size(); ++i) {
        for (size_t j = 0; j < candB.size(); ++j) {
          vec[0] = std::get<2>(candA[i]).x - std::get<2>(candB[j]).x;
          vec[1] = std::get<2>(candA[i]).y - std::get<2>(candB[j]).y;
          float norm = sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
          std::vector<cv::Point2f> points;
          for (int a = 0; a < mid_num; ++a) {
            points.emplace_back(cv::Point2f(
                int(round(std::get<2>(candA[i]).x - a * vec[0] / (mid_num - 1))),
                int(round(std::get<2>(candA[i]).y - a * vec[1] / (mid_num - 1)))));
          }
          vec[0] = vec[0] / norm;
          vec[1] = vec[1] / norm;
          std::vector<float> vec_x;
          std::vector<float> vec_y;
          std::vector<float> score_midpts;
          float sum = 0;
          int lencir = 0;
          for (size_t b = 0; b < points.size(); ++b) {
            vec_x.emplace_back(score_midx.at<float>(points[b].y, points[b].x));
            vec_y.emplace_back(score_midy.at<float>(points[b].y, points[b].x));
            score_midpts.emplace_back(
                abs(vec_x[b] * vec[0] + vec_y[b] * vec[1]));
            sum += score_midpts[b];
            if (score_midpts[b] > 0.05) lencir++;
          }
          float score_with_dist_prior =
              sum / score_midpts.size() + std::min(0.5 * width / norm - 1, 0.0);
          bool cirterion1 = lencir > 0.8 * score_midpts.size();
          bool cirterion2 = score_with_dist_prior > 0;
          if (cirterion1 && cirterion2) {
            connection_candidate.emplace_back(
                i, j, score_with_dist_prior,
                score_with_dist_prior + std::get<1>(candA[i]) + std::get<1>(candB[j]));
          }
        }
      }
      std::sort(connection_candidate.begin(), connection_candidate.end(),
                [](const std::tuple<int, int, float, float>& lhs,
                   const std::tuple<int, int, float, float>& rhs) {
                  return std::get<2>(lhs) > std::get<2>(rhs);
                });
      connection.clear();
      for (size_t c = 0; c < connection_candidate.size(); ++c) {
        int i = std::get<0>(connection_candidate[c]);
        int j = std::get<1>(connection_candidate[c]);
        float s = std::get<2>(connection_candidate[c]);
        if (!isThreeInConnection(connection, i) &&
            !isFourInConnection(connection, j)) {
          connection.emplace_back(std::get<0>(candA[i]), std::get<0>(candB[j]), s, i, j);
          if (connection.size() >= std::min(nA, nB)) break;
        }
      }
      connection_all.emplace_back(connection);
    } else {
      special_k.emplace_back(k);
      connection.clear();
      connection_all.emplace_back(connection);
    }
  }
}

struct OpenPoseResult {
  struct PosePoint {
    /// Point type \li \c 1 : "valid" \li \c 3 : "invalid"
    int type = 0;
    /// Coordinate point.
    cv::Point2f point;
  };
  /// A vector of pose, pose is represented by a vector of PosePoint.
  /// Joint points are arranged in order
  ///  0: head, 1: neck, 2: L_shoulder, 3:L_elbow, 4: L_wrist, 5: R_shoulder,
  ///  6: R_elbow, 7: R_wrist, 8: L_hip, 9:L_knee, 10: L_ankle, 11: R_hip,
  /// 12: R_knee, 13: R_ankle
  std::vector<std::vector<PosePoint>> poses;
};

std::vector<std::vector<OpenPoseResult::PosePoint>> getPoses(
    const AllPeaks& all_peaks, std::vector<AllConnection>& connection_all,
    std::vector<int>& special_k) {
  std::vector<std::vector<int>> subset(0, std::vector<int>(16, -1));
  Peaks candidate;
  for (auto& peaks : all_peaks) {
    for (auto& peak : peaks) {
      candidate.emplace_back(peak);
    }
  }
  for (size_t k = 0; k < mapIdx.size(); ++k) {
    if (find(special_k.begin(), special_k.end(), k) == special_k.end()) {
      int indexA = limbSeq[k][0];
      int indexB = limbSeq[k][1];
      for (size_t i = 0; i < connection_all[k].size(); ++i) {
        int found = 0;
        int partA = std::get<0>(connection_all[k][i]);
        int partB = std::get<1>(connection_all[k][i]);
        std::vector<int> subset_idx(2, -1);
        for (size_t j = 0; j < subset.size(); ++j) {
          if (subset[j][indexA] == partA || subset[j][indexB] == partB) {
            subset_idx[found] = j;
            found += 1;
          }
        }
        if (found == 1) {
          int j = subset_idx[0];
          if (subset[j][indexB] != partB) {
            subset[j][indexB] = partB;
            subset[j][15] += 1;
            subset[j][14] +=
                std::get<0>(candidate[partA]) + std::get<2>(connection_all[k][i]);
          }
        } else if (found == 2) {
          int j1 = subset_idx[0];
          int j2 = subset_idx[1];
          std::vector<int> membership(14, 0);
          for (size_t a = 0; a < membership.size(); ++a) {
            int x = subset[j1][a] >= 0 ? 1 : 0;
            int y = subset[j2][a] >= 0 ? 1 : 0;
            membership[a] = x + y;
          }
          if (find(membership.begin(), membership.end(), 2) ==
              membership.end()) {
            for (size_t a = 0; a < subset.size() - 2; ++a) {
              subset[j1][a] += (subset[j2][a] + 1);
            }
            for (size_t a = subset.size() - 2; a < subset.size(); ++a) {
              subset[j1][a] += subset[j2][a];
            }
            subset[j1][13] += std::get<2>(connection_all[k][i]);
          } else {
            subset[j1][indexB] = partA;
            subset[j1][15] += 1;
            subset[j1][14] +=
                std::get<0>(candidate[partB]) + std::get<2>(connection_all[k][i]);
          }
        } else if (found == 0 && k < 14) {
          std::vector<int> row(16, -1);
          row[indexA] = partA;
          row[indexB] = partB;
          row[15] = 2;
          row[14] = std::get<0>(candidate[partA]) + std::get<0>(candidate[partB]) +
                    std::get<2>(connection_all[k][i]);
          subset.emplace_back(row);
        }
      }
    }
  }
  for (size_t i = 0; i < subset.size(); ++i) {
    for (size_t j = 0; j < subset[i].size(); ++j) {
    }
    if (subset[i][15] < 4 || subset[i][14] / (subset[i][15]*1.0) < 0.4) {
      subset.erase(subset.begin() + i);
      --i;
    }
  }
  OpenPoseResult::PosePoint posePoint;
  std::vector<std::vector<OpenPoseResult::PosePoint>> poses(
      subset.size() + 1, std::vector<OpenPoseResult::PosePoint>(14, posePoint));
  for (size_t i = 0; i < subset.size(); ++i) {
    for (int j = 0; j < 14; ++j) {
      int idx = subset[i][j];
      if (idx == -1) {
        (poses[subset.size() - i][j]).type = 3;
        continue;
      }
      (poses[subset.size() - i][j]).type = 1;
      (poses[subset.size() - i][j]).point = std::get<2>(candidate[idx]);
    }
  }
  return poses;
}

vitis::ai::proto::DpuModelResult MyPostProcessor::process(
    const vart::simple_tensor_buffer_t<float>& chwdataL1_dpu,
    const vart::simple_tensor_buffer_t<float>& chwdataL2_dpu) {
  auto L1_shape = chwdataL1_dpu.tensor->get_shape();
  CHECK_EQ(L1_shape.size(), 4u);
  auto hL1 = L1_shape[1];
  auto wL1 = L1_shape[2];
  auto channelL1 = L1_shape[3];
  auto L2_shape = chwdataL2_dpu.tensor->get_shape();
  auto hL2 = L2_shape[1];
  auto wL2 = L2_shape[2];
  auto channelL2 = L2_shape[3];

  std::vector<float> chwdataL2;
  chwdataL2.reserve(hL2 * wL2 * channelL2);
  // transpose NHWC to NCHW
  for (int ih = 0; ih < hL2; ++ih) {
    for (int iw = 0; iw < wL2; ++iw) {
      for (int ic = 0; ic < channelL2; ++ic) {
        int offset = ic * wL2 * hL2 + ih * wL2 + iw;
        chwdataL2[offset] =
            chwdataL2_dpu.data[ih * wL2 * channelL2 + iw * channelL2 + ic];
      }
    }
  }
  int idx = -1;
  AllPeaks all_peaks;
  for (int i = 0; i < channelL2 - 1; ++i) {
    cv::Mat um(hL2, wL2, CV_32F, chwdataL2.data() + i * wL2 * hL2);
    resize(um, um, cv::Size(0, 0), 8, 8, cv::INTER_CUBIC);
    Peaks peaks;
#ifdef ENABLE_NEON
    find_peak_neon(um, peaks, idx);
#else
    find_peak(um, peaks, idx);
#endif
    all_peaks.emplace_back(peaks);
  }

  std::vector<float> chwdataL1;
  chwdataL1.reserve(hL1 * wL1 * channelL1);
  for (int ih = 0; ih < hL1; ++ih) {
    for (int iw = 0; iw < wL1; ++iw) {
      for (int ic = 0; ic < channelL1; ++ic) {
        int offset = ic * wL1 * hL1 + ih * wL1 + iw;
        chwdataL1[offset] =
            chwdataL1_dpu.data[ih * wL1 * channelL1 + iw * channelL1 + ic];
      }
    }
  }
  std::vector<cv::Mat> pafs;
  for (int i = 0; i < channelL1; ++i) {
    cv::Mat um(hL1, wL1, CV_32F, chwdataL1.data() + i * wL1 * hL1);
    cv::resize(um, um, cv::Size(0, 0), 8, 8, cv::INTER_CUBIC);
    pafs.emplace_back(um);
  }
  std::vector<int> special_k;
  std::vector<AllConnection> connection_all;
  auto sWidth = width_;
  auto sHeight = height_;
  findLines(sWidth, pafs, all_peaks, connection_all, special_k);
  auto poses = getPoses(all_peaks, connection_all, special_k);
  auto ret = vitis::ai::proto::DpuModelResult();
  auto key = ret.mutable_pose_detect_result();
  float scale_x = 1.0f / float(sWidth);
  float scale_y = 1.0f / float(sHeight);
  // TODO: why start from 1?
  for (size_t k = 1; k < poses.size(); ++k) {
    auto size = poses[k].size();
    CHECK_EQ(size, 14u);
    auto i = 0;

    if (poses[k][i].type == 1) {
      key->mutable_head()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_head()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_neck()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_neck()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_right_shoulder()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_right_shoulder()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_right_elbow()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_right_elbow()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_right_wrist()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_right_wrist()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_left_shoulder()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_left_shoulder()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_left_elbow()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_left_elbow()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_left_wrist()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_left_wrist()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_right_hip()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_right_hip()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_right_knee()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_right_knee()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_right_ankle()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_right_ankle()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_left_hip()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_left_hip()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_left_knee()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_left_knee()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
    if (poses[k][i].type == 1) {
      key->mutable_left_ankle()->set_x(poses[k][i].point.x * scale_x);
      key->mutable_left_ankle()->set_y(poses[k][i].point.y * scale_y);
    }
    i++;
  }
  return ret;
}

}  // namespace

extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<vitis::ai::XmodelPostprocessor<MyPostProcessor>>();
}
