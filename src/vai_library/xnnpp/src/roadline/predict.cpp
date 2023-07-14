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

#include "predict.hpp"

#include <sys/time.h>

#include <list>
#include <map>

namespace vitis {
namespace nnpp {
namespace roadline {

Predict::Predict(float ipm_width, float ipm_height)
    : ipm_width_{ipm_width}, ipm_height_{ipm_height} {}

bool Predict::curve_fit(std::vector<cv::Point> &key_point, int n, cv::Mat &A) {
  int N = key_point.size();
  cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
  for (int i = 0; i < n + 1; i++) {
    for (int j = 0; j < n + 1; j++) {
      for (int k = 0; k < N; k++) {
        X.at<double>(i, j) =
            X.at<double>(i, j) + std::pow(key_point[k].x, i + j);
      }
    }
  }
  cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
  for (int i = 0; i < n + 1; i++) {
    for (int k = 0; k < N; k++) {
      Y.at<double>(i, 0) =
          Y.at<double>(i, 0) + std::pow(key_point[k].x, i) * key_point[k].y;
    }
  }

  A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
  //求解矩阵A
  cv::solve(X, Y, A, cv::DECOMP_LU);
  return true;
}

void Predict::findLocalmaximum(vector<int> &datase,
                               vector<cv::Point_<int>> &seed) {
  auto peak_row = datase.begin();
  for (int i = 0; i < ipm_height_; i++) {
    int j = 0;
    while (j < ipm_width_ - 1) {
      int l = 0;
      while (*(peak_row + j) > 0 && (*(peak_row + j) == *(peak_row + j + 1)) &&
             j < ipm_width_ - 1) {
        l++;
        j++;
      }
      j++;
      if (l > 0) {
        l++;
        int max_idx = j - l / 2 - 1;
        seed.push_back(cv::Point_<int>(max_idx, i));
      }
    }
    peak_row = peak_row + ipm_width_;
  }
}

void Predict::cluster(const vector<int> &outImage, vector<int> &clusters) {
  vector<int> list;
  vector<vector<int>> indexs;
  for (int x = 0; x < ipm_height_; x++) {
    for (int y = 0; y < ipm_width_; y++) {
      if (outImage[ipm_width_ * x + y] > 0) {
        list.push_back(x);
        list.push_back(y);
        indexs.push_back(list);
        list.clear();
      }
    }
  }
  int range_x = 6;
  int range_y = 2;
  int cluster_class = 1, tem_class = 1;
  for (int i = indexs.size() - 1; i > -1; i--) {
    if (indexs[i].size() == 2) {
      indexs[i].push_back(cluster_class);
      tem_class = cluster_class;
      cluster_class += 1;
    } else {
      tem_class = indexs[i][2];
    }
    for (int j = indexs.size() - 1; j > -1; j--) {
      if (indexs[j].size() == 2 &&
          abs(indexs[i][0] - indexs[j][0]) <= range_x &&
          abs(indexs[i][1] - indexs[j][1]) <= range_y)
        indexs[j].push_back(tem_class);
    }
  }
  clusters = vector<int>(ipm_width_ * ipm_height_, 0);
  for (size_t i = 0; i < indexs.size(); i++) {
    clusters[indexs[i][0] * ipm_width_ + indexs[i][1]] = indexs[i][2];
  }
}

int Predict::majorityElement(const vector<int> &nums) {
  int cand = -1;
  int count = 0;

  int len = nums.size();
  for (int i = 1; i < len; i++) {
    if (count == 0) {
      count = 1;
      cand = nums[i];
    } else if (nums[i] == cand)
      count++;
    else
      count--;
  }
  return cand;
}

void Predict::voteClassOfClusters(const vector<int> &datase,
                                  const vector<int> &clusters,
                                  vector<int> &types) {
  assert(datase.size() > 0 && clusters.size() > 0);
  vector<vector<int>> tempClass(CNUM, vector<int>(0, 0));
  for (int i = 0; i < ipm_height_; i++) {
    for (int j = 0; j < ipm_width_; j++) {
      if (clusters[i * ipm_width_ + j] > 0) {
        assert(clusters[i * ipm_width_ + j] <= ipm_width_ * ipm_height_);
        tempClass[clusters[i * ipm_width_ + j]].push_back(
            datase[i * ipm_width_ + j]);
      }
    }
  }

  for (size_t i = 0; i < tempClass.size(); i++) {
    if (tempClass.size() > 0) types[i] = majorityElement(tempClass[i]);
  }
}

int Predict::getMaxX(const vector<cv::Point> &points) {
  int ret = 0;
  for (size_t i = 0; i < points.size(); i++) {
    ret = ret < points[i].x ? points[i].x : ret;
  }
  return ret;
}

int Predict::getMinX(const vector<cv::Point> &points) {
  int ret = INT_MAX;
  for (size_t i = 0; i < points.size(); i++) {
    ret = ret > points[i].x ? points[i].x : ret;
  }
  return ret;
}

}  // namespace roadline
}  // namespace nnpp
}  // namespace vitis
