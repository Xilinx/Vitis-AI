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

#include "./ipm_info.hpp"
#include "vitis/ai/xmodel_postprocessor.hpp"

#define CNUM 20

namespace {

static float get_param_float(const xir::Graph* graph, const char* name) {
  // python does not support float.
  return static_cast<float>(graph->get_attr<double>(std::string(name)));
}

static int get_param_int(const xir::Graph* graph, const char* name) {
  // python does not support float.
  return graph->get_attr<int>(std::string(name));
}

class MyPostProcessor {
 public:
  static xir::OpDef get_op_def() {
    return xir::OpDef("lane_detect")  //
        .add_input_arg(xir::OpArgDef{"input", xir::OpArgDef::REQUIRED,
                                     xir::DataType::Type::FLOAT, "5pt"})
        .set_annotation("postprocessor for lane_detect.");
  }

  explicit MyPostProcessor(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) {
    auto graph_input_shape = args.graph_input_tensor->get_shape();
    CHECK_EQ(graph_input_shape.size(), 4u);
    height_ = graph_input_shape[1];
    width_ = graph_input_shape[2];
    auto graph = args.graph;
    CHECK_EQ(args.graph_output_tensor_buffers.inputs.size(), 1u);
    auto input = args.graph_output_tensor_buffers.inputs[0];
    CHECK_EQ(input.args.size(), 1u);
    auto input_op = input.args[0].op;
    auto input_shape = input_op->get_output_tensor()->get_shape();
    CHECK_EQ(input_shape.size(), 4u);
    auto ipm_height = (float)input_shape[1];
    auto ipm_width = (float)input_shape[2];
    ipminfo_ = std::make_unique<vitis::nnpp::roadline::IpmInfo>(
        get_param_int(graph, "ratio"),  //
        ipm_width, ipm_height,          //
        get_param_float(graph, "ipm_left"),
        get_param_float(graph, "ipm_right"),  //
        get_param_float(graph, "ipm_top"),
        get_param_float(graph, "ipm_bottom"),                             //
        get_param_float(graph, "ipm_interpolation"),                      //
        get_param_float(graph, "ipm_vp_portion"),                         //
        get_param_float(graph, "focal_length_x"),                         //
        get_param_float(graph, "focal_length_y"),                         //
        get_param_float(graph, "optical_center_x"),                       //
        get_param_float(graph, "optical_center_y"),                       //
        get_param_float(graph, "camera_height"),                          //
        get_param_float(graph, "pitch"), get_param_float(graph, "yaw"));  //
    ipminfo_->initialize_ipm();
  }

  vitis::ai::proto::DpuModelResult process(
      const vart::simple_tensor_buffer_t<float>& tensor_buffer);

 private:
  int width_;
  int height_;
  std::unique_ptr<vitis::nnpp::roadline::IpmInfo> ipminfo_;
};

using Point = cv::Point_<int>;

std::vector<Point> findLocalmaximum(vector<int>& datase, int ipm_height_,
                                    int ipm_width_) {
  auto ret = std::vector<Point>();
  ret.reserve(datase.size());
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
        ret.push_back(Point{max_idx, i});
      }
    }
    peak_row = peak_row + ipm_width_;
  }
  return ret;
}

static void cluster(const vector<int>& outImage, vector<int>& clusters,
                    int ipm_height_, int ipm_width_) {
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

static int majorityElement(const vector<int>& nums) {
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

static void voteClassOfClusters(const vector<int>& datase,
                                const vector<int>& clusters, vector<int>& types,
                                int ipm_height_, int ipm_width_) {
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

static bool curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A) {
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

static int getMaxX(const vector<cv::Point>& points) {
  int ret = 0;
  for (size_t i = 0; i < points.size(); i++) {
    ret = ret < points[i].x ? points[i].x : ret;
  }
  return ret;
}

static int getMinX(const vector<cv::Point>& points) {
  int ret = std::numeric_limits<int>::max();
  for (size_t i = 0; i < points.size(); i++) {
    ret = ret > points[i].x ? points[i].x : ret;
  }
  return ret;
}

vitis::ai::proto::DpuModelResult MyPostProcessor::process(
    const vart::simple_tensor_buffer_t<float>& tensor_buffer) {
  auto base = tensor_buffer.data;
  auto input_shape = tensor_buffer.tensor->get_shape();
  CHECK_EQ(input_shape.size(), 4u);
  auto num_of_classes = input_shape[3];
  auto ipm_height = input_shape[1];
  auto ipm_width = input_shape[2];
  CHECK_EQ(num_of_classes, 4);
  vector<int> datase;
  int c = 0;
  datase.reserve(ipm_height * ipm_width);
  for (auto h = 0; h < ipm_height; ++h) {
    for (auto w = 0; w < ipm_width; ++w) {
      auto max_ind = std::max_element(base + c, base + c + num_of_classes);
      int posit = distance(base + c, max_ind);
      datase.push_back(posit);
      c = c + num_of_classes;
    }
  }
  auto seed = findLocalmaximum(datase, ipm_height, ipm_width);
  vector<int> data_ipm(datase.size(), 0);
  for (size_t i = 0; i < seed.size(); i++) {
    if (datase[seed[i].y * ipm_width + seed[i].x] == 1) {
      data_ipm[seed[i].y * ipm_width + seed[i].x] = 1;
    }
    if (datase[seed[i].y * ipm_width + seed[i].x] == 2) {
      data_ipm[seed[i].y * ipm_width + seed[i].x] = 2;
    }
    if (datase[seed[i].y * ipm_width + seed[i].x] == 3) {
      data_ipm[seed[i].y * ipm_width + seed[i].x] = 3;
    }
    if (datase[seed[i].y * ipm_width + seed[i].x] == 0) {
    }
  }

  vector<int> outImage(ipm_height * ipm_width);
  ipminfo_->IPM(data_ipm, outImage);

  // DBSCAN cluster
  vector<int> clusters;
  cluster(outImage, clusters, ipm_height, ipm_width);
  vector<vector<Point>> points_fitteds(CNUM, vector<Point>(0));

  // RecoverFromIPM
  vector<int> recoverImg(ipm_width * ipm_height);
  ipminfo_->Recover(clusters, recoverImg);
  for (int i = 0; i < ipm_height; i++) {
    for (int j = 0; j < ipm_width; j++)
      if (recoverImg[i * ipm_width + j] > 0) {
        points_fitteds[recoverImg[i * ipm_width + j]].push_back(
            Point{j * 8 + 4, i * 8 + 4});
      }
  }

  vector<int> types(CNUM, -1);
  voteClassOfClusters(datase, recoverImg, types, ipm_height, ipm_width);

  auto ret = vitis::ai::proto::DpuModelResult();
  auto r = ret.mutable_roadline_result();

  for (int i = 1; points_fitteds[i].size() > 0; i++) {
    cv::Mat A;
    vector<cv::Point> points_poly;
    curve_fit(points_fitteds[i], 1, A);
    int minX = getMinX(points_fitteds[i]);
    int maxX = getMaxX(points_fitteds[i]);
    for (int x = minX; x <= maxX; x++) {
      double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x;
      points_poly.push_back(cv::Point(x, y));
    }
    //
    auto line = r->mutable_line_attribute()->Add();
    line->set_type(types[i]);
    for (auto& p : points_poly) {
      auto point = line->mutable_point()->Add();
      point->set_x(static_cast<float>(p.x) / static_cast<float>(width_));
      point->set_y(static_cast<float>(p.y) / static_cast<float>(height_));
    }
  }
  return ret;
}

}  // namespace

extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<vitis::ai::XmodelPostprocessor<MyPostProcessor>>();
}
