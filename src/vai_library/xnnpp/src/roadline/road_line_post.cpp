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
#include "road_line_post.hpp"

#include <sys/stat.h>
#include <fstream>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/max_index.hpp>
#include <vitis/ai/profiling.hpp>

#include "predict.hpp"

using namespace std;
using namespace vitis::nnpp::roadline;

string g_roadline_acc_outdir = "";

namespace vitis {
namespace ai {
static std::unique_ptr<vitis::nnpp::roadline::IpmInfo> createIpm(
    const vitis::ai::proto::DpuModelParam& config) {
  return std::unique_ptr<vitis::nnpp::roadline::IpmInfo>(
      new vitis::nnpp::roadline::IpmInfo(
          config.roadline_param().ratio(), config.roadline_param().ipm_width(),
          config.roadline_param().ipm_height(),
          config.roadline_param().ipm_left(),
          config.roadline_param().ipm_right(),
          config.roadline_param().ipm_top(),
          config.roadline_param().ipm_bottom(),
          config.roadline_param().ipm_interpolation(),
          config.roadline_param().ipm_vp_portion(),
          config.roadline_param().focal_length_x(),
          config.roadline_param().focal_length_y(),
          config.roadline_param().optical_center_x(),
          config.roadline_param().optical_center_y(),
          config.roadline_param().camera_height(),
          config.roadline_param().pitch(), config.roadline_param().yaw()));
}

static std::unique_ptr<vitis::nnpp::roadline::Predict> createPredict(
    const vitis::ai::proto::DpuModelParam& config) {
  return std::unique_ptr<vitis::nnpp::roadline::Predict>(
      new vitis::nnpp::roadline::Predict(config.roadline_param().ipm_width(),
                                         config.roadline_param().ipm_height()));
}

inline int maxValMap(map<int, int>& mymap) {
  int max_v = 0;
  int max_k = 0;
  for (auto iter = mymap.begin(); iter != mymap.end(); iter++)
    if (iter->second > max_v) {
      max_v = iter->second;
      max_k = iter->first;
    }
  return max_k;
}

vector<Point> drawpoly(int x, int y, int w, int h, Mat& label, int sWidth,
                       int sHeight, int inWidth, int inHeight,
                       const vitis::ai::proto::DpuModelParam& config) {
  int resize_w = config.roadline_dp_param().resize_w();
  int resize_h = config.roadline_dp_param().resize_h();
  int crop_x = config.roadline_dp_param().crop_x();
  int crop_y = config.roadline_dp_param().crop_y();
  int crop_w = config.roadline_dp_param().crop_w();
  int crop_h = config.roadline_dp_param().crop_h();
  int epow = config.roadline_dp_param().epow();

  float scale_w = (float)inWidth / (float)resize_w;
  float scale_h = (float)inHeight / (float)resize_h;
  float scale_const_w = (float)crop_w / (float)sWidth;
  float scale_const_h = (float)crop_h / (float)sHeight;
  int h_step = (h - 1) / 14;
  if (h_step == 0) h_step = h - 1;
  vector<Point> mid_points;
  // select the actual line points in the area.
  map<int, int> index;
  for (int i = y + 1; i < y + h; i += h_step) {
    for (int j = x; j < x + w; ++j) {
      int value = label.at<int>(i, j);
      if (value != 0) {
        if (index.empty()) {
          index.insert(pair<int, int>(value, 1));
        } else if (index.find(value) != index.end()) {
          index[value]++;
        } else {
          index.insert(pair<int, int>(value, 1));
        }
      }
    }
  }
  int max_value = maxValMap(index);

  for (int i = y + 1; i < y + h; i += h_step) {
    vector<int> x_coor;
    for (int j = x; j < x + w; ++j) {
      if (label.at<int>(i, j) == max_value) {
        x_coor.push_back(j);
      }
    }
    int mid = accumulate(x_coor.begin(), x_coor.end(), 0.0);
    mid_points.push_back(Point((float)mid / x_coor.size() * scale_const_w,
                               (float)i * scale_const_h));
  }
  Mat pxm = Mat::zeros(epow + 1, epow + 1, CV_32FC1);
  Mat pym = Mat::zeros(epow + 1, 1, CV_32FC1);
  for (int i = 0; i < epow + 1; ++i) {
    for (size_t k = 0; k < mid_points.size(); ++k) {
      pym.at<float>(i, 0) += pow(mid_points[k].x, i) * mid_points[k].y;
    }
    for (int j = 0; j < epow + 1; ++j) {
      for (size_t k = 0; k < mid_points.size(); ++k) {
        pxm.at<float>(i, j) = pxm.at<float>(i, j) + pow(mid_points[k].x, i + j);
      }
    }
  }
  Mat res = Mat::zeros(epow + 1, 1, CV_32FC1);
  cv::solve(pxm, pym, res, DECOMP_LU);
  vector<Point> point_new;
  for (int i = x; i < w + x; ++i) {
    float iy;
    if (epow == 1)
      iy = res.at<float>(0, 0) + res.at<float>(1, 0) * i;
    else if (epow == 2)
      iy = res.at<float>(0, 0) + res.at<float>(1, 0) * i +
           res.at<float>(2, 0) * i * i;
    else {
      cerr << "epow error" << endl;
      exit(0);
    }
    point_new.push_back(Point((i + crop_x) * scale_w, (iy + crop_y) * scale_h));
  }
  return point_new;
}

RoadLinePost::~RoadLinePost(){};

RoadLinePost::RoadLinePost(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config)
    : ipminfo_(createIpm(config)),
      predict_(createPredict(config)),
      config_(config),
      input_tensors_(input_tensors) {
  auto layername =
      std::vector<std::string>(config.roadline_param().layer_name().begin(),
                               config.roadline_param().layer_name().end());
  for (auto i = 0u; i < layername.size(); i++) {
    for (auto j = 0u; j < output_tensors.size(); j++) {
      if (output_tensors[j].name.find(layername[i]) != std::string::npos) {
        output_tensors_.emplace_back(output_tensors[j]);
        break;
      }
    }
  }
}

RoadLineResult RoadLinePost::road_line_post_process_internal(int inWidth,
                                                             int inHeight,
                                                             unsigned int idx) {
  int sWidth = input_tensors_[0].width;
  int sHeight = input_tensors_[0].height;
  vector<RoadLineResult::Line> lines;
  string model_name = config_.name();
  if (model_name == "roadline") {
    std::unique_ptr<vitis::nnpp::roadline::IpmInfo> ipminfo =
        createIpm(config_);
    std::unique_ptr<vitis::nnpp::roadline::Predict> predict =
        createPredict(config_);
    if (ipminfo == nullptr || predict == nullptr ) {
       return RoadLineResult{sWidth, sHeight, lines};
    }
    int ipm_width = config_.roadline_param().ipm_width();
    int ipm_height = config_.roadline_param().ipm_height();
    vector<int> datase;
    auto base = (int8_t*)output_tensors_[0].get_data(idx);
    // read_input(base, "data.bin");
    for (size_t i = 0; i < output_tensors_[0].size / input_tensors_[0].batch;
         i = i + 4) {
      auto max_ind = max_element(base + i, base + i + 4);
      int posit = distance(base + i, max_ind);
      datase.push_back(posit);
    }
    vector<Point_<int>> seed;
    predict->findLocalmaximum(datase, seed);
    vector<int> data_ipm(ipm_width * ipm_height, 0);
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
    if (!g_roadline_acc_outdir.empty()) {
      if (access(g_roadline_acc_outdir.c_str(), 0) == -1){
        if (mkdir(g_roadline_acc_outdir.c_str(), 0777)) {
           // std::cout <<"mkdir failed " << g_roadline_acc_outdir <<"\n";
           return RoadLineResult{sWidth, sHeight, lines};
        }
      }
      std::ofstream out_datase(g_roadline_acc_outdir + "/datase.txt", ios::app);
      std::ofstream out_seedx(g_roadline_acc_outdir + "/seedx.txt", ios::app);
      std::ofstream out_seedy(g_roadline_acc_outdir + "/seedy.txt", ios::app);
      for (size_t i = 0; i < datase.size(); ++i) {
        out_datase << datase[i] << " ";
      }
      for (auto& point : seed) {
        out_seedx << point.y << " ";
        out_seedy << point.x << " ";
      }
      out_datase << endl;
      out_seedx << endl;
      out_seedy << endl;
      out_datase.close();
      out_seedx.close();
      out_seedy.close();
    }
    vector<int> outImage(ipm_width * ipm_height);
    ipminfo->IPM(data_ipm, outImage);

    // DBSCAN cluster
    vector<int> clusters;
    predict->cluster(outImage, clusters);
    vector<vector<cv::Point>> points_fitteds(CNUM, vector<cv::Point>(0));

    // RecoverFromIPM
    vector<int> recoverImg(ipm_width * ipm_height);
    ipminfo->Recover(clusters, recoverImg);
    for (int i = 0; i < ipm_height; i++) {
      for (int j = 0; j < ipm_width; j++)
        if (recoverImg[i * ipm_width + j] > 0) {
          points_fitteds[recoverImg[i * ipm_width + j]].push_back(
              Point_<int>(j * 8 + 4, i * 8 + 4));
        }
    }
    vector<int> types(CNUM, -1);
    predict->voteClassOfClusters(datase, recoverImg, types);

    Mat A;
    vector<cv::Point> points_poly;
    for (int i = 1; points_fitteds[i].size() > 0; i++) {
      predict->curve_fit(points_fitteds[i], 1, A);
      points_poly.clear();
      int minX = predict->getMinX(points_fitteds[i]);
      int maxX = predict->getMaxX(points_fitteds[i]);
      for (int x = minX; x <= maxX; x++) {
        double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x;
        points_poly.push_back(cv::Point(x, y));
      }
      lines.emplace_back(RoadLineResult::Line{types[i], points_poly});
    }
  } else if (model_name == "roadline_deephi") {
    int tensorSize = output_tensors_[0].size / input_tensors_[0].batch;
    int8_t* outdata = (int8_t*)output_tensors_[0].get_data(idx);
    int outHeight = output_tensors_[0].height;
    int outWidth = output_tensors_[0].width;
    int outChannel = output_tensors_[0].channel;
    float area_threshold = config_.roadline_dp_param().area_threshold();

    vector<uint8_t> max_pixel(tensorSize);
    vitis::ai::max_index_void(outdata, outWidth, outHeight, outChannel,
                              max_pixel.data());
    Mat out_img(outHeight, outWidth, CV_8UC1, max_pixel.data());

    Mat label, stats, centroids;
    // get the connected region of the dilate_img, its connected number will
    // store in nccomps. the label store each region, which be set as {1, 2 ...}
    // to show their sequence. the stats store each region's coordinate and area
    // centroids store each region's centre pointer.
    int nccomps =
        cv::connectedComponentsWithStats(out_img, label, stats, centroids);
    for (int i = 0; i < nccomps; ++i) {
      if (stats.at<int>(i, CC_STAT_LEFT) == 0) continue;
      if (stats.at<int>(i, CC_STAT_AREA) > area_threshold) {
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);
        lines.emplace_back(
            RoadLineResult::Line{1, drawpoly(x, y, w, h, label, sWidth, sHeight,
                                             inWidth, inHeight, config_)});
      }
    }
  }
  return RoadLineResult{sWidth, sHeight, lines};
}

std::vector<RoadLineResult> RoadLinePost::road_line_post_process(
    const std::vector<int>& inWidth, const std::vector<int>& inHeight,
    size_t batch_size) {
  auto ret = std::vector<vitis::ai::RoadLineResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; ++i) {
    ret.emplace_back(
        road_line_post_process_internal(inWidth[i], inHeight[i], i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
