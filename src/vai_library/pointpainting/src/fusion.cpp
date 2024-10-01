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

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include "./fusion.hpp"
#include "./utils.hpp"

DEF_ENV_PARAM(DEBUG_POINTPAINTING, "0");
DEF_ENV_PARAM(DEBUG_POINTPAINTING_RESULT, "0");
DEF_ENV_PARAM(DEBUG_POINTS_UNIQUE, "0");

namespace vitis { namespace ai { 
namespace pointpainting {

static cv::Mat build_cam2lidar(const CamInfo &cam_info) {
  cv::Mat cam2lidar = cv::Mat::eye(4, 4, CV_32FC1);
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      cam2lidar.at<float>(i, j) = cam_info.s2l_r[i * 3 + j];
    }
    cam2lidar.at<float>(i, 3) = cam_info.s2l_t[i];
  }
  return cam2lidar; 
}

//static void dot_product(float *m_left, int l_cols, int l_rows, float *m_right, int r_cols, float *output) {
//  for (auto i = 0; i < l_rows; ++i) {
//    for (auto j = 0; j < r_cols; ++i) {
//      auto c = 0.f;
//      for (auto k = 0; j < l_cols; ++i) {
//        c += m_left[i + l_cols + k] * m_right[k * r_cols + j];
//      }
//      output[i * r_cols + j] = c;
//    }
//  } 
//} 

static void get_one_hot_points(const cv::Mat &image, const cv::Mat &coors, 
                               const std::vector<float> &points,
                               std::vector<std::vector<float>> &result_points, int num_classes) {
  int w = image.cols;
  int h = image.rows;
  //std::vector<float> result_points;
  result_points.clear();
  auto coors_num = coors.rows;
  if (ENV_PARAM(DEBUG_POINTPAINTING)) {
    std::cout << "w: " << w << ", h: " << h 
              << ", step:" << image.step << std::endl;
    std::cout << "coors num: " << coors_num << std::endl;
  }

  for (auto i = 0; i < coors_num; ++i) {
    if (coors.at<float>(i, 0) <= 0.f || coors.at<float>(i, 0) > w) {
      continue;
    }
    if (coors.at<float>(i, 1) <= 0.f || coors.at<float>(i, 1) > h) {
      continue;
    }

    int icol = std::floor(coors.at<float>(i, 0));
    int irow = std::floor(coors.at<float>(i, 1));
    auto label = image.at<uint8_t>(irow, icol);
    if (label == 255) {
      label = num_classes - 1;
    }
    assert(label >=0 && label < num_classes);
    //auto index = result_points.size();
    //result_points.resize(index + 16);
    std::vector<float> p(16, 0);
    //std::copy(points.begin() + i * 5, points.begin() + (i + 1) * 5, 
    //          result_points.data() + index);
    //std::fill(result_points.data() + index + 5, result_points.data() + index + 16, 0);
    //result_points[index + 5 + label] = 1;
    std::copy(points.begin() + i * 5, points.begin() + (i + 1) * 5, 
              p.data());
    p[5 + label] = 1;
    result_points.emplace_back(p);
  }
  if (ENV_PARAM(DEBUG_POINTPAINTING)) {
    std::cout << "result_points size:" << result_points.size() << std::endl;
  }
  return ;
}

//bool compare(float (*l)[5], float (*r)[5]) {
//  bool smaller = false; 
//  for (auto i = 0; i < 5; i++) {
//    if (l[i] < r[i]) {
//      smaller = true;
//      break;
//    }
//  }
//  return smaller;
//}

bool compare(std::vector<float> l, std::vector<float> r) {
  bool smaller = false; 
  for (auto i = 0; i < 5; i++) {
    if (l[i] == r[i]) {
      continue;
    }
    return l[i] > r[i] ? false : true;
  }
  return smaller;
}

bool equal(std::vector<float> l, std::vector<float> r) {
  for (auto i = 0; i < 5; i++) {
    if (l[i] != r[i]) {
      return false;
    }
  }
  return true;
}



std::vector<float> fusion(const std::vector<CamInfo> &cam_infos, 
                          const std::vector<float> &points,
                          int dim,
                          const std::vector<cv::Mat> &images,
                          int num_classes) {
  std::vector<float> result;
  std::vector<std::vector<float>> result_for_sort;
  for (auto n = 0u; n < cam_infos.size(); ++n) {
__TIC__(SINGLE_CAM)
    auto& cam_info = cam_infos[n];
    auto points_num = points.size() / dim;
    auto &image = images[n];
    if (ENV_PARAM(DEBUG_POINTPAINTING)) {
      std::cout << "points num: " << points.size() << std::endl;;
      std::cout << "image size: " << image.cols 
                << " * " << image.rows 
                << " * " << image.channels() << std::endl;
    }
__TIC__(CAM_2_LIDAR)
    // 1. cam to lidar
    auto cam2lidar = build_cam2lidar(cam_info);
    if (ENV_PARAM(DEBUG_POINTPAINTING)) {
      std::cout << "cam2lidar: ";
      for (auto i = 0; i < cam2lidar.rows; ++i) {
        for (auto j = 0; j < cam2lidar.cols; ++j) {
          std::cout << cam2lidar.at<float>(i, j) << " ";
        }
      }
      std::cout << std::endl;
    }
 
__TOC__(CAM_2_LIDAR)
    // 2. lidar to cam 
__TIC__(INV)
    cv::Mat lidar2cam = cam2lidar.inv();
__TOC__(INV)
    if (ENV_PARAM(DEBUG_POINTPAINTING)) {
      std::cout << "lidar2cam: ";
      for (auto i = 0; i < lidar2cam.rows; ++i) {
        for (auto j = 0; j < lidar2cam.cols; ++j) {
          std::cout << lidar2cam.at<float>(i, j) << " ";
        }
      }
      std::cout << std::endl;
    }

    // 3. read points and make coors matrix
    // ones
__TIC__(MAKE_COORS)
    auto coors = cv::Mat(points_num, 4, CV_32FC1, 1);
    for (auto i = 0; i < coors.rows; ++i) {
      memcpy(coors.ptr<float>(i), points.data() + i * dim, 3 * sizeof(float));
    }
__TOC__(MAKE_COORS)
    // 4. dot product 
    //auto coors_cam = coors.dot(lidar2cam.t()); 
__TIC__(DOT_0)
    cv::Mat coors_cam = coors * lidar2cam.t(); 
__TOC__(DOT_0)
    //if (ENV_PARAM(DEBUG_POINTPAINTING)) {
    //  debug_mat(coors_cam, "coors_cam");
    //}

    // 5. remove points back to cam
__TIC__(SELECT_POINTS)
    std::vector<float> valid_points;
    std::vector<float> valid_coors_cam;
    for (auto i = 0; i < coors_cam.rows; ++i) {
      if (coors_cam.at<float>(i, 2) < 0.01) {
        continue;
      }
      std::copy(points.data() + i * dim, points.data() + (i + 1) * dim,
                std::back_inserter(valid_points));
      std::copy(coors_cam.ptr<float>(i), coors_cam.ptr<float>(i + 1),
                std::back_inserter(valid_coors_cam));
    }
__TOC__(SELECT_POINTS)

    if (ENV_PARAM(DEBUG_POINTPAINTING)) {
      std::cout << "valid_coors_cam size:" << valid_coors_cam.size() << std::endl;
      std::cout << "valid_points size:" << valid_points.size() << std::endl;
    }
__TIC__(DOT_1)
    cv::Mat m_valid_coors_cam(valid_coors_cam.size() / 4, 4, CV_32FC1, valid_coors_cam.data());

    cv::Mat cam_intr(3, 3, CV_32FC1, const_cast<float *>(cam_info.cam_intr.data()));
    cv::Rect rect(0, 0, 3, m_valid_coors_cam.rows);
    auto coors_cam_rect = m_valid_coors_cam(rect);
    //auto coors_img = coors_cam_rect.dot(cam_intr.t());
    if (ENV_PARAM(DEBUG_POINTPAINTING)) {
    std::cout << "coors_cam_rect col: " <<  coors_cam_rect.cols << ", row:" << coors_cam_rect.rows << std::endl;
    std::cout << "cam_intr col: " <<  cam_intr.cols << ", row:" << cam_intr.rows << std::endl;
    }
    
    cv::Mat coors_img = coors_cam_rect * cam_intr.t();
__TOC__(DOT_1)

    //if (ENV_PARAM(DEBUG_POINTPAINTING)) {
    //  debug_mat(coors_img, "coors_img");
    //}

    // 6. coors image normalize
__TIC__(NORM)
    for (auto i = 0; i < coors_img.rows; ++i) {
      for (auto j = 0; j < 3; ++j) {
        coors_img.at<float>(i, j) /= coors_img.at<float>(i, 2);
      }
    }
__TOC__(NORM)
 
    //if (ENV_PARAM(DEBUG_POINTPAINTING)) {
    //  save_mat(coors_img, std::string("coors_img_normalize") + std::to_string(n));
    //  //debug_mat(coors_img, "coors_img_norm");
    //}
    // 7. get one hot points  
    //auto result_points = get_one_hot_points(image, coors_img, valid_points);
__TIC__(ONE_HOT)
    std::vector<std::vector<float>> result_points;
    get_one_hot_points(image, coors_img, valid_points, result_points, num_classes);
__TOC__(ONE_HOT)
    //if (ENV_PARAM(DEBUG_POINTPAINTING)) {
    //  save_vector(result_points, std::string("ori_result_") + std::to_string(n));
    //}

__TIC__(COPY)
    std::copy(result_points.begin(), result_points.end(), std::back_inserter(result_for_sort));
__TOC__(COPY)
__TOC__(SINGLE_CAM)
  }

  if (ENV_PARAM(DEBUG_POINTPAINTING_RESULT)) {
    std::ofstream o("./no_sort.txt");
    for (auto i = 0u; i < result_for_sort.size(); ++i) {
      for (auto j = 0u; j < result_for_sort[i].size(); ++j) {
        o << result_for_sort[i][j] << " ";
      }
      o << std::endl;
    } 
    o.close();
  }

  // (optional) sort and unique 
  if (ENV_PARAM(DEBUG_POINTS_UNIQUE)) {
    // sort  
__TIC__(SORT)
    std::stable_sort(result_for_sort.begin(), result_for_sort.end(), compare);
__TOC__(SORT)
    //std::stable_sort(result_for_sort.data(), result_for_sort.data() + result_for_sort.size(), compare);
    //if (ENV_PARAM(DEBUG_POINTPAINTING)) {
    //  debug_vector(result_for_sort, "result");
    //}
    // unique
__TIC__(UNIQUE)
    auto last = std::unique(result_for_sort.begin(), result_for_sort.end(), equal);
__TOC__(UNIQUE)

__TIC__(COPY_2)
    for (auto r = result_for_sort.begin(); r != last ; ++r) {
      std::copy(r->begin(), r->end(), std::back_inserter(result));
    }
__TOC__(COPY_2)
    if (ENV_PARAM(DEBUG_POINTPAINTING_RESULT)) {
      auto n = std::distance(result_for_sort.begin(), last);
      std::cout << "n: " << n << std::endl;
      std::ofstream o("./result.txt");
      for (auto i = 0; i < n; ++i) {
        for (auto j = 0u; j < result_for_sort[i].size(); ++j) {
          o << result_for_sort[i][j] << " ";
        }
        o << std::endl;
      } 
      o.close();
    }

  } else {
__TIC__(COPY_2)
    for (auto it = result_for_sort.begin(); it != result_for_sort.end(); ++it) {
      std::copy(it->begin(), it->end(), std::back_inserter(result));
    }
__TOC__(COPY_2)
  }     

  return result;
}

}}}
