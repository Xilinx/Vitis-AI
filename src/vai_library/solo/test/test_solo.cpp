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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/solo.hpp>
#include <sstream>
#include <iomanip>

std::vector<std::vector<uint8_t>> color_map {
{ 41 , 215 , 206 }, { 40 , 136 , 255 }, { 66 , 211 , 46 }, { 168 , 29 , 134 }, { 145 , 253 , 52 }, 
{ 173 , 100 , 99 }, { 113 , 27 , 229 },{ 177 , 161 , 16 }, { 17 , 217 , 46 }, { 227 , 157 , 70 }, { 206 , 249 , 35 }, 
{ 164 , 225 , 245 }, { 16 , 3 , 124 },  { 33 , 11 , 240 }, { 187 , 70 , 199 }, { 62 , 19 , 108 }, { 62 , 51 , 172 }, 
{ 168 , 110 , 113 }, { 72 , 62 , 84 }, { 245 , 143 , 170 }, { 234 , 228 , 0 }, { 1 , 169 , 30 }, { 32 , 34 , 168 }, 
{ 207 , 4 , 155 }, { 172 , 133 , 179 }, { 230 , 111 , 194 },{ 21 , 165 , 138 }, { 163 , 64 , 51 }, { 2 , 65 , 7 }, 
{ 229 , 214 , 12 }, { 209 , 209 , 221 }, { 49 , 191 , 177 }, { 140 , 135 , 150 }, { 137 , 32 , 97 }, { 52 , 6 , 157 }, 
{ 248 , 81 , 39 }, { 212 , 60 , 86 }, { 130 , 215 , 115 }, { 44 , 177 , 241 }, { 219 , 60 , 37 }, { 100 , 124 , 189 }, 
{ 63 , 135 , 50 }, { 162 , 204 , 97 }, { 84 , 221 , 181 }, { 83 , 139 , 119 }, { 169 , 34 , 230 }, { 125 , 6 , 159 }, 
{ 217 , 99 , 100 }, { 218 , 17 , 54 }, { 53 , 138 , 43 }, { 71 , 215 , 225 }, { 109 , 5 , 86 }, { 211 , 10 , 133 }, 
{ 208 , 214 , 9 }, { 13 , 93 , 10 }, { 190 , 143 , 46 }, { 201 , 204 , 109 }, { 42 , 23 , 46 }, { 30 , 216 , 194 }, 
{ 103 , 35 , 29 }, { 97 , 31 , 71 }, { 189 , 103 , 156 }, { 105 , 249 , 121 }, { 22 , 188 , 210 }, { 113 , 158 , 9 }, 
{ 166 , 158 , 31 }, { 253 , 172 , 135 }, { 158 , 145 , 45 }, { 111 , 225 , 98 }, { 115 , 204 , 90 },{ 197 , 108 , 244 }, 
{ 176 , 109 , 0 }, { 205 , 63 , 88 }, { 138 , 130 , 20 }, { 2 , 25 , 3 }, { 179 , 60 , 246 }, { 66 , 104 , 40 }, 
{ 224 , 126 , 196 }, { 218 , 149 , 152 }, { 39 , 124 , 172 }};

std::vector<std::string> class_name = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
"truck", "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", 
"bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove", "skateboard", "surfboard", "tennis_racket", 
"bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", 
"chair", "couch", "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink", 
"refrigerator", "book", "clock", "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush"};

DEF_ENV_PARAM(SOLO_FULL_OUTPUT, "0");

auto score_thr = 0.3f;

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cerr << "usage: " << argv[0] << "  modelname  image_file_url " << endl;
    abort();
  }
  Mat input_img = imread(argv[2]);
  if (input_img.empty()) {
    cerr << "can't load image! " << argv[2] << endl;
    return -1;
  }
  auto det = vitis::ai::Solo::create(argv[1]);  // Init
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto result = det->run(input_img);
  auto cate_labels = result.cate_labels;
  auto cate_scores = result.cate_scores;
  if (ENV_PARAM(SOLO_FULL_OUTPUT)==1) {
    for(auto i = 0u; i < cate_labels.size(); i++) {
      std::cout<< cate_labels[i] << " " << cate_scores[i] << std::endl;
    }
  }
  auto vis_inds = cate_scores.filter(score_thr);
  auto seg_label  = result.seg_masks.slice_dim0(vis_inds, true);
  cate_labels = cate_labels.bool_select(vis_inds);
  cate_scores = cate_scores.bool_select(vis_inds);
  auto num_mask = seg_label.shape[0];
  std::vector<int> md;
  for(auto i = 0u; i < num_mask; i++) {
    md.push_back(seg_label.slice({{i, i + 1}, {0, seg_label.shape[1]}, {0, seg_label.shape[2]}}).sum());
  }
  Ndarray<int> mask_density(md, {md.size()});
  auto orders = mask_density.argsort(1);
  //vitis::ai::pr("", orders, 0, 13); 
  cate_labels = cate_labels.select(orders);
  cate_scores = cate_scores.select(orders);
  seg_label = seg_label.slice_dim0(orders);
  CHECK_EQ(input_img.cols, seg_label.shape[2]);
  CHECK_EQ(input_img.rows, seg_label.shape[1]);
  CHECK_EQ(color_map.size(), class_name.size());
  cv::Mat output_img;
  input_img.copyTo(output_img);
  std::vector<cv::Point> points;
  for(auto i = 0u; i < num_mask; i++) {
    auto cur_mask = seg_label.slice({{i, i + 1}, {0, seg_label.shape[1]}, {0, seg_label.shape[2]}});
    bool put_text = false;
    for(auto k = 0u; k < cur_mask.shape[1]; k++) {
      for(auto j = 0u; j < cur_mask.shape[2]; j++) {
        if(cur_mask[k * cur_mask.shape[2] + j] == 1) {
          auto value = input_img.at<Vec3b>(k, j);
          value[0] = value[0] * 0.4 + color_map[cate_labels[i]][0] * 0.6;
          value[1] = value[1] * 0.4 + color_map[cate_labels[i]][1] * 0.6;
          value[2] = value[2] * 0.4 + color_map[cate_labels[i]][2] * 0.6; 
          output_img.at<Vec3b>(k, j)= value;
          if (put_text == false) {
            points.push_back(cv::Point(j + 1, k + 5));
            put_text = true;
          }
        }
      }
    }
  }
  for(auto i = 0u; i < num_mask; i++) {
    std::stringstream ss; 
    ss << std::setprecision(4) << std::round(cate_scores[i] * 1000) / 1000;
    std::cout << "object: " << i << " " << class_name[cate_labels[i]] << "   " << std::setprecision(5) << cate_scores[i] << std::endl;
    cv::putText(output_img, class_name[cate_labels[i]]+ " " + ss.str(), 
    points[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, 1);
  }
  cv::imwrite("solo_result.jpg", output_img);
  return 0;
}
