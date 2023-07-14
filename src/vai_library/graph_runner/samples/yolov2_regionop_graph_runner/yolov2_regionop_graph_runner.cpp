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
#include <glog/logging.h>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/graph_runner.hpp"

// env var control the nms confidence threshold: 0.001 for accuracy test
DEF_ENV_PARAM_2(CONF_THRESH, "0.3", float)

using namespace std;

const float scale_0 = 0.00390625;
float conf_thresh = 0.001;
float nms_thresh = 0.5;
std::vector<cv::Mat> images;

static void preprocess_yolov2(const std::vector<std::string>& files, 
		                const std::vector<vart::TensorBuffer*>& input_tensor_buffers);
static void postprocess_yolov2(const std::vector<vart::TensorBuffer*>& output_tensor_buffers);
static int get_fix_point(const xir::Tensor* tensor);
static std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor);
static void read_images(const std::vector<std::string>& files, size_t batch);
static void set_input_image(const cv::Mat& image, void* data1, float scale);
static float overlap(float x1, float w1, float x2, float w2) ;
static float cal_iou(vector<float> box, vector<float> truth) ;
static void applyNMS(const vector<vector<float>>& boxes,
              const vector<float>& scores,
              const float nms,
              const float conf,
              int cls_index,
              vector<std::pair<int,int>>& res) ;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }
  std::string g_xmodel_file = std::string(argv[1]);
  std::vector<std::string> g_image_files;
  for (auto i = 2; i < argc; i++) {
    g_image_files.push_back(std::string(argv[i]));
  }

  //create graph runner
  auto graph = xir::Graph::deserialize(g_xmodel_file);
  auto attrs = xir::Attrs::create();
  auto runner =
      vitis::ai::GraphRunner::create_graph_runner(graph.get(), attrs.get());
  CHECK(runner != nullptr);

  //get input/output tensor buffers
  auto input_tensor_buffers = runner->get_inputs();
  auto output_tensor_buffers = runner->get_outputs();

  //preprocess and fill input
  preprocess_yolov2(g_image_files, input_tensor_buffers);
  //sync input tensor buffers
  for (auto& input : input_tensor_buffers) {
      input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                   input->get_tensor()->get_shape()[0]);
    }

  //run graph runner
  auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
  auto status = runner->wait((int)v.first, -1);
  CHECK_EQ(status, 0) << "failed to run the graph";

  //sync output tensor buffers
  for (auto output : output_tensor_buffers) {
      output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                   output->get_tensor()->get_shape()[0]);
    }

  //postprocess and print yolov2 result
  postprocess_yolov2(output_tensor_buffers);

  return 0;
}

static void preprocess_yolov2(const std::vector<std::string>& files, 
		                const std::vector<vart::TensorBuffer*>& input_tensor_buffers) {
  auto input_tensor = input_tensor_buffers[0]->get_tensor();
  auto batch = input_tensor->get_shape().at(0);
  auto height = input_tensor->get_shape().at(1);
  auto width = input_tensor->get_shape().at(2);

  int fixpos = get_fix_point(input_tensor);
  float input_fixed_scale = std::exp2f(1.0f * (float)fixpos);   (void)input_fixed_scale;

  auto size = cv::Size(width, height);
  images.resize(batch);
  read_images(files, batch);
  CHECK_EQ(images.size(), batch) 
	  << "images number be read into input buffer must be equal to batch";

  for (int index = 0; index < batch; ++index) {
    cv::Mat resize_image;
    if (size != images[index].size()) {
      cv::resize(images[index], resize_image, size);
    } else {
      images[index].copyTo(resize_image);
    }
    uint64_t data_in = 0u;
    size_t size_in = 0u;
    auto idx = get_index_zeros(input_tensor);
    idx[0] = (int)index;
    std::tie(data_in, size_in) = input_tensor_buffers[0]->data(idx);

    set_input_image(resize_image, (void*)data_in, input_fixed_scale);
  }
}

//yolov2 postprocess
static void postprocess_yolov2(const std::vector<vart::TensorBuffer*>& output_tensor_buffers) {
  auto output_tensor = output_tensor_buffers[0]->get_tensor();
  auto batch = output_tensor->get_shape().at(0);
  auto size = output_tensor_buffers.size();
  CHECK_EQ(size, 1) << "output_tensor_buffers.size() must be 1";

  conf_thresh = ENV_PARAM(CONF_THRESH );
  for (int batch_index = 0; batch_index < batch; ++batch_index) {
      uint64_t data_out = 0u;
      size_t size_out = 0u;
      auto idx = get_index_zeros(output_tensor);
      idx[0] = (int)batch_index;
      std::tie(data_out, size_out) = output_tensor_buffers[0]->data(idx);

      std::vector<std::vector<float>> nms_scores(20); 
      std::vector<std::vector<std::vector<float>>> nms_boxes(20);

      float* fpout = (float*)data_out;
      for(int i=0; i<13*13*5; i++) {
         if (fpout[ 13*13*5*4 + i*4] >= conf_thresh ) {
            nms_scores[ fpout[13*13*5*4 *2 + i*4 ]].emplace_back( fpout[ 13*13*5*4 *1 + i*4 ] );
            nms_boxes[  fpout[13*13*5*4 *2 + i*4 ]].emplace_back( std::vector<float>{  
                                                     fpout[13*13*5*4 *0 + i*4 + 0 ],
                                                     fpout[13*13*5*4 *0 + i*4 + 1 ],
                                                     fpout[13*13*5*4 *0 + i*4 + 2 ],
                                                     fpout[13*13*5*4 *0 + i*4 + 3 ]
                                                   });
         } 
      }
  
      std::vector<std::pair<int, int>> keep;
      for (int i=0; i<20; i++) {
         if (nms_scores[i].size()) {
            applyNMS(nms_boxes[i], nms_scores[i], nms_thresh, 0, i, keep);
         }
      }
  
      int retnum = keep.size();  
      for(int i=0; i<retnum; i++) {
         std::cout << float(keep[i].first) << "  " << nms_scores[keep[i].first][keep[i].second] << " : " 
                   << nms_boxes[ keep[i].first ][ keep[i].second][0] *images[batch_index].cols << " " 
                   << nms_boxes[ keep[i].first ][ keep[i].second][1] *images[batch_index].rows << " " 
                   << nms_boxes[ keep[i].first ][ keep[i].second][2] *images[batch_index].cols << " " 
                   << nms_boxes[ keep[i].first ][ keep[i].second][3] *images[batch_index].rows << "\n";
      }
  }
}

static int get_fix_point(const xir::Tensor* tensor) {
  CHECK(tensor->has_attr("fix_point"))
        << "get tensor fix_point error! has no fix_point attr, tensor name is "
        << tensor->get_name();
  return tensor->template get_attr<int>("fix_point");
}

static std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor) {
  auto ret = tensor->get_shape();
  std::fill(ret.begin(), ret.end(), 0);
  return ret;
}

void read_images(const std::vector<std::string>& files,
		                                        size_t batch) {
  for (auto index = 0u; index < batch; ++index) {
    const auto& file = files[index % files.size()];
    images[index] = cv::imread(file);
    CHECK(!images[index].empty()) << "cannot read image from " << file;
  }
}

static void set_input_image(const cv::Mat& image, void* data1, float scale) {
  float mean[3] = {0.0, 0.0, 0.0};
  signed char* data = (signed char*)data1;
  scale*=scale_0;
  for (int h = 0; h < image.rows; h++) {
    for (int w = 0; w < image.cols; w++) {
      for (int c = 0; c < 3; c++) {
        auto image_data = (image.at<cv::Vec3b>(h, w)[c] - mean[c]) * scale;
        image_data = std::max(std::min(image_data, 127.0f), -128.0f);
        data[h * image.cols * 3 + w * 3 + c] = (int)image_data;
      }
    }
  }
}

static float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1 - w1 / 2.0, x2 - w2 / 2.0);
  float right = min(x1 + w1 / 2.0, x2 + w2 / 2.0);
  return right - left;
}

static float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}

static void applyNMS(const vector<vector<float>>& boxes,
              const vector<float>& scores,
              const float nms,
              const float conf,
              int cls_index,
              vector<std::pair<int,int>>& res) {
  const size_t count = boxes.size();
  vector<pair<float, size_t>> order;
  for (size_t i = 0; i < count; ++i) {
    order.push_back({scores[i], i});
  }
  stable_sort(order.begin(), order.end(),
              [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
                return ls.first > rs.first;
              });
  vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
            [](auto& km) { return km.second; });
  vector<bool> exist_box(count, true);

  for (size_t _i = 0; _i < count; ++_i) {
    size_t i = ordered[_i];
    if (!exist_box[i]) continue;
    if (scores[i] < conf) {
      exist_box[i] = false;
      continue;
    }
    /* add a box as result */
    res.push_back(std::make_pair(cls_index, i));
    // cout << "nms push "<< i<<endl;
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = ordered[_j];
      if (!exist_box[j]) continue;
      float ovr = 0.0;
      ovr = cal_iou(boxes[j], boxes[i]);
      if (ovr >= nms) exist_box[j] = false;
    }
  }
}


