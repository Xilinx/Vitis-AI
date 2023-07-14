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

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/facedetect.hpp>
#include <vitis/ai/nnpp/facedetect.hpp>
#include <xir/graph/graph.hpp>
extern int g_last_frame_id;

extern int GLOBAL_ENABLE_C_SOFTMAX;

// A struct that can storage model info
struct ModelInfo {
  int w;
  int h;
} model_info_facedetect;

namespace vitis {
namespace ai {

static ModelInfo get_model_zise(const std::string& filename) {
  std::string model_name = "/usr/share/vitis_ai_library/models/" + filename +
                           "/" + filename + ".xmodel";
  auto graph = xir::Graph::deserialize(model_name);
  auto root = graph->get_root_subgraph();
  auto children = root->children_topological_sort();
  if (children.empty()) {
    std::cout << "no subgraph" << std::endl;
  }
  for (auto c : children) {
    CHECK(c->has_attr("device"));
    auto device = c->get_attr<std::string>("device");
    if (device == "DPU") {
      auto inputs = c->get_sorted_input_tensors();
      if(inputs.size()) {
        auto input = inputs[0];
        // for (auto input : inputs) {
        int height = input->get_shape().at(1);
        int width = input->get_shape().at(2);
        //        std::cout << "model width: " << width  << " model heigth: " <<
        //        height << std::endl;
        return ModelInfo{width, height};
      }
    }
  }
  return {0, 0};
}

static std::vector<std::string> split(const std::string& s,
                                      const std::string& delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.push_back(s.substr(pos, len - pos));
      break;
    }
    elems.push_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
}

struct FaceDetectReadImages : public ReadImagesThread {
  FaceDetectReadImages(const std::string& images_list_file, queue_t* queue)
      : ReadImagesThread(images_list_file, queue) {}

  static std::shared_ptr<FaceDetectReadImages> instance() {
    static std::weak_ptr<FaceDetectReadImages> the_instance;
    std::shared_ptr<FaceDetectReadImages> ret;
    if (the_instance.expired()) {
      auto read_queue = std::unique_ptr<queue_t>{new queue_t{50}};
      ret = std::make_shared<FaceDetectReadImages>("", read_queue.get());
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  cv::Mat image_pre_resize(cv::Mat image) {
    int width = model_info_facedetect.w;
    int height = model_info_facedetect.h;

    cv::Mat canvas(width, height, CV_8UC3, cv::Scalar(0, 0, 0));
    float scale = (float)image.cols / (float)image.rows;
    float network_scale = (float)width / (float)height;
    if (scale >= network_scale) {
      resize(canvas, canvas,
             cv::Size(image.cols, ceil((float)image.cols / network_scale)));
    } else {
      resize(canvas, canvas,
             cv::Size(ceil((float)image.rows * network_scale), image.rows));
    }

    image.copyTo(canvas(cv::Rect_<int>(0, 0, image.cols, image.rows)));

    return canvas;
  }

  virtual int run() override {
    std::ifstream fs(images_list_file_);
    std::string line;
    std::string single_name;
    while (getline(fs, line)) {
      auto image = cv::imread(line);
      if (image.empty()) {
        std::cerr << "cannot read image: " << line;
        continue;
      }
      auto dirName = split(line, "/")[0];
      auto namesp = split(line, dirName + "/");
      single_name = split(namesp[1], ".")[0];
      auto image_resize = image_pre_resize(image);
      int w = image_resize.cols;
      int h = image_resize.rows;
      while (
          !queue_->push(FrameInfo{++frame_id_, image_resize, single_name, w, h},
                        std::chrono::milliseconds(500))) {
        --frame_id_;
        if (is_stopped()) {
          return -1;
        }
      }
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "push image frame_id " << frame_id_ << ",read images queue size "
          << queue_->size();
    }
    g_last_frame_id = frame_id_;
    return -1;
  }
};

struct FaceDetectAcc : public AccThread {
  FaceDetectAcc(std::string output_file)
      : AccThread(), of(output_file, std::ofstream::out) {
    dpu_result.frame_id = -1;
  }

  virtual ~FaceDetectAcc() { of.close(); }

  static std::shared_ptr<FaceDetectAcc> instance(std::string output_file) {
    static std::weak_ptr<FaceDetectAcc> the_instance;
    std::shared_ptr<FaceDetectAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<FaceDetectAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (FaceDetectResult*)dpu_result.result_ptr.get();
    auto image_name = split(dpu_result.single_name, ".")[0];
    std::cout << image_name << std::endl;
    of << image_name << std::endl;
    of << result->rects.size() << std::endl;
    for (const auto& r : result->rects) {
      std::cout << " " << r.score << " "  //
                << r.x << " "             //
                << r.y << " "             //
                << r.width << " "         //
                << r.height << " "        //
                << std::endl;
      of << r.x * dpu_result.w << " " << r.y * dpu_result.h << " "
         << r.width * dpu_result.w << " " << r.height * dpu_result.h << " "
         << r.score << " " << std::endl;
    }
  }

  virtual int run() override {
    if (g_last_frame_id == int(dpu_result.frame_id)) return -1;
    if (queue_->pop(dpu_result, std::chrono::milliseconds(50000)))
      process_result(dpu_result);
    return 0;
  }

  DpuResultInfo dpu_result;
  std::ofstream of;
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  GLOBAL_ENABLE_C_SOFTMAX = 2;
  model_info_facedetect = vitis::ai::get_model_zise(argv[1]);
  return vitis::ai::main_for_accuracy_demo(
      argc, argv, [&] { return vitis::ai::FaceDetect::create(argv[1]); },
      vitis::ai::FaceDetectAcc::instance(argv[3]), 2,
      vitis::ai::FaceDetectReadImages::instance());
}
