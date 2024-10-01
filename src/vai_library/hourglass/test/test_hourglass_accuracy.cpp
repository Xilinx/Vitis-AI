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
#include <json-c/json.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/hourglass.hpp>
#include <vitis/ai/nnpp/hourglass.hpp>
using namespace std;
using namespace cv;
using PosePoint = vitis::ai::HourglassResult::PosePoint;
string g_output_file;
extern int g_last_frame_id;

namespace vitis {
namespace ai {

struct HourglassReadImages : public ReadImagesThread {
  HourglassReadImages(const std::string& images_list_file, queue_t* queue)
      : ReadImagesThread(images_list_file, queue) {}

  static std::shared_ptr<HourglassReadImages> instance() {
    static std::weak_ptr<HourglassReadImages> the_instance;
    std::shared_ptr<HourglassReadImages> ret;
    if (the_instance.expired()) {
      auto read_queue = std::unique_ptr<queue_t>{new queue_t{50}};
      ret = std::make_shared<HourglassReadImages>("", read_queue.get());
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  virtual int run() override {
    std::ifstream fs(images_list_file_);
    std::string line;
    std::string single_name;
    while (getline(fs, line)) {
      Mat image(256, 256, CV_8UC3);
      FILE* fpr = fopen(line.c_str(), "rb");
      if (fpr == NULL) {
        std::cerr << "cannot read image: " << line;
        continue;
      }
      uchar* pData = image.data;
      for (int i = 0; i < 256 * 256; i++) {
        auto fread_size = fread(&pData[3 * i + 2], sizeof(char), 1, fpr);
        CHECK(fread_size == 1) << "fread size error! ";
        fread_size = fread(&pData[3 * i + 1], sizeof(char), 1, fpr);
        CHECK(fread_size == 1) << "fread size error! ";
        fread_size = fread(&pData[3 * i + 0], sizeof(char), 1, fpr);
        CHECK(fread_size == 1) << "fread size error! ";
      }
      fclose(fpr);

      single_name = get_single_name(line);
      int w = image.cols;
      int h = image.rows;
      while (!queue_->push(FrameInfo{++frame_id_, image, single_name, w, h},
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

struct HourglassAccThread : public AccThread {
  static std::shared_ptr<HourglassAccThread> instance() {
    static std::weak_ptr<HourglassAccThread> the_instance;
    std::shared_ptr<HourglassAccThread> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<HourglassAccThread>();
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  virtual int run() override {
    std::ofstream of(g_output_file, std::ofstream::out);
    of << "[";
    bool is_print = false;
    while (1) {
      if (is_print) {
        of << ", ";
      } else
        is_print = true;

      DpuResultInfo dpu_result;
      if (!queue_->pop(dpu_result, std::chrono::milliseconds(5000))) {
        LOG(INFO) << "pop dpu result time out";
        return 0;
      }
      void* r = dpu_result.result_ptr.get();
      auto result = (HourglassResult*)r;
      json_object* str_imageid = json_object_new_string(
          // dpu_result.single_name.substr(0, dpu_result.single_name.size() -
          // 4).c_str());
          dpu_result.single_name.c_str());
      json_object* value = json_object_new_object();
      json_object_object_add(value, "image_id", str_imageid);
      json_object* pose_array = json_object_new_array();
      // vector<int> a = {5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 0, 1};

      PosePoint posePoint;
      vector<PosePoint> jo(16, posePoint);
      vector<PosePoint> j = result->poses;
      for (size_t i = 0; i < jo.size(); ++i) {
        // jo[i] = j[a[i]];
        json_object* point_array = json_object_new_array();
        jo[i] = j[i];
        json_object_array_add(point_array,
                              json_object_new_double(jo[i].point.x / 4));
        json_object_array_add(point_array,
                              json_object_new_double(jo[i].point.y / 4));
        json_object_array_add(point_array, json_object_new_int(jo[i].type));
        json_object_array_add(pose_array, point_array);
      }

      json_object_object_add(value, "joint_self", pose_array);
      // cout << json_object_to_json_string(value);
      of << json_object_to_json_string(value);
      json_object_put(value);

      LOG_IF(INFO, 0) << "test hourglass, queue size: " << queue_->size()
                      << " dpu_result id: " << dpu_result.frame_id;
      if (g_last_frame_id == int(dpu_result.frame_id)) {
        of << "]";
        of.close();
        LOG(INFO) << "test hourglass accuracy done! byebye " << queue_->size();
        exit(0);
      }
    }
    return 0;
  }
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  string model = argv[1];
  g_output_file = argv[3];
  return vitis::ai::main_for_accuracy_demo(
      argc, argv, [model] { return vitis::ai::Hourglass::create(model); },
      vitis::ai::HourglassAccThread::instance(), 2,
      vitis::ai::HourglassReadImages::instance());
}
