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
#include <vitis/ai/nnpp/openpose.hpp>
#include <vitis/ai/openpose.hpp>
using namespace std;
using Peak = std::tuple<int, float, cv::Point2f>;
using PosePoint = vitis::ai::OpenPoseResult::PosePoint;
string g_output_file;
extern int g_last_frame_id;

namespace vitis {
namespace ai {

struct OpenPoseAccThread : public AccThread {
  static std::shared_ptr<OpenPoseAccThread> instance() {
    static std::weak_ptr<OpenPoseAccThread> the_instance;
    std::shared_ptr<OpenPoseAccThread> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<OpenPoseAccThread>();
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  virtual int run() override {
    std::cout << g_output_file << endl;
    std::ofstream of(g_output_file, std::ofstream::out);
    of << "[";
    bool is_print = false;
    while (1) {
      if (is_print) {
        of << ", ";
      } else
        is_print = true;

      DpuResultInfo dpu_result;
      if (!queue_->pop(dpu_result, std::chrono::milliseconds(500))) {
        return 0;
      }
      void* r = dpu_result.result_ptr.get();
      auto result = (OpenPoseResult*)r;
      json_object* str_imageid = json_object_new_string(
          dpu_result.single_name.substr(0, dpu_result.single_name.size() - 4)
              .c_str());
      json_object* value = json_object_new_object();
      json_object_object_add(value, "image_id", str_imageid);
      json_object* human = json_object_new_object();
      vector<int> a = {5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 0, 1};
      for (size_t k = 1; k < result->poses.size(); ++k) {
        PosePoint posePoint;
        vector<PosePoint> jo(14, posePoint);
        vector<PosePoint> j = result->poses[k];
        string humanid = "human";
        humanid += to_string(result->poses.size() - k);
        json_object* point_array = json_object_new_array();
        for (int i = 0; i < 14; ++i) {
          jo[i] = j[a[i]];
          json_object_array_add(point_array,
                                json_object_new_double(jo[i].point.x));
          json_object_array_add(point_array,
                                json_object_new_double(jo[i].point.y));
          json_object_array_add(point_array, json_object_new_int(jo[i].type));
        }
        json_object_object_add(human, humanid.c_str(), point_array);
      }

      json_object_object_add(value, "keypoint_annotations", human);
      of << json_object_to_json_string(value);
      json_object_put(value);

      LOG_IF(INFO, 1) << "test openpsoe, queue size: " << queue_->size()
                      << " dpu_result id: " << dpu_result.frame_id;
      if (g_last_frame_id == int(dpu_result.frame_id)) {
        of << "]";
        of.close();
        LOG(INFO) << "test openpsoe accuracy done! byebye " << queue_->size();
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
      argc, argv, [model] { return vitis::ai::OpenPose::create(model); },
      vitis::ai::OpenPoseAccThread::instance(), 2);
}
