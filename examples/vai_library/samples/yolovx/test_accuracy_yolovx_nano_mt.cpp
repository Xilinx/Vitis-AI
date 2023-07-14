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
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/nnpp/yolovx.hpp>
#include <vitis/ai/yolovx.hpp>
extern int g_last_frame_id;
std::string model_name;
bool is_first = true;
using namespace std;
namespace vitis {
namespace ai {

vector<int> coco_id_map_dict() {
  vector<int> category_ids;
  category_ids = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15,
                  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
                  33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                  62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                  80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
  return category_ids;
}

int imagename_to_id(string imagename) {
  int idx1 = imagename.size();
  int idx2 = imagename.find_last_of('_');
  string id = imagename.substr(idx2 + 1, idx1 - idx2);
  int image_id = atoi(id.c_str());
  return image_id;
}

struct YolovXAcc : public AccThread {
  YolovXAcc(std::string output_file)
      : AccThread(), of(output_file, std::ofstream::out) {
    dpu_result.frame_id = -1;
  }

  virtual ~YolovXAcc() { of.close(); }

  static std::shared_ptr<YolovXAcc> instance(std::string output_file) {
    static std::weak_ptr<YolovXAcc> the_instance;
    std::shared_ptr<YolovXAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<YolovXAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto ccoco_id_map_dict = coco_id_map_dict();
    auto result = (YOLOvXResult*)dpu_result.result_ptr.get();
    for (auto& result : result->bboxes) {
      auto& box = result.box;
      int label = result.label;
      float xmin = box[0];
      float ymin = box[1];
      float xmax = box[2];
      float ymax = box[3];
      if (xmin < 0) xmin = 0;
      if (ymin < 0) ymin = 0;
      float confidence = result.score;
      of << fixed << setprecision(0)
         << "{\"image_id\":" << imagename_to_id(dpu_result.single_name)
         << ", \"category_id\":" << ccoco_id_map_dict[label] << ", \"bbox\":["
         << fixed << setprecision(6) << xmin << ", " << ymin << ", "
         << xmax - xmin << ", " << ymax - ymin << "], \"score\":" << confidence
         << "}," << endl;
    }
  }

  virtual int run() override {
    if (is_first) {
      of << "[" << endl;
      is_first = false;
    }
    if (g_last_frame_id == int(dpu_result.frame_id)) {
      of.seekp(-2L, ios::end);
      of << endl << "]" << endl;
      exit(0);
    }
    if (queue_->pop(dpu_result, std::chrono::milliseconds(50000))) {
      process_result(dpu_result);
    }
    return 0;
  }

  DpuResultInfo dpu_result;
  std::ofstream of;
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  model_name = argv[1];
  return vitis::ai::main_for_accuracy_demo(
      argc, argv,
      [&] { return vitis::ai::YOLOvX::create(model_name + "_acc"); },
      vitis::ai::YolovXAcc::instance(argv[3]), 2);
}
