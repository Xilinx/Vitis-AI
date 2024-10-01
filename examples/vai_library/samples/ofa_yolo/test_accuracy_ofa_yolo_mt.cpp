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
#include <vitis/ai/nnpp/ofa_yolo.hpp>
#include <vitis/ai/ofa_yolo.hpp>

using namespace std;
namespace vitis {
namespace ai {

struct OFAYOLOAcc : public AccThread {
  OFAYOLOAcc(std::string output_file)
      : AccThread(), of(output_file, std::ofstream::out) {
    dpu_result.frame_id = -1;
    of << "[" << endl;
    category_ids = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
                    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                    80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
  }

  virtual ~OFAYOLOAcc() {
    of.seekp(-2L, ios::end);
    of << endl << "]" << endl;
    of.close();
  }

  static std::shared_ptr<OFAYOLOAcc> instance(std::string output_file) {
    static std::weak_ptr<OFAYOLOAcc> the_instance;
    std::shared_ptr<OFAYOLOAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<OFAYOLOAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (OFAYOLOResult*)dpu_result.result_ptr.get();
    for (auto& box : result->bboxes) {
      float xmin = box.x * dpu_result.w;
      float ymin = box.y * dpu_result.h;
      float width = box.width * dpu_result.w;
      float height = box.height * dpu_result.h;
      if (xmin < 0) xmin = 1;
      if (ymin < 0) ymin = 1;
      float confidence = box.score;
      of << fixed << setprecision(0)
         << "{\"image_id\":" << atoi(dpu_result.single_name.c_str())
         << ", \"category_id\":" << category_ids[box.label] << ", \"bbox\":["
         << fixed << setprecision(6) << xmin << ", " << ymin << ", " << width
         << ", " << height << "], \"score\":" << confidence << "}," << endl;
    }
  }

  virtual int run() override {
    if (g_last_frame_id == int(dpu_result.frame_id)) return -1;
    if (queue_->pop(dpu_result, std::chrono::milliseconds(50000))) {
      process_result(dpu_result);
    }
    return 0;
  }

  DpuResultInfo dpu_result;
  std::ofstream of;
  vector<int> category_ids;
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  return vitis::ai::main_for_accuracy_demo(
      argc, argv,
      [&] { return vitis::ai::OFAYOLO::create(string(argv[1]) + "_acc"); },
      vitis::ai::OFAYOLOAcc::instance(argv[3]), 2);
}
