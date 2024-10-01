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
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/nnpp/yolovx.hpp>
#include <vitis/ai/yolovx.hpp>
extern int g_last_frame_id;

namespace vitis {
namespace ai {

struct YOLOVXAcc : public AccThread {
  YOLOVXAcc(std::string output_file)
      : AccThread(), of(output_file, std::ofstream::out), is_print(false) {
    dpu_result.frame_id = -1;
    of << "[";
  }

  virtual ~YOLOVXAcc() {
    of << "]";
    of.close();
  }

  static std::shared_ptr<YOLOVXAcc> instance(std::string output_file) {
    static std::weak_ptr<YOLOVXAcc> the_instance;
    std::shared_ptr<YOLOVXAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<YOLOVXAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (YOLOvXResult*)dpu_result.result_ptr.get();
    for (auto& r : result->bboxes) {
      if (is_print) {
        of << ",\n";
      } else {
        is_print = true;
      }
      json_object* objitem = json_object_new_object();
      json_object* str_image_id = json_object_new_int(dpu_result.frame_id - 1);
      json_object_object_add(objitem, "image_id", str_image_id);
      json_object* str_category_id = json_object_new_int(r.label + 1);
      json_object_object_add(objitem, "category_id", str_category_id);
      json_object* str_score = json_object_new_double(r.score);
      json_object_object_add(objitem, "score", str_score);
      json_object* bbox_array = json_object_new_array();
      json_object_array_add(bbox_array, json_object_new_double(r.box[0]));
      json_object_array_add(bbox_array, json_object_new_double(r.box[1]));
      json_object_array_add(bbox_array,
                            json_object_new_double(r.box[2] - r.box[0]));
      json_object_array_add(bbox_array,
                            json_object_new_double(r.box[3] - r.box[1]));
      json_object_object_add(objitem, "bbox", bbox_array);
      of << json_object_to_json_string(objitem);
      json_object_put(objitem);
    }
  }

  virtual int run() override {
    if (g_last_frame_id == int(dpu_result.frame_id)) return -1;
    if (getQueue()->pop(dpu_result, std::chrono::milliseconds(5000))) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "[" << name() << "] process result id :" << dpu_result.frame_id
          << ", dpu queue size " << getQueue()->size();

      process_result(dpu_result);
    }
    return 0;
  }

  DpuResultInfo dpu_result;
  std::ofstream of;
  bool is_print;
};  // namespace ai

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  return vitis::ai::main_for_accuracy_demo(
      argc, argv,
      [&] { return vitis::ai::YOLOvX::create(argv[1] + std::string("_acc")); },
      vitis::ai::YOLOVXAcc::instance(argv[3]), 2);
}
