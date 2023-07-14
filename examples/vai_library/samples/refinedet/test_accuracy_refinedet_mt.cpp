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
#include <vitis/ai/refinedet.hpp>
extern int g_last_frame_id;
extern int GLOBAL_ENABLE_C_SOFTMAX;
static std::map<std::string, std::vector<std::string>> label_maps{
    {"caffe_model", {"", "person", ""}},
    {"refinedet_VOC_tf",
     {"none", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
      "bus",  "car",       "cat",       "chair",  "cow",         "diningtable",
      "dog",  "horse",     "motorbike", "person", "pottedplant", "sheep",
      "sofa", "train",     "tvmonitor"}}};
std::vector<std::string> label_map;
namespace vitis {
namespace ai {
struct refinedetAcc : public AccThread {
  refinedetAcc(std::string output_file)
      : AccThread(), of(output_file, std::ofstream::out) {
    dpu_result.frame_id = -1;
  }

  virtual ~refinedetAcc() { of.close(); }

  static std::shared_ptr<refinedetAcc> instance(std::string output_file) {
    static std::weak_ptr<refinedetAcc> the_instance;
    std::shared_ptr<refinedetAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<refinedetAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (RefineDetResult*)dpu_result.result_ptr.get();
    for (auto& it : result->bboxes) {
      of << dpu_result.single_name.substr(0, dpu_result.single_name.size() - 4)
         << " " << label_map[it.label] << " " << it.score << " "
         << it.x * dpu_result.w << " " << it.y * dpu_result.h << " "
         << (it.x + it.width) * dpu_result.w << " "
         << (it.y + it.height) * dpu_result.h << std::endl;
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
};  // namespace ai

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  std::string model_name = argv[1];
  GLOBAL_ENABLE_C_SOFTMAX = 2;

  if (label_maps.count(model_name)) {
    label_map = label_maps[model_name];
  } else {
    label_map = label_maps["caffe_model"];
  }
  return vitis::ai::main_for_accuracy_demo(
      argc, argv,
      [&] { return vitis::ai::RefineDet::create(model_name + "_acc"); },
      vitis::ai::refinedetAcc::instance(argv[3]), 2);
}
