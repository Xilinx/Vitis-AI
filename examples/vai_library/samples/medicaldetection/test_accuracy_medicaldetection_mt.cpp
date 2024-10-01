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
#include <vitis/ai/medicaldetection.hpp>
extern int g_last_frame_id;

namespace vitis {
namespace ai {

std::vector<std::string> classTypes = {"BE", "cancer", "HGD", "polyp",
                                       "suspicious"};
struct MedicalDetectionAcc : public AccThread {
  MedicalDetectionAcc(std::string output_file)
      : AccThread(), of(output_file, std::ofstream::out) {
    dpu_result.frame_id = -1;
  }

  virtual ~MedicalDetectionAcc() { of.close(); }

  static std::shared_ptr<MedicalDetectionAcc> instance(
      std::string output_file) {
    static std::weak_ptr<MedicalDetectionAcc> the_instance;
    std::shared_ptr<MedicalDetectionAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<MedicalDetectionAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (MedicalDetectionResult*)dpu_result.result_ptr.get();
    for (auto& res : result->bboxes) {
      of << dpu_result.single_name.substr(0, dpu_result.single_name.size() - 4)
         << " " << classTypes[res.label - 1] << " " << res.score << " "
         << res.x * dpu_result.w << " " << res.y * dpu_result.h << " "
         << (res.x + res.width) * dpu_result.w << " "
         << (res.y + res.height) * dpu_result.h << std::endl;
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
  return vitis::ai::main_for_accuracy_demo(
      argc, argv,
      [&] {
        return vitis::ai::MedicalDetection::create(argv[1] +
                                                   std::string("_acc"));
      },
      vitis::ai::MedicalDetectionAcc::instance(argv[3]), 2);
}
