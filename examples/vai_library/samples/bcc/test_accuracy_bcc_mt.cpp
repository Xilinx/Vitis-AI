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
#include <vitis/ai/bcc.hpp>
#include <vitis/ai/demo_accuracy.hpp>
extern int g_last_frame_id;

namespace vitis {
namespace ai {

struct BCCAcc : public AccThread {
  BCCAcc(std::string output_file)
      : AccThread(), of(output_file, std::ios_base::out) {
    dpu_result.frame_id = -1;
  }

  virtual ~BCCAcc() { of.close(); }

  static std::shared_ptr<BCCAcc> instance(std::string output_file) {
    static std::weak_ptr<BCCAcc> the_instance;
    std::shared_ptr<BCCAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<BCCAcc>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (BCCResult*)dpu_result.result_ptr.get();
    of << result->count << "\n";
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
  std::string model_name = argv[1];
  return vitis::ai::main_for_accuracy_demo(
      argc, argv, [&] { return vitis::ai::BCC::create(model_name); },
      vitis::ai::BCCAcc::instance(argv[3]), 2);
}
