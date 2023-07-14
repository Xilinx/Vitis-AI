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
#include <vitis/ai/nnpp/rcan.hpp>
#include <vitis/ai/rcan.hpp>
extern int g_last_frame_id;
extern int GLOBAL_ENABLE_ROUND_SETINPUT;

std::string model_name;
std::string output_name;
bool is_first = true;
using namespace std;
namespace vitis {
namespace ai {

struct RcanAcc : public AccThread {
  RcanAcc() : AccThread() { dpu_result.frame_id = -1; }

  virtual ~RcanAcc() {}

  static std::shared_ptr<RcanAcc> instance() {
    static std::weak_ptr<RcanAcc> the_instance;
    std::shared_ptr<RcanAcc> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<RcanAcc>();
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  void process_result(DpuResultInfo dpu_result) {
    auto result = (RcanResult*)dpu_result.result_ptr.get();
    auto image_name = dpu_result.single_name;
    cv::imwrite(output_name + "/" +
                    image_name.substr(0, image_name.size() - 4) + "_result.png",
                result->feat);
  }

  virtual int run() override {
    if (is_first) {
      is_first = false;
    }
    if (g_last_frame_id == int(dpu_result.frame_id)) {
      exit(0);
    }
    if (queue_->pop(dpu_result, std::chrono::milliseconds(50000))) {
      process_result(dpu_result);
    }
    return 0;
  }

  DpuResultInfo dpu_result;
};

}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  model_name = argv[1];
  output_name = argv[3];
  GLOBAL_ENABLE_ROUND_SETINPUT = 1;
  return vitis::ai::main_for_accuracy_demo(
      argc, argv, [&] { return vitis::ai::Rcan::create(model_name); },
      vitis::ai::RcanAcc::instance(), 2);
}
