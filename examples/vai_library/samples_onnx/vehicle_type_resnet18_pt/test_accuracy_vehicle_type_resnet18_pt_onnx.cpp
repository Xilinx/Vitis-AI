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
#include "./vehicle_type_resnet18_pt_onnx.hpp"

using namespace std;
using namespace cv;
string g_output_file;
extern int g_last_frame_id;

namespace vitis {
namespace ai {
struct VehicleTypeResnet18PtOnnxAccThread : public AccThread {
  VehicleTypeResnet18PtOnnxAccThread(std::string output_file)
      : AccThread(), of(output_file, std::ofstream::out) {
    dpu_result.frame_id = -1;
  }
  virtual ~VehicleTypeResnet18PtOnnxAccThread() { of.close(); }
  static std::shared_ptr<VehicleTypeResnet18PtOnnxAccThread> instance(
      std::string output_file) {
    static std::weak_ptr<VehicleTypeResnet18PtOnnxAccThread> the_instance;
    std::shared_ptr<VehicleTypeResnet18PtOnnxAccThread> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<VehicleTypeResnet18PtOnnxAccThread>(output_file);
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  virtual int run() override {
    if (g_last_frame_id == int(dpu_result.frame_id)) return -1;
    if (getQueue()->pop(dpu_result, std::chrono::milliseconds(5000))) {
      auto res = (VehicleTypeResnet18PtOnnxResult*)dpu_result.result_ptr.get();
      for (auto score : res->scores) {
        of << "/" << dpu_result.single_name << " " << score.index << endl;
      }
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }
  DpuResultInfo dpu_result;
  std::ofstream of;
};
}  // namespace ai
}  // namespace vitis

int main(int argc, char* argv[]) {
  if (argc < 4) {
    cout << "Please input a model name as the first param!" << endl;
    cout << "Please input your image path list as the second param!" << endl;
    cout << "The third param is a txt to store results!" << endl;
    cout << "The fourth param is thread nums, eg: '-t 4', default single "
            "thread if not filled "
         << endl;
  }
  // string model = argv[1] + string("_acc");
  string model = argv[1];
  g_output_file = argv[3];
  return vitis::ai::main_for_accuracy_demo(
      argc, argv, [model] { return VehicleTypeResnet18PtOnnx::create(model); },
      vitis::ai::VehicleTypeResnet18PtOnnxAccThread::instance(argv[3]), 2);
}
