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
#include <iostream>
#include <fstream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/classification.hpp>
#include <vitis/ai/nnpp/classification.hpp>
using namespace std;
using namespace cv;
string g_output_file;
extern int g_last_frame_id;

namespace vitis {
namespace ai {

struct ClassificationAccThread : public AccThread {

  static std::shared_ptr<ClassificationAccThread> instance() {
    static std::weak_ptr<ClassificationAccThread> the_instance;
    std::shared_ptr<ClassificationAccThread> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<ClassificationAccThread>();
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
    return ret;
  }

  virtual int run() override {
    std::cout<<g_output_file<<endl;
    std::ofstream of(g_output_file, std::ofstream::out);

    while(1){
      DpuResultInfo dpu_result;
      if (!queue_->pop(dpu_result, std::chrono::milliseconds(5000))) {
        cout<<" get dpu_result timeout"<<endl; 
        exit(0);
      }
      void *r = dpu_result.result_ptr.get();
      auto res = (ClassificationResult*)r;
      for (size_t j = 0; j < res->scores.size(); ++j) {
      int index = res->scores[j].index;
      cout << dpu_result.single_name << " " << index << " "
           <<dpu_result.w << " "<<dpu_result.h << " "
           << res->scores[j].score << " " << endl;
      of << "/"<<dpu_result.single_name << " " << index << endl;
      }
      LOG_IF(INFO, 0)
      <<"test class, queue size: "<<queue_->size()
      <<" dpu_result id: "<<dpu_result.frame_id;
      if(g_last_frame_id == int(dpu_result.frame_id)){
        of.close();
        LOG(INFO)
        <<"test class accuracy done! byebye "<<queue_->size();
        exit(0);
      }
    }
    return 0;
  }
};

} // namespace ai
} // namespace vitis

int main(int argc, char *argv[]) {
  string model = argv[1] + string("_acc");
  g_output_file = argv[3];
  return vitis::ai::main_for_accuracy_demo(
      argc, argv,
      [model] {
        return vitis::ai::Classification::create(model);
      },
      vitis::ai::ClassificationAccThread::instance(),
      2);
}

