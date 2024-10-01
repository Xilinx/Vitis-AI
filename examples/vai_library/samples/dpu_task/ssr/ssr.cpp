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
#include "./ssr.hpp"

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/profiling.hpp>

DEF_ENV_PARAM(DEBUG_SSR, "0");
DEF_ENV_PARAM(DUMP_SSR, "0");

using namespace std;
using namespace cv;

namespace vitis {
namespace ai {
class SSRImp : public SSR {
 public:
  SSRImp(const std::string& model_name);

 public:
  virtual ~SSRImp();
  virtual void run(const std::vector<cv::Mat>& imgs) override;
  virtual std::vector<cv::Mat> get_result() override;
  virtual size_t get_input_batch() override;
  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;

 private:
  void ssr_run(const vector<cv::Mat>& input_images);

 public:
  std::unique_ptr<vitis::ai::DpuTask> task_;
  vector<vitis::ai::library::InputTensor> inputs_;
  vector<vitis::ai::library::OutputTensor> outputs_;
};

SSR::SSR(const std::string& model_name) {}

SSR::~SSR() {}

std::unique_ptr<SSR> SSR::create(const std::string& model_name) {
  return std::unique_ptr<SSRImp>(new SSRImp(model_name));
//  return std::make_unique<SSRImp>(model_name);
}

SSRImp::SSRImp(const std::string& model_name): SSR(model_name) {
  task_ = vitis::ai::DpuTask::create(model_name);
  task_->setMeanScaleBGR({127.5, 127.5, 127.5},
                        {0.007843, 0.007843, 0.007843});
}

SSRImp::~SSRImp() {}

// run the fadnet
void SSRImp::ssr_run(const std::vector<cv::Mat>& input_image) {
  vector<cv::Mat> mats;
  auto input_tensor = task_->getInputTensor(0u)[0];
  int sWidth = input_tensor.width;
  int sHeight = input_tensor.height;

  __TIC__(SSR_RESIZE)
  for(size_t i = 0; i < input_tensor.batch; ++i) {
    if(input_image[i].cols == sWidth && input_image[i].rows == sHeight)
      mats.push_back(input_image[i]);
    else{
      cv::Mat img_res;
      resize(input_image[i], img_res, cv::Size(sWidth, sHeight));
      mats.push_back(img_res);
    }
  }
  __TOC__(SSR_RESIZE)

  __TIC__(SSR_SET_IMG)
  task_->setImageRGB(mats);
  //CHECK(
  //ifstream("./input.bin").read((char*)input_tensor.get_data(0), input_tensor.size/input_tensor.batch).good())
  //  << "fail to read file";
  __TOC__(SSR_SET_IMG)

  __TIC__(SSR_DPU)
  task_->run(0u);
  __TOC__(SSR_DPU)
}

void SSRImp::run(const std::vector<cv::Mat>& imgs) {
  ssr_run(imgs);
  return;
}

std::vector<cv::Mat> SSRImp::get_result() {
  std::vector<cv::Mat> rets;
  __TIC__(SSR_POST_ARM)
  auto output_tensor = task_->getOutputTensor(0u)[0];
  float scale = vitis::ai::library::tensor_scale(output_tensor);
  if (ENV_PARAM(DEBUG_SSR)) {
    LOG(INFO) << "output info: width: " << output_tensor.width
  	    << " height: "  << output_tensor.height
	    << " channel: "  << output_tensor.channel
	    << " batch: "  << output_tensor.batch
	    << " scale: "  << scale
	    ;
  }

  for(size_t i = 0; i < output_tensor.batch; ++i) {
    int8_t* output_ptr = (int8_t*)output_tensor.get_data(i);
    Mat res(Size(output_tensor.width, output_tensor.height), CV_8UC3);
    for(size_t h = 0; h < output_tensor.height; ++h) {
      for(size_t w = 0; w < output_tensor.width; ++w) {
        for(size_t c = 0; c < output_tensor.channel; ++c) {
          int pos = (h * output_tensor.width + w) * output_tensor.channel + c;
	  float temp = output_ptr[pos] * scale;
	  if(temp > 1.0) temp = 1.0;
	  else if (temp < -1.0) temp = -1.0;
          res.at<Vec3b>(h,w)[2-c] = (uint8_t)(temp*127.5 + 127.5);
	}
      }
    }
    rets.push_back(res);
    if (ENV_PARAM(DEBUG_SSR)) {
      LOG(INFO) << "batch success";
    }
  }
  __TOC__(SSR_POST_ARM)
  return rets;
}

size_t SSRImp::get_input_batch() { return task_->get_input_batch(0, 0); }
int SSRImp::getInputWidth() const {
  return task_->getInputTensor(0u)[0].width;
}
int SSRImp::getInputHeight() const {
  return task_->getInputTensor(0u)[0].height;
}

}  // namespace ai
}
