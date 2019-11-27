/*
 * Copyright 2019 Xilinx Inc.
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
#include "./tfssd_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xilinx/ai/env_config.hpp>
#include <xilinx/ai/profiling.hpp>

#include <iostream>
#include <fstream>

// change this value to 1, if you want to test importing preprocess data from some file directly
#define TEST_IMPORT_DATA  0

using namespace std;
namespace xilinx {
namespace ai {
DEF_ENV_PARAM(ENABLE_SSD_DEBUG, "0");

TFSSDImp::TFSSDImp(const std::string &model_name, bool need_preprocess)
    : xilinx::ai::TConfigurableDpuTask<TFSSD>(model_name, need_preprocess),
      processor_{xilinx::ai::TFSSDPostProcess::create(
          configurable_dpu_task_->getInputTensor()[0],
          configurable_dpu_task_->getOutputTensor()[0],
          configurable_dpu_task_->getConfig(),
	  configurable_dpu_task_->get_dpu_meta_info()
	  )} {}

TFSSDImp::~TFSSDImp() {}

#if TEST_IMPORT_DATA > 0
void myreadfile(int8_t*conf, int size1, std::string filename)
{
   ifstream Tin;  Tin.open(filename, ios_base::in|ios_base::binary);
   if(!Tin)  {      cout<<"Can't open the file!";      return ;  }
   for(int i=0; i<size1; i++)    Tin.read( (char*)conf+i, 1);
}
#endif

TFSSDResult TFSSDImp::run(const cv::Mat &input_img) {
  cv::Mat img;
  auto size = cv::Size(getInputWidth(), getInputHeight());

  if (size != input_img.size()) {
    cv::resize(input_img, img, size, 0, 0, cv::INTER_LINEAR);
  } else {
    img = input_img;
  }
  __TIC__(SSD_total)
  __TIC__(SSD_setimg)

  configurable_dpu_task_->setInputImageRGB(img);

#if TEST_IMPORT_DATA > 0
  int8_t* conf_c = (int8_t*)configurable_dpu_task_->getInputTensor()[0][0].data;
  myreadfile( conf_c, 270000, "/home/somedir/preprocessor.bin");
#endif

  __TOC__(SSD_setimg)
  __TIC__(SSD_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(SSD_dpu)

  __TIC__(SSD_post)
  auto results = processor_->ssd_post_process();
  __TOC__(SSD_post)

  __TOC__(SSD_total)
  return results;
}

} // namespace ai
} // namespace xilinx
