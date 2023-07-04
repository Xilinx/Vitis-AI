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
#ifndef DEEPHI_Unet2D_HPP_
#define DEEPHI_Unet2D_HPP_

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/unet2d.hpp>

namespace vitis {

namespace ai {

class Unet2DImp
    : public vitis::ai::TConfigurableDpuTask<Unet2D> {
 public:
  Unet2DImp(const std::string &model_name, bool need_preprocess = true);
  virtual ~Unet2DImp();

 private:
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  void preprocess(float* img, int len, int idx);
  virtual Unet2DResult run(float* img, int len) override;
  virtual Unet2DResult run(const std::vector<float>& img) override;
  virtual std::vector<Unet2DResult> run( const std::vector<float*> &img, int len) override;
  virtual std::vector<Unet2DResult> run( const std::vector<std::vector<float>> &input_img) override;

  std::vector<Unet2DResult> unet2d_post_process();
  Unet2DResult unet2d_post_process(int idx);

  float scale_i = 0.0f;
  float scale_o = 0.0f;
  unsigned int batch_size;
  int real_batch_size = 1;
};

}  // namespace ai
}  // namespace vitis

#endif
