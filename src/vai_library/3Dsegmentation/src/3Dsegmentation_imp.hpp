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
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/3Dsegmentation.hpp>

namespace vitis {
namespace ai {

class Segmentation3DImp : public vitis::ai::TConfigurableDpuTask<Segmentation3D> {
 public:
  explicit Segmentation3DImp(const std::string &model_name, bool need_preprocess);
  virtual ~Segmentation3DImp();

 private:
  virtual Segmentation3DResult run(std::vector<std::vector<float>> &array) override;
  virtual std::vector<Segmentation3DResult> run(std::vector<std::vector<std::vector<float>>>& arrays) override;
 
  void preprocess(const V2F& array, int idx) ;
  void postprocess( Segmentation3DResult& rs, int idx);

  void post_prec(const std::vector<float>& proj_range, 
                           const std::vector<int>& proj_argmax, 
                           int idx,  V1I& );
  void topk(int idx, float* inv, int k,V1I& out_idx );
  bool enable_knn;
  int in_scale;
  V2F depth;
  V2F py, px;
  int* proj_idx;
  V1I pointsize;
  V2F proj_range;
  V1F proj_unfold;
  V1F proj_unfold2;
  V2I idx_list;  // should be V2I with batch
  std::unique_ptr<float> k2_distances;
  V1F unproj_unfold_1_argmax;
  V1I knn_idx;
  V1F knn_argmax;
  std::unique_ptr<float> knn_argmax_onehot;
  std::vector<int8_t*> input_ptr;
  V1F sensor_std_scale;
  V1F sensor_mean_std_scale;
  std::vector<int8_t*> output_ptr;
  V1I proj_argmax;
  V1I unproj_argmax;
  int size_all;

};
}  // namespace ai
}  // namespace vitis
