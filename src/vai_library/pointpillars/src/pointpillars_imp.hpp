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

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/pointpillars.hpp>
#include "./postprocess/pointpillars_post.hpp"
#include "./preprocess.hpp"

namespace vitis { namespace ai{ 

class PointPillarsImp:  public PointPillars
{
public:
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> m0_;
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> m1_;
  virtual size_t get_input_batch() const override {
    return m0_->get_input_batch();
  }
  PointPillarsImp(const std::string &model_name, const std::string &model_name1);
  virtual ~PointPillarsImp();

private:
  virtual PointPillarsResult run( const V1F& v1f) override;
  virtual PointPillarsResult run( const float*, int) override;
  virtual std::vector<PointPillarsResult> run( const V2F& v2f) override;
  virtual std::vector<PointPillarsResult> run( const std::vector<const float*>&,const std::vector<int>&) override;
  virtual void do_pointpillar_display(PointPillarsResult& res, int flag, DISPLAY_PARAM& g_test,
            cv::Mat& rgb_map, cv::Mat& bev_map, int, int, ANNORET& annoret)  override;

  int batchnum = 0;
  int realbatchnum = 0;
  std::unique_ptr<PointPillarsPost> post_;
  std::unique_ptr<PointPillarsPre> pre_;
};

}}

