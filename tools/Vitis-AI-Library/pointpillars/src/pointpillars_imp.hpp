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

#pragma once 

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/pointpillars.hpp>
#include "./postprocess/pointpillars_post.hpp"
#include "./preprocess.hpp"

namespace vitis { namespace ai{ 

template <typename Interface>
class TConfigurableDpuTask2 : public Interface {
 public:
  explicit TConfigurableDpuTask2(const std::string& model_name)
      : configurable_dpu_task_{
            ConfigurableDpuTask::create(model_name, false)} {};
  TConfigurableDpuTask2(const TConfigurableDpuTask2&) = delete;
  virtual ~TConfigurableDpuTask2(){};

  virtual size_t get_input_batch() const override {
    return configurable_dpu_task_->get_input_batch();
  }
  virtual PointPillarsResult run(const V1F&) override { return PointPillarsResult{}; }
  virtual PointPillarsResult run(const float*, int len) override { return PointPillarsResult{}; }
  virtual void do_pointpillar_display(vitis::ai::PointPillarsResult&, int flag, vitis::ai::DISPLAY_PARAM&, 
                  cv::Mat&, cv::Mat&, int, int, ANNORET& annoret) override {}
 public: // change here: from protected to public
  std::unique_ptr<ConfigurableDpuTask> configurable_dpu_task_;
};

class PointPillarsImp:  public PointPillars
{
    vitis::ai::TConfigurableDpuTask2<PointPillars> m0_;
    vitis::ai::TConfigurableDpuTask2<PointPillars> m1_;
public:
  virtual size_t get_input_batch() const override {
    return m0_.configurable_dpu_task_->get_input_batch();
  }
  PointPillarsImp(const std::string &model_name, const std::string &model_name1);
  virtual ~PointPillarsImp();

private:
  virtual PointPillarsResult run( const V1F& v1f) override;
  virtual PointPillarsResult run( const float*, int) override;
  virtual void do_pointpillar_display(PointPillarsResult& res, int flag, DISPLAY_PARAM& g_test,
            cv::Mat& rgb_map, cv::Mat& bev_map, int, int, ANNORET& annoret)  override;

  std::unique_ptr<PointPillarsPost> post_;
  std::unique_ptr<PointPillarsPre> pre_;
};

}}

