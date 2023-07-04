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

#include "vitis/ai/nnpp/retinanet.hpp"
#include "./retinanet_detector.hpp"

namespace vitis {
namespace ai {


class RetinaNetPost : public vitis::ai::RetinaNetPostProcess {
public:
    RetinaNetPost(const std::vector<vitis::ai::library::InputTensor>& input_tensors,
        const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
        const vitis::ai::proto::DpuModelParam& config);
    virtual ~RetinaNetPost();

    virtual std::vector<std::vector<RetinaNetBoundingBox>> retinanet_post_process(size_t batch_size) override;
    std::vector<RetinaNetBoundingBox> retinanet_post_process_internal_uniform(unsigned int idx);
    // RetinaNetResult retinanet_post_process_internal();
    // RetinaNetResult retinanet_mlperf_post_process();
    // std::vector<RetinaNetResult> post_processing_arm(const cv::Mat &input_img);

private:
    //std::vector<std::shared_ptr<std::vector<float>>> priors_;
    // DetectorImp
    float conf_threshold_;
    float nms_threshold_;
    size_t top_k_;
    float box_threshold_;
    std::unique_ptr<vitis::ai::RetinaNetDetector> detector_;
    const std::vector<vitis::ai::library::InputTensor> input_tensors_;
    const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
};

}  // namespace ai
}  // namespace vitis
