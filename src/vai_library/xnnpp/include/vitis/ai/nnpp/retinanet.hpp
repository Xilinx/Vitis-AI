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
/*
 * Filename: retinanet.hpp
 *
 * Description:
 * This network is used to detecting objects from a input image.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <memory>
#include <vector>
#include <vitis/ai/library/tensor.hpp>
namespace vitis {
namespace ai {
/**
 * @struct RetinaNetResult
 * @brief Struct of the result returned by the RetinaNet neural network.
 */
struct RetinaNetBoundingBox {
    float x;
    float y;
    float w;
    float h;
    float score;
    size_t label;
};

struct RetinaNetLevelInfo {
    size_t level_id;
    size_t anchor_id;
    float score;
    size_t class_id;
};

struct RetinaNetNMSInfo {
    std::vector<float> scores;
    std::vector<std::vector<float>> boxes;
    std::vector<size_t> pos;
};

/**
 * @class RetinaNetPostProcess
 * @brief Class of the RetinaNet post-process. It initializes the parameters once
 * instead of computing them each time the program executes.
 * */
class RetinaNetPostProcess {
public:
  /**
   * @brief Create an RetinaNetPostProcess object.
   * @param input_tensors A vector of all input-tensors in the network.
   * Usage: input_tensors[input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   * Usage: output_tensors[output_index].
   * @param config The DPU model configuration information.
   * @return An unique pointer of RetinaNetPostProcess.
   */
static std::unique_ptr<RetinaNetPostProcess> create(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config);

    /**
    * @brief The batch mode post-processing function of the RetinaNet network.
    * @return The vector of struct of RetinaNetResult.
    */

    virtual std::vector<std::vector<RetinaNetBoundingBox>> retinanet_post_process(size_t batch_size) = 0;
    /**
    * @cond NOCOMMENTS
    */
    virtual ~RetinaNetPostProcess();

protected:
    explicit RetinaNetPostProcess();

    RetinaNetPostProcess(const RetinaNetPostProcess&) = delete;

    RetinaNetPostProcess& operator=(const RetinaNetPostProcess&) = delete;
    /**
    * @endcond
    */
};

}  // namespace ai
}  // namespace vitis
