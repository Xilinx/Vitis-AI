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
 * Filename: pointpainting.hpp
 *
 * Description:
 * This network is used to detecting objects from a input points data.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */

#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/pointpillars_nuscenes.hpp>

namespace vitis {
namespace ai {

/// pointpainting result  
using PointPaintingResult = PointPillarsNuscenesResult; 

/**
 * @brief Base class for pointpating.
 *
 * Input is points data and related params.
 *
 * Output is a struct of detection results, named PointPaintingResult.
 *
 * Sample code :
   @code
     ...
     std::string anno_file_name = "./sample_pointpainting.info";
     PointsInfo points_info;  
     std::vector<cv::Mat> images;
     read_inno_file_pointpainting(anno_file_name, points_info, 5, points_info.sweep_infos, 16, images); 
     std::string seg_model = "semanticfpn_nuimage_576_320_pt";
     std::string model_0 = "pointpainting_nuscenes_40000_64_0_pt";
     std::string model_1 = "pointpainting_nuscenes_40000_64_1_pt";
     auto pointpainting = vitis::ai::PointPainting::create(
          seg_model, model_0, model_1);
     auto ret = pointpainting->run(images, points_info);
     ...
   please see the test sample for detail.
   @endcode
 */

class PointPainting{
  public:
   /**
    * @brief Factory function to get an instance of derived classes of class PointPainting
    * @param seg_model_name  Segmentation model name
    * @param pp_model_name_0  The first pointpillars nuscenes model name
    * @param pp_model_name_1  The second pointpillars nuscenes model name
    * @return An instance of PointPainting class.
    */
    static std::unique_ptr<PointPainting> create(const std::string &seg_model_name,
                                                const std::string &pp_model_name_0,
                                                const std::string &pp_model_name_1,
                                                bool need_preprocess = true);
   /**
    * @cond NOCOMMENTS
    */
  protected:
    explicit PointPainting();
    PointPainting(const PointPainting &) = delete;
    PointPainting &operator=(const PointPainting &) = delete;

  public:
    virtual ~PointPainting();
    /**
     * @endcond
     */

    /**
     * @brief Function to get input width of the first model of pointpainting (segmentation model).
     *
     * @return Input width of the first model (segmentation model). 
     */
    virtual int getInputWidth() const = 0;

    /**
     *@brief Function to get input height of the first model of pointpainting (segmentation model).
     *
     *@return Input height of the first model (segmentation model). 
     */
    virtual int getInputHeight() const = 0;

    /**
     * @brief Function to get the number of pointpillars inputs processed by the DPU at one time.
     * @note Batch size of different DPU core may be different, it depends on the IP used. For pointpainting class, segmentation model and pointpillars models may be running on different DPU cores.
     *
     * @return Batch size of pointpillars model.
     */
    virtual size_t get_pointpillars_batch() const = 0; 

    /**
     * @brief Function to get the number of segmentation inputs processed by the DPU at one time.
     * @note Batch size of different DPU core may be different, it depends on the IP used. For pointpainting class, segmentation model and pointpillars models may be running on different DPU cores.
     *
     * @return Batch size of segmentation model.
     */
    virtual size_t get_segmentation_batch() const = 0;

    /**
     * @brief Function of get result of the pointpainting full flow.
     *
     * @param input_images Images from different cameras for segmentation . 
     * @param points_info points data and camera related params.
     *
     * @return PointPaintingResult.
     *
     */
    virtual PointPaintingResult run(const std::vector<cv::Mat> &input_images, 
                                    const vitis::ai::pointpillars_nus::PointsInfo &points_info) = 0;

    /**
     * @brief Function of get result of the pointpainting full flow in batch mode.
     *
     * @param batch_input_images Batch input of images from different cameras for segmentation. The size should be equal to the result of get_pointpillars_batch. 
     * @param batch_points_info Batch input of points datas and camera related params.The size should be equal to the result of get_pointpillars_batch.
     *
     * @return A Vector of PointPaintingResult.
     *
     */
    virtual std::vector<PointPaintingResult> run(
            const std::vector<std::vector<cv::Mat>> &batch_input_images,
            const std::vector<vitis::ai::pointpillars_nus::PointsInfo> &batch_points_info) = 0;

    /**
     * @brief Function of get result of the segmentation in batch mode.
     *
     * @param batch_images: Batch input of images from different cameras for segmentation. The size should be equal to the result of get_segmentation_batch.
     *
     * @return A Vector of segmentation result(cv::Mat).
     *
     */
    virtual std::vector<cv::Mat> runSegmentation(std::vector<cv::Mat> batch_images) = 0;

    /**
     * @brief Function of get result points fusion. 
     *
     * @param seg_images Segmentation result images.
     * @param points_info Points data and camera related params. 
     *
     * @return Points data after fusion.
     *
     */
    virtual std::vector<float> fusion(const std::vector<cv::Mat> &seg_images, 
                                    const vitis::ai::pointpillars_nus::PointsInfo &points_info) = 0;

    /**
     * @brief Function of get result of segmentation and points fusion. 
     *
     * @param input_images: Images for segmentation
     * @param points: Points data and camera related params. 
     *
     * @return an instance of PointsInfo with points data result 
     *
     */

    virtual vitis::ai::pointpillars_nus::PointsInfo runSegmentationFusion(
                const std::vector<cv::Mat> &input_images,
                const vitis::ai::pointpillars_nus::PointsInfo &points) = 0;

    /**
     * @brief Function of get result of pointpillars nuscenes neural network.
     *
     * @param points_info Points data and camera related params. 
     *
     * @return PointPaintingResult(same as PointPillarsNuscenesResult).
     *
     */
    virtual PointPaintingResult runPointPillars(const vitis::ai::pointpillars_nus::PointsInfo &points_info) = 0;

    /**
     * @brief Function of get result of pointpillars nuscenes neural network in batch mode.
     *
     * @param batch_points_info A batch of Points data and camera related params.
     *
     * @return A Vector of PointPaintingResult(same as PointPillarsNuscenesResult).
     *
     */
    virtual std::vector<PointPaintingResult> runPointPillars(
            const std::vector<vitis::ai::pointpillars_nus::PointsInfo> &batch_points_info) = 0;
};


}  // namespace ai
}  // namespace vitis
