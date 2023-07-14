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
 * Filename: carplaterecog.hpp
 *
 * Description:
 * This library recog platenumber of single vehicle
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/platedetect.hpp>
#include <vitis/ai/platerecog.hpp>
#include <vitis/ai/ssd.hpp>


namespace xir {
  class Attrs;
};

namespace vitis {
namespace ai {

/**
 * @struct CarPlateRecogResult
 * @brief Struct of the result returned by the Carplaterecog network.
 *
 */
struct CarPlateRecogResult {
  int width;
  int height;
  std::vector<std::pair<SSDResult::BoundingBox,PlateRecogResult>> platerecogs;
};

/*
struct PlateRecogResult {
  /// width of input image.
  int width;
  /// width of input image.
  int height;

  struct BoundingBox {
    /// plate confidence, the value range is 0 to 1.
    float score;
    ///  x-coordinate of the plate relative to the input image.
    int x;
    /// y-coordinate of the plate relative to the input image.
    int y;
    /// plate width
    int width;
    /// plate height
    int height;
  };

  /// the position of plate
  BoundingBox box;
  /// plate number
  std::string plate_number;
  /// plate color
  std::string plate_color;
};
*/

/**
 * @brief Base class for detecting and recognizing plate in a vehicle image
 (cv::Mat).
 *
 * Input is a vehicle image (cv::Mat).
 *
 * Output position, number and color of a plate int the input image.
 *
 * sample code:
 * @code
   cv::Mat image = cv::imread("plate.jpg");
   auto network = xilinx::platerecog::PlateRecog::create(true);
   auto r = network->run(image);
   auto plate_number = r.plate_number.
   auto plate_color = r.plate_colot.
   auto x = r.box.x;
   auto y = r.box.y;
   auto width = r.box.width;
   auto height = r.box.height;
   @endcode
 *
 *
 *
 */
class CarPlateRecog {
 public:
  /**
   * @brief Factory function to get a instance of derived classes of class
   * CarPlateRecog
   *
   * @param need_mean_scale_process normalize with mean/scale or not, true by
   * default
   *
   * @returen A CarPlateRecog class Instance
   */
  static std::unique_ptr<CarPlateRecog> create(
      const std::string &cardetect_model, const std::string &platedetect_model, const std::string &carplaterecog_model,
      bool need_preprocess = true);

  static std::unique_ptr<CarPlateRecog> create(
      const std::string &cardetect_model, const std::string &platedetect_model, const std::string &carplaterecog_model, xir::Attrs *attrs,
      bool need_preprocess = true);

 public:
  explicit CarPlateRecog();
  CarPlateRecog(const CarPlateRecog &) = delete;
  virtual ~CarPlateRecog();

 public:
  /**
   * @brief Function of get running result of CarPlateRecog network
   * Set a Car image and get plate position ,plate number and plate color
   *
   * @param img input Data of input image (cv::Mat) of detected counterpart
   * and resized as inputwidth an outputheight.
   *
   * @return the plate position ,palte number and plate color
   */
  virtual CarPlateRecogResult run(const cv::Mat &image) = 0;
  virtual std::vector<CarPlateRecogResult> run(const std::vector<cv::Mat> &image) = 0;

  /**
   * @brief Function to get InputWidth of the carplaterecog network (input image
   * cols).
   *
   * @return InputWidth of the carplaterecog network.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeigth of the carplaterecog network (input image
   *rows).
   *
   *@return InputHeight of the carplaterecog network.
   */
  virtual int getInputHeight() const = 0;

  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be differnt. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const = 0;
};
}  // namespace ai
}  // namespace vitis
