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

namespace onnx_multitaskv3 {

struct Vehiclev3Result {
  /// number of classes
  /// \li \c  0 : "car"
  /// \li \c 1 : "sign"
  /// \li \c  2 : "person"
  int label;
  /// Confidence of this target.
  float score;
  /// x-coordinate. x is normalized relative to the input image columns.
  /// Range from 0 to 1.
  float x;
  /// y-coordinate. y is normalized relative to the input image rows.
  /// Range from 0 to 1.
  float y;
  /// Width. Width is normalized relative to the input image columns,
  /// Range from 0 to 1.
  float width;
  /// Height. Heigth is normalized relative to the input image rows,
  /// Range from 0 to 1.
  float height;
  /// The angle between the target vehicle and ourself.
  float angle;
};

/**
 *@struct MultiTaskv3Result
 *@brief  Struct of the result returned by the MultiTaskv3 network.
 */
struct MultiTaskv3Result {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// Detection result of SSD task.
  std::vector<Vehiclev3Result> vehicle;
  /// Segmentation result to visualize, cv::Mat type is CV_8UC1 or CV_8UC3.
  cv::Mat segmentation;
  /// Lane segmentation.
  cv::Mat lane;
  /// Drivable area.
  cv::Mat drivable;
  /// Depth estimation.
  cv::Mat depth;
};
}