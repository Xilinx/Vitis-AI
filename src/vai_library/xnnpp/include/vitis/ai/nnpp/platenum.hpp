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
#include <tuple>
#include <vector>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {
/**
 *@struct PlateNumResult
 *@brief Struct of the result of the platenum network.
 */
struct PlateNumResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// The plate number
  std::string plate_number;
  /// The plate color, Blue / Yellow
  std::string plate_color;
};

/**
 *@brief The post-processing function of the platenum network.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@param det_threshold The results will be filtered by score >= det_threshold.
 *@return the result of the platenum.
 */
std::vector<PlateNumResult> plate_num_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors, const std::vector<int>& sub_x, const std::vector<int>& sub_y);

const std::vector<std::tuple<float, unsigned int, unsigned int>> v_maxparam = {
    std::make_tuple(0.125f, 32, 1), std::make_tuple(0.125f, 26, 1),
    std::make_tuple(0.125f, 35, 1), std::make_tuple(0.125f, 35, 1),
    std::make_tuple(0.125f, 35, 1), std::make_tuple(0.125f, 35, 1),
    std::make_tuple(0.125f, 35, 1), std::make_tuple(0.3125f, 2, 1)};
typedef float (*softmax_output_t)[35];

const std::vector<std::string> charactor_py_ = {
    "jing",  "hu",   "jin",  "yu",    "ji",    "jin", "meng", "liao", "ji",
    "hei",   "su",   "zhe",  "wan",   "min",   "gan", "lu",   "yu",   "e",
    "xiang", "yue",  "gui",  "qiong", "chuan", "gui", "yun",  "zang", "shan",
    "gan",   "qing", "ning", "xin",   "0",     "1",   "2",    "3",    "4",
    "5",     "6",    "7",    "8",     "9",     "A",   "B",    "C",    "D",
    "E",     "F",    "G",    "H",     "J",     "K",   "L",    "M",    "N",
    "O",     "P",    "Q",    "R",     "S",     "T",   "U",    "V",    "W",
    "X",     "Y",    "Z"};
const std::vector<std::string> charactor_ch_ = {
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
    "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
    "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0",  "1",
    "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "A",  "B",  "C",
    "D",  "E",  "F",  "G",  "H",  "J",  "K",  "L",  "M",  "N",  "O",
    "P",  "Q",  "R",  "S",  "T",  "U",  "V",  "W",  "X",  "Y",  "Z"};
const std::vector<std::string> color_ = {"Blue", "Yellow"};
}  // namespace ai
}  // namespace vitis
