/*
-- Copyright 2019 Xilinx Inc.
-- 
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
-- 
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
*/


/*
-- This code is based on: https://github.com/nutonomy/second.pytorch.git
-- 
-- MIT License
-- Copyright (c) 2018 
-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:
-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.
*/

#include "nms.h"
#include "nms_cpu.h"
PYBIND11_MODULE(nms, m)
{
    m.doc() = "non_max_suppression asd";
    m.def("non_max_suppression", &non_max_suppression<double>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "keep_out"_a = 2, "nms_overlap_thresh"_a = 3, "device_id"_a = 4);
    m.def("non_max_suppression", &non_max_suppression<float>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "keep_out"_a = 2, "nms_overlap_thresh"_a = 3, "device_id"_a = 4);
    m.def("non_max_suppression_cpu", &non_max_suppression_cpu<double>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "order"_a = 2, "nms_overlap_thresh"_a = 3, "eps"_a = 4);
    m.def("non_max_suppression_cpu", &non_max_suppression_cpu<float>, py::return_value_policy::reference_internal, "bbox iou", 
          "boxes"_a = 1, "order"_a = 2, "nms_overlap_thresh"_a = 3, "eps"_a = 4);
    m.def("rotate_non_max_suppression_cpu", &rotate_non_max_suppression_cpu<float>, py::return_value_policy::reference_internal, "bbox iou", 
          "box_corners"_a = 1, "order"_a = 2, "standup_iou"_a = 3, "thresh"_a = 4);
    m.def("rotate_non_max_suppression_cpu", &rotate_non_max_suppression_cpu<double>, py::return_value_policy::reference_internal, "bbox iou", 
          "box_corners"_a = 1, "order"_a = 2, "standup_iou"_a = 3, "thresh"_a = 4);
}