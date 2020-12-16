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

#ifndef NMS_H
#define NMS_H
#include <pybind11/pybind11.h>
// must include pybind11/stl.h if using containers in STL in arguments.
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

template <typename DType, int BLOCK_THREADS>
int _nms_gpu(int *keep_out, const DType *boxes_host, int boxes_num,
          int boxes_dim, DType nms_overlap_thresh, int device_id);

constexpr int const threadsPerBlock = sizeof(unsigned long long) * 8;

namespace py = pybind11;
using namespace pybind11::literals;
template <typename DType>
int non_max_suppression(
    py::array_t<DType> boxes,
    py::array_t<int> keep_out,
    DType nms_overlap_thresh,
    int device_id)
{
  py::buffer_info info = boxes.request();
  auto boxes_ptr = static_cast<DType *>(info.ptr);
  py::buffer_info info_k = keep_out.request();
  auto keep_out_ptr = static_cast<int *>(info_k.ptr);
  
  return _nms_gpu<DType, threadsPerBlock>(keep_out_ptr, boxes_ptr, boxes.shape(0), boxes.shape(1), nms_overlap_thresh, device_id);

}
#endif