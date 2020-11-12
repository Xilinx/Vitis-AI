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

#ifndef BOX_OPS_H
#define BOX_OPS_H

#include <pybind11/pybind11.h>
#include <algorithm>
#include <boost/geometry.hpp>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;

template<typename DType, typename ShapeContainer>
inline py::array_t<DType> constant(ShapeContainer shape, DType value){
    // create ROWMAJOR array.
    py::array_t<DType> array(shape);
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), value);
    return array;
}

template<typename DType>
inline py::array_t<DType> zeros(std::vector<long int> shape){
    return constant<DType, std::vector<long int>>(shape, 0);
}
template <typename DType>
py::array_t<DType>
rbbox_iou(py::array_t<DType> box_corners, py::array_t<DType> qbox_corners,
          py::array_t<DType> standup_iou, DType standup_thresh) {
  namespace bg = boost::geometry;
  typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
  typedef bg::model::polygon<point_t> polygon_t;
  polygon_t poly, qpoly;
  std::vector<polygon_t> poly_inter, poly_union;
  DType inter_area, union_area;
  auto box_corners_r = box_corners.template unchecked<3>();
  auto qbox_corners_r = qbox_corners.template unchecked<3>();
  auto standup_iou_r = standup_iou.template unchecked<2>();
  auto N = box_corners_r.shape(0);
  auto K = qbox_corners_r.shape(0);
  py::array_t<DType> overlaps = zeros<DType>({N, K});
  auto overlaps_rw = overlaps.template mutable_unchecked<2>();
  if (N == 0 || K == 0) {
    return overlaps;
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      if (standup_iou_r(n, k) <= standup_thresh)
        continue;
      bg::append(poly, point_t(box_corners_r(n, 0, 0), box_corners_r(n, 0, 1)));
      bg::append(poly, point_t(box_corners_r(n, 1, 0), box_corners_r(n, 1, 1)));
      bg::append(poly, point_t(box_corners_r(n, 2, 0), box_corners_r(n, 2, 1)));
      bg::append(poly, point_t(box_corners_r(n, 3, 0), box_corners_r(n, 3, 1)));
      bg::append(poly, point_t(box_corners_r(n, 0, 0), box_corners_r(n, 0, 1)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 0, 0), qbox_corners_r(k, 0, 1)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 1, 0), qbox_corners_r(k, 1, 1)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 2, 0), qbox_corners_r(k, 2, 1)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 3, 0), qbox_corners_r(k, 3, 1)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 0, 0), qbox_corners_r(k, 0, 1)));

      bg::intersection(poly, qpoly, poly_inter);

      if (!poly_inter.empty()) {
        inter_area = bg::area(poly_inter.front());
        bg::union_(poly, qpoly, poly_union);
        if (!poly_union.empty()) {
          union_area = bg::area(poly_union.front());
          overlaps_rw(n, k) = inter_area / union_area;
        }
        poly_union.clear();
      }
      poly.clear();
      qpoly.clear();
      poly_inter.clear();
    }
  }
  return overlaps;
}


#endif