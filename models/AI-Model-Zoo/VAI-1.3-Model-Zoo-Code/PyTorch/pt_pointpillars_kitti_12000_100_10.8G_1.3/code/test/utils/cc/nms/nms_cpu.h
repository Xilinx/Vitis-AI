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

#ifndef NMS_CPU_H
#define NMS_CPU_H
#include <pybind11/pybind11.h>
// must include pybind11/stl.h if using containers in STL in arguments.
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <boost/geometry.hpp>
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
std::vector<int> non_max_suppression_cpu(
    py::array_t<DType> boxes,
    py::array_t<int> order,
    DType thresh,
    DType eps=0)
{
    auto ndets = boxes.shape(0);
    auto boxes_r = boxes.template unchecked<2>();
    auto order_r = order.template unchecked<1>();
    auto suppressed = zeros<int>({ndets});
    auto suppressed_rw = suppressed.template mutable_unchecked<1>();
    auto area = zeros<DType>({ndets});
    auto area_rw = area.template mutable_unchecked<1>();
    // get areas
    for(int i = 0; i < ndets; ++i){
        area_rw(i) = (boxes_r(i, 2) - boxes_r(i, 0) + eps) * (boxes_r(i, 3) - boxes_r(i, 1) + eps);
    }
    std::vector<int> keep;
    int i, j;
    DType xx1, xx2, w, h, inter, ovr;
    for(int _i = 0; _i < ndets; ++_i){
        i = order_r(_i);
        if(suppressed_rw(i) == 1)
            continue;
        keep.push_back(i);
        for(int _j = _i + 1; _j < ndets; ++_j){
            j = order_r(_j);
            if(suppressed_rw(j) == 1)
                continue;
            xx2 = std::min(boxes_r(i, 2), boxes_r(j, 2));
            xx1 = std::max(boxes_r(i, 0), boxes_r(j, 0));
            w = xx2 - xx1 + eps;
            if (w > 0){
                xx2 = std::min(boxes_r(i, 3), boxes_r(j, 3));
                xx1 = std::max(boxes_r(i, 1), boxes_r(j, 1));
                h = xx2 - xx1 + eps;
                if (h > 0){
                    inter = w * h;
                    ovr = inter / (area_rw(i) + area_rw(j) - inter);
                    if(ovr >= thresh)
                        suppressed_rw(j) = 1;
                }
            }
        }
    }
    return keep;
}

template <typename DType>
std::vector<int> rotate_non_max_suppression_cpu(
    py::array_t<DType> box_corners,
    py::array_t<int> order,
    py::array_t<DType> standup_iou,
    DType thresh)
{
    auto ndets = box_corners.shape(0);
    auto box_corners_r = box_corners.template unchecked<3>();
    auto order_r = order.template unchecked<1>();
    auto suppressed = zeros<int>({ndets});
    auto suppressed_rw = suppressed.template mutable_unchecked<1>();
    auto standup_iou_r = standup_iou.template unchecked<2>();
    std::vector<int> keep;
    int i, j;

    namespace bg = boost::geometry;
    typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
    typedef bg::model::polygon<point_t> polygon_t;
    polygon_t poly, qpoly;
    std::vector<polygon_t> poly_inter, poly_union;
    DType inter_area, union_area, overlap;

    for(int _i = 0; _i < ndets; ++_i){
        i = order_r(_i);
        if(suppressed_rw(i) == 1)
            continue;
        keep.push_back(i);
        for(int _j = _i + 1; _j < ndets; ++_j){
            j = order_r(_j);
            if(suppressed_rw(j) == 1)
                continue;
            if (standup_iou_r(i, j) <= 0.0)
                continue;
            // std::cout << "pre_poly" << std::endl;
            try {
                bg::append(poly, point_t(box_corners_r(i, 0, 0), box_corners_r(i, 0, 1)));
                bg::append(poly, point_t(box_corners_r(i, 1, 0), box_corners_r(i, 1, 1)));
                bg::append(poly, point_t(box_corners_r(i, 2, 0), box_corners_r(i, 2, 1)));
                bg::append(poly, point_t(box_corners_r(i, 3, 0), box_corners_r(i, 3, 1)));
                bg::append(poly, point_t(box_corners_r(i, 0, 0), box_corners_r(i, 0, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 0, 0), box_corners_r(j, 0, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 1, 0), box_corners_r(j, 1, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 2, 0), box_corners_r(j, 2, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 3, 0), box_corners_r(j, 3, 1)));
                bg::append(qpoly, point_t(box_corners_r(j, 0, 0), box_corners_r(j, 0, 1)));
                bg::intersection(poly, qpoly, poly_inter);
            } catch (const std::exception& e) {
                std::cout << "box i corners:" << std::endl;
                for(int k = 0; k < 4; ++k){
                    std::cout << box_corners_r(i, k, 0) << " " << box_corners_r(i, k, 1) << std::endl;
                }
                std::cout << "box j corners:" <<  std::endl;
                for(int k = 0; k < 4; ++k){
                    std::cout << box_corners_r(j, k, 0) << " " << box_corners_r(j, k, 1) << std::endl;
                }
                // throw e;
                continue;
            }
            // std::cout << "post_poly" << std::endl;
            // std::cout << "post_intsec" << std::endl;
            if (!poly_inter.empty())
            {
                inter_area = bg::area(poly_inter.front());
                // std::cout << "pre_union" << " " << inter_area << std::endl;
                bg::union_(poly, qpoly, poly_union);
                /*
                if (poly_union.empty()){
                    std::cout << "intsec area:" << " " << inter_area << std::endl;
                    std::cout << "box i corners:" << std::endl;
                    for(int k = 0; k < 4; ++k){
                        std::cout << box_corners_r(i, k, 0) << " " << box_corners_r(i, k, 1) << std::endl;
                    }
                    std::cout << "box j corners:" <<  std::endl;
                    for(int k = 0; k < 4; ++k){
                        std::cout << box_corners_r(j, k, 0) << " " << box_corners_r(j, k, 1) << std::endl;
                    }
                }*/
                // std::cout << "post_union" << poly_union.empty() << std::endl;
                if (!poly_union.empty()){ // ignore invalid box
                    union_area = bg::area(poly_union.front());
                    // std::cout << "post union area" << std::endl;
                    // std::cout << union_area << "debug" << std::endl;
                    overlap = inter_area / union_area;
                    if(overlap >= thresh)
                        suppressed_rw(j) = 1;
                    poly_union.clear();
                }
            }
            poly.clear();
            qpoly.clear();
            poly_inter.clear();

        }
    }
    return keep;
}

#endif
