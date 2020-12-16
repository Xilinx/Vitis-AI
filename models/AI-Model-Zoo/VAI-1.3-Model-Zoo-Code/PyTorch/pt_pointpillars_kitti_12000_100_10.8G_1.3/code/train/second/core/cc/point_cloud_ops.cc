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


#include "point_cloud_ops.h"

PYBIND11_MODULE(point_cloud_ops_cc, m) {
  m.doc() = "pybind11 example plugin";
  m.def("points_to_voxel", &points_to_voxel<float, 3>, "matrix tensor_square",
        "points"_a = 1, "voxels"_a = 2, "coors"_a = 3,
        "num_points_per_voxel"_a = 4, "voxel_size"_a = 5, "coors_range"_a = 6,
        "max_points"_a = 7, "max_voxels"_a = 8);
  m.def("points_to_voxel", &points_to_voxel<double, 3>, "matrix tensor_square",
        "points"_a = 1, "voxels"_a = 2, "coors"_a = 3,
        "num_points_per_voxel"_a = 4, "voxel_size"_a = 5, "coors_range"_a = 6,
        "max_points"_a = 7, "max_voxels"_a = 8);
}