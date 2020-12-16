
# This code is based on: https://github.com/nutonomy/second.pytorch.git
# 
# MIT License
# Copyright (c) 2018 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import numba


@numba.njit
def is_line_segment_intersection_jit(lines1, lines2):
    """check if line segments1 and line segments2 have cross point
    
    Args:
        lines1 (float, [N, 2, 2]): [description]
        lines2 (float, [M, 2, 2]): [description]
    
    Returns:
        [type]: [description]
    """

    # Return true if line segments AB and CD intersect
    N = lines1.shape[0]
    M = lines2.shape[0]
    ret = np.zeros((N, M), dtype=np.bool_)
    for i in range(N):
        for j in range(M):
            A = lines1[i, 0]
            B = lines1[i, 1]
            C = lines2[j, 0]
            D = lines2[j, 1]
            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (D[0] - A[0])
            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
            if acd != bcd:
                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (D[0] - A[0])
                if abc != abd:
                    ret[i, j] = True
    return ret

@numba.njit
def line_segment_intersection(line1, line2, intersection):
    A = line1[0]
    B = line1[1]
    C = line2[0]
    D = line2[1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0]*B[1] - B[0]*A[1]
            CDDC = C[0]*D[1] - D[0]*C[1]
            DH = BA1 * DC0 - BA0 * DC1
            intersection[0] = (ABBA * DC0 - BA0 * CDDC) / DH
            intersection[1] = (ABBA * DC1 - BA1 * CDDC) / DH
            return True
    return False



def _ccw(A, B, C):
    return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > (
        B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])


def is_line_segment_cross(lines1, lines2):
    # 10x slower than jit version with 1000-1000 random lines input.
    # lines1, [N, 2, 2]
    # lines2, [M, 2, 2]
    A = lines1[:, 0, :][:, np.newaxis, :]
    B = lines1[:, 1, :][:, np.newaxis, :]
    C = lines2[:, 0, :][np.newaxis, :, :]
    D = lines2[:, 1, :][np.newaxis, :, :]
    return np.logical_and(
        _ccw(A, C, D) != _ccw(B, C, D),
        _ccw(A, B, C) != _ccw(A, B, D))


@numba.jit(nopython=False)
def surface_equ_3d_jit(polygon_surfaces):
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


@numba.jit(nopython=False)
def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces, 
            max_num_points_of_surface, 3] 
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces 
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jit(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                     + points[i, 1] * normal_vec[j, k, 1] \
                     + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


@numba.jit
def points_in_convex_polygon_jit(points, polygon, clockwise=True):
    """check points is in 2d convex polygons. True when point in polygon
    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    if clockwise:
        vec1 = polygon - polygon[:, [num_points_of_polygon - 1] +
                                 list(range(num_points_of_polygon - 1)), :]
    else:
        vec1 = polygon[:, [num_points_of_polygon - 1] +
                       list(range(num_points_of_polygon - 1)), :] - polygon
    # vec1: [num_polygon, num_points_of_polygon, 2]
    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)
    success = True
    cross = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            success = True
            for k in range(num_points_of_polygon):
                cross = vec1[j, k, 1] * (polygon[j, k, 0] - points[i, 0])
                cross -= vec1[j, k, 0] * (polygon[j, k, 1] - points[i, 1])
                if cross >= 0:
                    success = False
                    break
            ret[i, j] = success
    return ret

def points_in_convex_polygon(points, polygon, clockwise=True):
    """check points is in convex polygons. may run 2x faster when write in
    cython(don't need to calculate all cross-product between edge and point)
    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_lines = polygon.shape[1]
    polygon_next = polygon[:, [num_lines - 1] + list(range(num_lines - 1)), :]
    if clockwise:
        vec1 = (polygon - polygon_next)[np.newaxis, ...]
    else:
        vec1 = (polygon_next - polygon)[np.newaxis, ...]
    vec2 = polygon[np.newaxis, ...] - points[:, np.newaxis, np.newaxis, :]
    # [num_points, num_polygon, num_points_of_polygon, 2]
    cross = np.cross(vec1, vec2)
    return np.all(cross > 0, axis=2)

