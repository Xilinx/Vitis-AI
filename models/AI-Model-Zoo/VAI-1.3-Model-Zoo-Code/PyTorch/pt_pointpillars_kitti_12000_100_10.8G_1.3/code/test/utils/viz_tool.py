# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
import os
import cv2
from tqdm import tqdm
from easydict import EasyDict


def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX'] ; maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY'] ; maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ'] ; maxZ = BoundaryCond['maxZ']
    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0]<=maxX) & (PointCloud[:, 1] >= minY) & (PointCloud[:, 1]<=maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2]<=maxZ))
    PointCloud = PointCloud[mask]
    return PointCloud


def makeBVFeature(PointCloud_, BoundaryCond, Discretization, norm_H = False):
    '''
    Note: the velodyne coordinate definition relative to ego
            x -> forward
            y -> left
            z -> up
    '''
    Height = int( (BoundaryCond['maxX'] - BoundaryCond['minX']) / Discretization + 1)
    Width  = int( (BoundaryCond['maxY'] - BoundaryCond['minY']) / Discretization + 1)

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:,0] = np.int_(np.floor(PointCloud[:,0] / Discretization))
    PointCloud[:,1] = np.int_(np.floor(PointCloud[:,1] / Discretization) + Width/2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:,2],PointCloud[:,1],PointCloud[:,0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height,Width))

    _, indices = np.unique(PointCloud[:,0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]
    #some important problem is image coordinate is (y,x), not (x,y)
    heightMap[np.int_(PointCloud_frac[:,0]), np.int_(PointCloud_frac[:,1])] = PointCloud_frac[:,2]

    if norm_H:
        mask = heightMap != 0
        heightMap[mask] = (heightMap[mask] - BoundaryCond['minZ']) / (BoundaryCond['maxZ'] - BoundaryCond['minZ'])

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height,Width))
    densityMap = np.zeros((Height,Width))

    _, indices, counts = np.unique(PointCloud[:,0:2], axis = 0, return_index=True,return_counts = True)
    PointCloud_top = PointCloud[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1)/np.log(64))

    intensityMap[np.int_(PointCloud_top[:,0]), np.int_(PointCloud_top[:,1])] = PointCloud_top[:,3]
    densityMap[np.int_(PointCloud_top[:,0]), np.int_(PointCloud_top[:,1])] = normalizedCounts

    RGB_Map = np.zeros((Height,Width,3))
    RGB_Map[:,:,0] = intensityMap      # b_map
    RGB_Map[:,:,1] = heightMap         # g_map
    RGB_Map[:,:,2] = densityMap        # r_map

    save = np.zeros((512,1024,3))
    save = RGB_Map[0:512,0:1024,:] * 255
    save = save.astype(np.uint8)

    return save


def preprocess(PointCloud, BoundaryCond, Resolution):
    PointCloudInRange = removePoints(PointCloud, BoundaryCond)
    BEVMap = makeBVFeature(PointCloudInRange, BoundaryCond , Resolution, norm_H = True)
    return BEVMap


def isBack(pt3d, forward_axis):
    return pt3d[forward_axis] < 0


def isAnyBack(corners, forward_axis):
    return any(corners[:,forward_axis] < 0)


def get2DBboxes(objs):
    bboxes = []
    for obj in objs:
        if isBack([obj.x, obj.y, obj.z], 2):
            continue
        bboxes.append([obj.x1, obj.y1, obj.x2, obj.y2])
    return bboxes


def getCam2velo(calib_file):
    with open(calib_file, 'r') as fr:
        lines = fr.readlines()
    cam2velo = None
    for line in lines:
        key, val_str = line.strip().split(": ")
        if key == "Tr_velo_to_cam":
            velo2cam = np.array([float(v) for v in val_str.split(" ")]).reshape((3,4))
            velo2cam = np.vstack([velo2cam, [0,0,0,1]])
            cam2velo = np.linalg.inv(velo2cam)
            break
    return cam2velo


def getP(calib_file):
    with open(calib_file, 'r') as fr:
        lines = fr.readlines()
    P0 = None
    for line in lines:
        key, val_str = line.strip().split(": ")
        if key == "P0":
            P0 = np.array([float(v) for v in val_str.split(" ")]).reshape((3,4))
            break
    return P0


def rectifyCoord(obj):
    ''' Convert from Coord A to Coord B
        Coord A:
            x -> right
            y -> forward
            z -> up
        Coord B:
            x -> forward
            y -> left
            z -> up
    '''
    rect_obj = EasyDict(obj)
    rect_obj.a = obj.a - np.pi/2
    trans_mat = np.array([[ 0, 1, 0],
                          [-1, 0, 0],
                          [ 0, 0, 1]])
    [rect_obj.x, rect_obj.y, rect_obj.z] = np.dot(trans_mat, np.array([obj.x, obj.y, obj.z]).T)
    return rect_obj


def rotZ_3d(t):
    s = np.sin(t)
    c = np.cos(t)
    return np.array([[c, -s, 0], 
                     [s,  c, 0], 
                     [0,  0, 1]])


def rotZ_2d(t):
    s = np.sin(t)
    c = np.cos(t)
    return np.array([[c, -s], 
                     [s,  c]])


def get_2DBboxes_from_anno(dt_anno, score_thresh):
    bboxes = []
    # for score, box_2d, box3d_lidar in zip(dt_anno['score'], dt_anno['bbox'], dt_anno['box3d_lidar']):
    #     corners = generate8CornersKitti(obj.x, obj.y, obj.z, obj.h, obj.w, obj.l, obj.a)
    for score, box_2d, box3d_cam in zip(dt_anno['score'], dt_anno['bbox'], dt_anno['box3d_camera']):
        x, y, z, l, h, w, a = box3d_cam
        corners = generate8CornersKitti(x, y, z, h, w, l, a)
        if score < score_thresh or isAnyBack(corners, forward_axis=2):
            continue
        # if score < score_thresh or isBack(box3d_lidar[:3], 0):
        #     continue
        bboxes.append(box_2d)
    return bboxes


def anno2objs(anno, score_thresh):
    objs = []
    # for score, box3d_cam, box3d_lidar in zip(anno['score'], anno['box3d_camera'], anno['box3d_lidar']):
    #     if score < score_thresh or isBack(box3d_lidar[:3], 0):
    #         continue
    for score, box3d_cam in zip(anno['score'], anno['box3d_camera']):
        x, y, z, l, h, w, a = box3d_cam
        corners = generate8CornersKitti(x, y, z, h, w, l, a)
        if score < score_thresh or isAnyBack(corners, forward_axis=2):
            continue
        obj = EasyDict()
        obj.x, obj.y, obj.z = box3d_cam[:3]
        obj.l, obj.h, obj.w = box3d_cam[3:6]
        obj.a = box3d_cam[6]
        objs.append(obj)
    return objs


def label2BevImage_v4(dt_anno, BoundaryCond, Discretization, score_thresh=0.4):

    Height = int( (BoundaryCond['maxX'] - BoundaryCond['minX']) / Discretization + 1)
    Width  = int( (BoundaryCond['maxY'] - BoundaryCond['minY']) / Discretization + 1)
    bboxes = []
    for score, box3d_lidar in zip(dt_anno['score'], dt_anno['box3d_lidar']):
        if score < score_thresh:
            continue
        # img_x/img_w: vertical axis
        # img_y/img_h: horizontal axis
        box_x, box_y = box3d_lidar[:2]
        box_h, box_w = box3d_lidar[3:5]
        rot_angle = box3d_lidar[6]
        img_x = np.int_(np.floor(box_x / Discretization))
        img_y = np.int_(np.floor(box_y / Discretization) + Width/2)
        img_w = np.int_(np.floor(box_w / Discretization))
        img_h = np.int_(np.floor(box_h / Discretization))
        angle = -rot_angle - np.pi/2
        box = [img_x, img_y, img_w, img_h, angle]
        corners = center2corners(box)
        bboxes.append(corners)
    return bboxes


def center2corners(box):
    img_x, img_y, img_w, img_h, angle = box
    corner = []
    offset = 1/2. * np.array([[ 1, -1, -1,  1],
                              [ 1,  1, -1, -1]])
    offset *= np.tile(np.array([[img_w],[img_h]]), (1,4))
    offset = np.dot(rotZ_2d(angle), offset)
    corners = offset + np.tile(np.array([[img_x],[img_y]]), (1,4)) 
    return corners.T


def test_center2corners():
    box = [0,0,2,4,np.pi/2]
    corners = center2corners(box)
    print("box: ", box)
    print("corners: ", corners)


def drawBboxes(img, bboxes):
    for box in bboxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0))
    return img


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def generate8Corners(x, y, z, h, w, l, ry):
    size = np.array([l, h, w])
    center = np.array([x, y, z])
    offset = np.array([
                       [ 1/2,  1/2,  1/2], 
                       [-1/2,  1/2,  1/2], 
                       [-1/2, -1/2,  1/2], 
                       [ 1/2, -1/2,  1/2],
                       [ 1/2,  1/2, -1/2],
                       [-1/2,  1/2, -1/2],  
                       [-1/2, -1/2, -1/2], 
                       [ 1/2, -1/2, -1/2] 
                      ])
    offset *= np.tile(size, (8,1))
    offset = np.dot(roty(ry), offset.T).T
    corners = offset + np.tile(center, (8,1))
    return corners


def generate8CornersKitti(x, y, z, h, w, l, ry):
    size = np.array([l, h, w])
    center = np.array([x, y, z])
    offset = np.array([
                       [ 1/2,    0,  1/2], 
                       [-1/2,    0,  1/2], 
                       [-1/2,   -1,  1/2], 
                       [ 1/2,   -1,  1/2],
                       [ 1/2,    0, -1/2],
                       [-1/2,    0, -1/2],  
                       [-1/2,   -1, -1/2], 
                       [ 1/2,   -1, -1/2] 
                      ])
    offset *= np.tile(size, (8,1))
    offset = np.dot(roty(ry), offset.T).T
    corners = offset + np.tile(center, (8,1))
    return corners


def project_3d_pts(corners, P):
    homo_pts = np.hstack((corners, np.ones((len(corners), 1))))
    proj_pts = np.dot(P, homo_pts.T).T
    pc_img = np.zeros((8,3))
    pc_img[:,0] = proj_pts[:,0] / proj_pts[:,2]
    pc_img[:,1] = proj_pts[:,1] / proj_pts[:,2]
    pc_img[:,2] = proj_pts[:,2]
    return pc_img


def draw_projected_box3d(image, qs, color=(255,255,255), thickness=1):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):

       i,j=k,(k+1)%4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
    return image


def get_dir_points(obj, length):
    ''' get two points for the heading angle
    '''
    offset = np.array([[0,0,0], [length,0,0]])
    offset = np.dot(roty(obj.a), offset.T).T
    center = np.array([obj.x, obj.y, obj.z])
    dir_pts = offset + np.tile(center, (2,1))
    return dir_pts


def draw_proj_box(img, obj, P):
    ''' draw projected 3d bounding boxe on the rgb image of camera
    img: rgb image
    obj: 3d bounding box in the Camera Coordinate for KITTI
    P  : matrix of transformation from the Camera Coordinate to the image plane (P2 * rect)
    '''
    corners = generate8CornersKitti(obj.x, obj.y, obj.z, obj.h, obj.w, obj.l, obj.a)
    if isAnyBack(corners, forward_axis=2):
        return img
    proj_pts = project_3d_pts(corners, P)
    img = draw_projected_box3d(img, proj_pts, color=(0,255,255), thickness=1)

    # draw a line for the heading angle
    direct_pts = np.zeros((8,3))
    dir_pts = get_dir_points(obj, length=2)
    direct_pts[0:2,:] = dir_pts
    pts = project_3d_pts(direct_pts, P)
    pt1, pt2 = (int(pts[0,0]), int(pts[0,1])), (int(pts[1,0]), int(pts[1,1]))
    img = cv2.line(img, pt1, pt2, color=(0,255,0), thickness=1)
    img = cv2.circle(img, pt1, 2, color=(0,0,255), thickness=1)
    return img


def drawRotBboxes(img, bboxes):
    ''' Input: 
            bboxes: list of corners whose shape is 4x2, each row [x, y]
            [x, y]: x -> vertical, y-> horizontal
    '''
    for corners in bboxes:
        num = len(corners)
        for i in range(num):
            pt1 = tuple(corners[i % num][::-1].astype(int))
            pt2 = tuple(corners[(i+1) % num][::-1].astype(int))
            cv2.line(img, pt1, pt2, (0,255,0))
        ori_start = tuple(((corners[0] + corners[2])/2.)[::-1].astype(int))
        ori_end = tuple(((corners[0] + corners[3])/2.)[::-1].astype(int))
        cv2.line(img, ori_start, ori_end, (0,255,0))
    return img


def draw2DBboxes(img, bboxes):
    ''' Input:
            bboxes: list of points [x1, y1, x2, y2]
            x: horizontal, y: vertical
    '''
    for box in bboxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0))
    return img

