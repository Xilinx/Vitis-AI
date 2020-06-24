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


import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
import cv2
import time

class Align(object):
  def __init__(self):
    self.ref_face_size_=(96,112)
    self.ref_face_pts_=[
        [30.29459953,  51.69630051],
        [65.53179932,  51.50139999],
        [48.02519989,  71.73660278],
        [33.54930115,  92.3655014],
        [62.72990036,  92.20410156]
    ]
    
  def tformfwd(self,trans, uv):
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy
  def findNonreflectiveSimilarity(self,uv, xy):
    K = 2
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    # print '--->x, y:\n', x, y
    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
    # We know that X * r = U
    if rank(X) >= 2 * K:
      r, _, _, _ = lstsq(X, U)
      r = np.squeeze(r)
    else:
      raise Exception('cp2tform:twoUniquePointsReq')
    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]
    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])
    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])
    return T, Tinv
  def warp_and_crop_face(self,src_img,
                       facial_pts):
    ref_pts = np.float32(self.ref_face_pts_)
    ref_pts_shp = ref_pts.shape
    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    trans1, trans1_inv = self.findNonreflectiveSimilarity(src_pts, ref_pts)
    xyR = ref_pts
    xyR[:, 0] = -1 * xyR[:, 0]
    trans2r, trans2r_inv = self.findNonreflectiveSimilarity(src_pts, xyR)
    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    trans2 = np.dot(trans2r, TreflectY)
    # Figure out if trans1 or trans2 is better
    xy1 = self.tformfwd(trans1, src_pts)
    norm1 = norm(xy1 - ref_pts)

    xy2 = self.tformfwd(trans2, src_pts)
    norm2 = norm(xy2 - ref_pts)
    if norm1 <= norm2:
        trans = trans1.copy()
        trans_inv = trans1_inv.copy()
    else:
        trans2_inv = inv(trans2)
        trans = trans2.copy()
        trans_inv = trans2_inv.copy()
    tfm = trans[:, 0:2].T

    face_img = cv2.warpAffine(src_img, tfm, (self.ref_face_size_[0], self.ref_face_size_[1]),borderMode=cv2.BORDER_REPLICATE)
    #face_img = cv2.warpAffine(src_img, tfm, (self.ref_face_size_[0], self.ref_face_size_[1]),borderMode=cv2.BORDER_CONSTANT)
    
    return face_img
