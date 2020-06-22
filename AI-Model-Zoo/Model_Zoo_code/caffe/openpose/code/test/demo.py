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

# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.

"""Test a regression network on ai challenger."""

import time
import math

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)


# Add caffe to PYTHONPATH
caffe_path = osp.join('caffe-root/caffe-open', 'python')
add_path(caffe_path)

import caffe
import numpy as np
import cv2

from scipy.ndimage.filters import gaussian_filter


class Config:
    def __init__(self):
        self.use_gpu = 1
        self.gpuID = 1
        self.caffemodel = osp.join(this_dir, '../..', 'float', 'trainval.caffemodel')
        self.deployFile = osp.join(this_dir, '../..', 'float', 'test.prototxt')
        self.description_short = 'openpose'
        self.npoints = 14
        self.mean = [127.5, 127.5, 127.5]
        self.stride = 32
        self.padValue=127.5
        self.test_image = 'demo/demo.jpg'
        self.save_image = 'output/demo_output.png'

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]
    
    pad = 4 * [None]
    pad[0] = 0
    pad[1] = 0
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)
    
    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis = 0)
    pad_left = np.tile(img_padded[:,0:1,:] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis = 1)
    pad_down = np.tile(img_padded[-2:-1,:,:] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis = 0)
    pad_right = np.tile(img_padded[:,-2:-1,:] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis = 1)
    
    return img_padded, pad


def preprocess(img, param):
    img_out = np.float32(img)
    img_out, pad = padRightDownCorner(img_out, param.stride, param.padValue)
    img_out[:, :, 0] = img_out[:, :, 0] - param.mean[0]
    img_out[:, :, 1] = img_out[:, :, 1] - param.mean[1]
    img_out[:, :, 2] = img_out[:, :, 2] - param.mean[2]
    img_out = img_out / 128.0
    
    # change H*W*C -> C*H*W
    return np.transpose(img_out, (2, 0, 1)), pad


def applymodel(net, image, param):
    npoints = param.npoints
    oriImg = image.copy()
    orishape = oriImg.shape
    
    imageToTest_padded, pad = preprocess(oriImg, param)
    sz = imageToTest_padded.shape
    height = sz[1]
    width = sz[2]
    net.blobs['data'].reshape(1,3,height,width)
    net.blobs['data'].data[...] = imageToTest_padded.reshape((1, 3, height, width))
    net.forward()
    heatmap = net.blobs['Mconv7_stage6_L2'].data[0,...]
    heatmap = np.transpose(np.squeeze(heatmap), (1,2,0))
    heatmap = cv2.resize(heatmap, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:height-pad[2], :width-pad[3], :]
    heatmap = cv2.resize(heatmap, (orishape[1], orishape[0]), interpolation=cv2.INTER_CUBIC)

    paf = net.blobs['Mconv7_stage6_L1'].data[0,...]
    paf = np.transpose(np.squeeze(paf), (1,2,0)) # output 0 is PAFs
    paf = cv2.resize(paf, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    paf = paf[:sz[1]-pad[2], :sz[2]-pad[3], :]
    paf = cv2.resize(paf, (orishape[1], orishape[0]), interpolation=cv2.INTER_CUBIC)
    
    return heatmap, paf



def draw_joints(image, heatmap, paf, save_image, param):
    oriImg = image.copy()
    
    
    limbSeq = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], \
           [8,9], [9,10], [1,11], [11,12], [12,13]]
    # the middle joints heatmap correpondence
    mapIdx = [[15,16], [17,18], [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], \
          [31,32], [33,34], [35,36], [37,38], [39,40]]
    
    heatmap_avg = heatmap
    paf_avg = paf

    all_peaks = []
    peak_counter = 0

    for part in range(param.npoints):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > 0.1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = list(range(peak_counter, peak_counter + len(peaks)))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)


    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x-(param.npoints+1) for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]]
        candB = all_peaks[limbSeq[k][1]]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        indexA += 1
        indexB += 1
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    if (vec == [0,0]).all():
                        vec = [1,1]
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    vec = np.divide(vec, norm)

                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))
                    startend = list(startend)
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > 0.05)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0

                    #print(score_with_dist_prior)
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])
            #print(connection_candidate)
            #print(candA,candB)
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    #print(connection_all)
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 16))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    #print(candidate)
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k])
            #print(partAs, partBs,len(connection_all[k]))
            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
                #print(found)
                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    if(subset[j][indexA] != partAs[i]):
                        subset[j][indexA] = partAs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partAs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print("found = 2")
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 14:
                    row = -1 * np.ones(16)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    #print(subset)
    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    canvas = oriImg.copy()

    for i in range(13):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    
    stickwidth = 4

    for i in range(len(limbSeq)):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            
            
    cv2.imwrite(save_image, canvas)


if __name__ == '__main__':
    param = Config()
    if param.use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(param.gpuID)
    net = caffe.Net(param.deployFile, param.caffemodel, caffe.TEST)
    net.name = param.description_short

    
    test_image = param.test_image
    save_image = param.save_image
    image = cv2.imread(test_image)
    imageToTest = cv2.resize(image, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    heatmap, paf = applymodel(net, imageToTest, param)
    
    draw_joints(imageToTest, heatmap, paf, save_image, param)





