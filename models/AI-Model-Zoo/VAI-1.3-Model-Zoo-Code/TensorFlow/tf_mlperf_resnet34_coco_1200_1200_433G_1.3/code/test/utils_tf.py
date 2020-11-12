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
import itertools
import json
import time
from math import sqrt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# This function is from https://github.com/kuangliu/pytorch-ssd.
def calc_iou_tensor(box1, box2):
    """ Calculation of IoU based on two boxes tensor,
        Reference to https://github.com/kuangliu/pytorch-ssd
        input:
            box1 (N, 4) 
            box2 (M, 4)
        output:
            IoU (N, M)
    """
    N = box1.shape[0]
    M = box2.shape[0]

    be1 = np.expand_dims(box1, 1).repeat(M, axis=1)
    be2 = np.expand_dims(box2, 0).repeat(N, axis=0)
    lt = np.maximum(be1[:,:,:2], be2[:,:,:2])
    rb = np.minimum(be1[:,:,2:], be2[:,:,2:])

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:,:,0]*delta[:,:,1]

    delta1 = be1[:,:,2:] - be1[:,:,:2]
    area1 = delta1[:,:,0]*delta1[:,:,1]
    delta2 = be2[:,:,2:] - be2[:,:,:2]
    area2 = delta2[:,:,0]*delta2[:,:,1]

    iou = intersect/(area1 + area2 - intersect)
    return iou

def softmax_cpu(x, dim=-1):
    x = np.exp(x)
    s = np.expand_dims(np.sum(x, axis=dim), dim)
    return x/s

def dboxes_R34_coco():
    figsize = [1200, 1200]
    strides = [3,3,2,2,2,2]
    feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]]
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]] 
    aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]] 
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

# This function is from https://github.com/kuangliu/pytorch-ssd.
class Encoder(object):
    """
        Inspired by https://github.com/kuangliu/pytorch-ssd
        Transform between (bboxes, lables) <-> SSD output
        
        dboxes: default boxes in size 8732 x 4, 
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format 
        
        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        #self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.dboxes_xywh = np.expand_dims(dboxes(order="xywh"),0)
        self.nboxes = self.dboxes.shape[0]
        #print("# Bounding boxes: {}".format(self.nboxes))
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def scale_back_batch(self, bboxes_in, scores_in,device):
        """
            Do scale and transform from xywh to ltrb
            suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        
        bboxes_in = bboxes_in.transpose([0,2,1])
        scores_in = scores_in.transpose([0,2,1])

        bboxes_in[:, :, :2] = self.scale_xy*bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh*bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = np.exp(bboxes_in[:, :, 2:])*self.dboxes_xywh[:, :, 2:]

        # Transform format to ltrb 
        l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                     bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        return bboxes_in, softmax_cpu(scores_in, dim=-1)
   
    def decode_batch(self, bboxes_in, scores_in,  criteria = 0.45, max_output=200,device=0):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in,device)
        output = []
        for bbox, prob in zip(bboxes, probs):
            output.append(self.decode_single(bbox, prob, criteria, max_output))
            #print(output[-1])
        return output

    # perform non-maximum suppression
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        # Reference to https://github.com/amdegroot/ssd.pytorch
       
        bboxes_out = []        
        scores_out = []
        labels_out = []

        for i, score in enumerate(np.split(scores_in, scores_in.shape[1], 1)):
            # skip background
            # print(score[score>0.90])
            if i == 0: continue
            score = score.squeeze(1)
            mask = score > 0.05

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.shape[0] == 0: continue

            score_sorted = np.sort(score, axis=0)
            score_idx_sorted = np.argsort(score, axis=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []
        
            while score_idx_sorted.size > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = np.expand_dims(bboxes[idx, :],0)
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < criteria 
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i]*len(candidates))

        bboxes_out = np.concatenate(bboxes_out, axis=0)
        labels_out = np.array(labels_out, dtype=np.long)
        scores_out = np.concatenate(scores_out, axis=0)

        max_ids = np.argsort(scores_out, axis=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, \
                       scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size_w,self.fig_size_h = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh
        
        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps_w = [st[0] for st in steps]
        self.steps_h = [st[1] for st in steps]
        self.scales = scales
        fkw = self.fig_size_w//np.array(self.steps_w)
        fkh = self.fig_size_h//np.array(self.steps_h)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):
            sfeat_w,sfeat_h=sfeat
            sk1 = scales[idx][0]/self.fig_size_w
            sk2 = scales[idx+1][1]/self.fig_size_h
            sk3 = sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]
            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat_w), range(sfeat_h)):
                    cx, cy = (j+0.5)/fkh[idx], (i+0.5)/fkw[idx]
                    self.default_boxes.append((cx, cy, w, h)) 
        self.dboxes = np.array(self.default_boxes)
        self.dboxes.clip(min=0, max=1, out=self.dboxes)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.copy()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5*self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5*self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5*self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5*self.dboxes[:, 3]
    
    @property
    def scale_xy(self):
        return self.scale_xy_
    
    @property    
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes

# This function is from https://github.com/zengarden/light_head_rcnn
def cocoval(detected_json, eval_json):
    eval_gt = COCO(eval_json)

    eval_dt = eval_gt.loadRes(detected_json)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')

    # cocoEval.params.imgIds = eval_gt.getImgIds()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


class MyEncoder(json.JSONEncoder):
   def default(self, obj):
     if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
       np.int16, np.int32, np.int64, np.uint8,
       np.uint16,np.uint32, np.uint64)):
       return int(obj)
     elif isinstance(obj, (np.float_, np.float16, np.float32, 
       np.float64)):
       return float(obj)
     elif isinstance(obj, (np.ndarray,)): # add this line
       return obj.tolist() # add this line
     return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    pass
