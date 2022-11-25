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
# bash

import numpy as np
import itertools
import math

def center2point(center_x, center_y, width, height):
    return center_x - width / 2., center_y - height / 2., center_x + width / 2., center_y + height / 2.

def point2center(xmin, ymin, xmax, ymax):
    width, height = (xmax - xmin), (ymax - ymin)
    return xmin + width / 2., ymin + height / 2., width, height

def clip(xmin, ymin, xmax, ymax, min_value=0.0, max_value=1.0):
    xmin, ymin = np.clip(xmin, min_value, max_value), np.clip(ymin, min_value, max_value)
    xmax, ymax = np.clip(xmax, min_value, max_value), np.clip(ymax, min_value, max_value)
    return xmin, ymin, xmax, ymax

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

def pboxes_vgg_voc():
    input_shape = [320, 320]
    feature_shapes = [(40, 40), (20, 20), (10, 10), (5, 5)] 
    min_sizes = [(32,), (64,), (128,), (256,)] 
    max_sizes = [(64,), (128,), (256,), (315,)] 
    aspect_ratios = [(2.,), (2.,), (2.,), (2.,)] 
    steps = [(8, 8), (16, 16), (32, 32), (64, 64)]  
    offset=0.5
    pboxes = Priorbox(input_shape, feature_shapes, min_sizes, max_sizes, aspect_ratios, steps)

    return pboxes

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

    def __init__(self, pboxes):
        self.pboxes = pboxes(order="ltrb")
        #self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.pboxes_xywh = pboxes(order="xywh")
        self.nboxes = self.pboxes.shape[0]
        #print("# Bounding boxes: {}".format(self.nboxes))
        self.scale_xy = pboxes.scale_xy
        self.scale_wh = pboxes.scale_wh

    def scale_back_batch(self, arm_scores_in, arm_bboxes_in, odm_scores_in, odm_bboxes_in, device):
        """
            Do scale and transform from xywh to ltrb
            suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        
        arm_bboxes_in[:, :, :2] = self.scale_xy*arm_bboxes_in[:, :, :2]
        arm_bboxes_in[:, :, 2:] = self.scale_wh*arm_bboxes_in[:, :, 2:]
        odm_bboxes_in[:, :, :2] = self.scale_xy*odm_bboxes_in[:, :, :2]
        odm_bboxes_in[:, :, 2:] = self.scale_wh*odm_bboxes_in[:, :, 2:]
        
        for idx in range(arm_bboxes_in.shape[0]):
            arm_bboxes_in[idx, :, :2] = arm_bboxes_in[idx, :, :2]*self.pboxes_xywh[:, 2:] + self.pboxes_xywh[:, :2]
            arm_bboxes_in[idx, :, 2:] = np.exp(arm_bboxes_in[idx, :, 2:])*self.pboxes_xywh[:, 2:]
         
            odm_bboxes_in[idx, :, :2] = odm_bboxes_in[idx, :, :2]*arm_bboxes_in[idx, :, 2:] + arm_bboxes_in[idx, :, :2]
            odm_bboxes_in[idx, :, 2:] = np.exp(odm_bboxes_in[idx, :, 2:])*arm_bboxes_in[idx, :, 2:]

        # Transform format to ltrb 
        l, t, r, b = odm_bboxes_in[:, :, 0] - 0.5*odm_bboxes_in[:, :, 2],\
                     odm_bboxes_in[:, :, 1] - 0.5*odm_bboxes_in[:, :, 3],\
                     odm_bboxes_in[:, :, 0] + 0.5*odm_bboxes_in[:, :, 2],\
                     odm_bboxes_in[:, :, 1] + 0.5*odm_bboxes_in[:, :, 3]
        
        l, t, r, b = clip(l, t, r, b)
        odm_bboxes_in[:, :, 0] = l
        odm_bboxes_in[:, :, 1] = t
        odm_bboxes_in[:, :, 2] = r
        odm_bboxes_in[:, :, 3] = b
        #w = r - l
        #h = b - t
                
        arm_scores_in = softmax_cpu(arm_scores_in, dim=-1)
        odm_scores_in = softmax_cpu(odm_scores_in, dim=-1)
        
        arm_mask = arm_scores_in[:, :, 1] <= 0.01
        arm_mask = np.tile(np.expand_dims(arm_mask, -1), (1, 1, odm_scores_in.shape[-1]))
        #odm_scores_in[w <= 0.003] = 0.0
        #odm_scores_in[h <= 0.003] = 0.0 
        odm_scores_in[arm_mask] = 0.0

        return odm_bboxes_in, odm_scores_in
   
    def decode_batch(self, arm_scores_in, arm_bboxes_in, odm_scores_in, odm_bbxoes_in, criteria = 0.45, max_output=200,device=0):
        bboxes, probs = self.scale_back_batch(arm_scores_in, arm_bboxes_in, odm_scores_in, odm_bbxoes_in, device)
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
            mask = score > 0.005

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

class Priorbox(object):
 
    def __init__(self, input_shape, feature_shapes, min_sizes, max_sizes, aspect_ratios, steps, 
                 offset=0.5, priorbox_order=False, clip=False, scale_xy=0.1, scale_wh=0.2):
        ''' Priorbox class
        Args:
          input_shape: shape of network input.
          feature_shapes: shapes(h, w) list of detection heads feature map.
          min_sizes: list of min sizes.
          max_sizes: list of max sizes.
          aspect_ratios: list of prior apsect ratios per piexl.
          steps: list of steps
          prior_order: num_prior * h * w if false else h * w * num_prior  
          clip: whether prior need clip or not. 
        '''
        super(Priorbox, self).__init__()
       
        self.input_shape = input_shape
        self.feature_shapes = feature_shapes
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.offsets = [offset] * len(steps) 
        self.priorbox_order = priorbox_order
        self.clip = clip

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        prior_boxes = []
           
        for idx, feature_shape in enumerate(self.feature_shapes):
            f_h, f_w = feature_shape
            step = self.steps[idx]
            f_h_s = float(self.input_shape[0]) / step[0]
            f_w_s = float(self.input_shape[1]) / step[1]
            offset = self.offsets[idx]
            min_size = self.min_sizes[idx]
            if self.max_sizes:
                max_size = self.max_sizes[idx]
            
            prior_whs_ratios = []
            for size in min_size:
                p_w = float(size) / self.input_shape[1]
                p_h = float(size) / self.input_shape[0]
                prior_whs_ratios.append((p_w, p_h))
            if self.max_sizes:
                for sizes in zip(min_size, max_size):
                    size = math.sqrt(sizes[0] * sizes[1])   
                    p_w = size / self.input_shape[1]
                    p_h = size / self.input_shape[0]
                    prior_whs_ratios.append((p_w, p_h))

            for size in min_size:
                for alpha in self.aspect_ratios[idx]:
                    s_alpha = math.sqrt(alpha)
                    p_w = float(size) / self.input_shape[1]
                    p_h = float(size) / self.input_shape[0]
                    prior_whs_ratios.append((p_w * s_alpha, p_h / s_alpha))
                    prior_whs_ratios.append((p_w / s_alpha, p_h * s_alpha)) 
        
            if not self.priorbox_order:
                for (h, w) in itertools.product(range(f_h), range(f_w)):
                    cx = (w + offset) / f_w_s
                    cy = (h + offset) / f_h_s
                    for p_wh in prior_whs_ratios:
                        prior_boxes.append([cx, cy, p_wh[0], p_wh[1]])  
            else:
                for p_wh in prior_whs_ratios:
                    for (h, w) in itertools.product(range(f_h), range(f_w)):
                        cx = (w + offset) / f_w_s
                        cy = (h + offset) / f_h_s
                        prior_boxes.append([cx, cy, p_wh[0], p_wh[1]])

        prior_boxes_ltrb = [list(center2point(*prior_box)) for prior_box in prior_boxes]           
        if self.clip:
             prior_boxes_ltrb = [list(clip(*prior_box_ltrb)) for prior_box_ltrb in prior_boxes_ltrb]           
             prior_boxes = [list(point2center(*prior_box_ltrb)) for prior_box_ltrb in prior_boxes_ltrb]

        self.pboxes = np.array(prior_boxes)
        self.pboxes_ltrb = np.array(prior_boxes_ltrb)

    @property
    def scale_xy(self):
        return self.scale_xy_
    
    @property    
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.pboxes_ltrb
        if order == "xywh": return self.pboxes

if __name__ == "__main__":
    pass
