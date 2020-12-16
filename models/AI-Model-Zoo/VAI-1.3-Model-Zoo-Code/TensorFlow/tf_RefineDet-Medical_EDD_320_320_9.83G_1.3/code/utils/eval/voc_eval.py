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


#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from dataset import dataset_common

def parse_rec(image_boxes_info):
    objects = []
    xmin = image_boxes_info[3][0]
    ymin = image_boxes_info[3][1]
    xmax = image_boxes_info[3][2]
    ymax = image_boxes_info[3][3]
    label = image_boxes_info[3][4]
    difficult = image_boxes_info[3][5]
    for i in range(len(xmin)):
        obj_struct = {}
        obj_struct['name'] = label[i]
        obj_struct['difficult'] = difficult[i]
        obj_struct['bbox'] = [int(xmin[i]) - 1,
                              int(ymin[i]) - 1,
                              int(xmax[i]) - 1,
                              int(ymax[i]) - 1]
        objects.append(obj_struct)
    return objects

def do_python_eval(det_bboxes_by_class, gt_bboxes_by_imagename, use_07=True):
    aps = []
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    for cls_name, cls_pair in dataset_common.EDD_LABELS.items():
        if 'none' in cls_name:
            continue
        cls_id = cls_pair[0]
        rec, prec, ap = voc_eval(det_bboxes_by_class, gt_bboxes_by_imagename, cls_id,
                ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls_name, ap))
    print ('MAP = {:.4f}'.format(np.mean(aps)))
    return np.mean(aps)

def voc_ap(rec, prec, use_07_metric=True):
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(det_bboxes_by_class, gt_bboxes_dict_by_imagename, class_id, ovthresh=0.5, use_07_metric=True):
    imagenames = gt_bboxes_dict_by_imagename.keys() 
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(gt_bboxes_dict_by_imagename[imagename])
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == class_id]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    if class_id not in det_bboxes_by_class.keys():
        rec = -1.         
        prec = -1.
        ap = -1.
    else: 
        det_bboxes = det_bboxes_by_class[class_id]
        if any(det_bboxes) == 1:
            image_ids = [x[0] for x in det_bboxes]
            confidence = np.array([float(x[1]) for x in det_bboxes])
            BB = np.array([[float(z) for z in x[2:]] for x in det_bboxes])
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                           (BBGT[:, 2] - BBGT[:, 0]) *
                           (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.
     
    return rec, prec, ap

if __name__ == '__main__':
    do_python_eval()
