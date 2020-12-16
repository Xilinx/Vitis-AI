#coding=utf-8

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
import cv2

def comput_iou(image_height, image_width, gt_landmark, pred_landmark):
    gt_mask = np.zeros((image_height, image_width))
    pred_mask = np.zeros((image_height, image_width))
    set_mask(gt_mask, gt_landmark)
    set_mask(pred_mask, pred_landmark)
    gt_area = np.sum(gt_mask.flatten())
    pred_area = np.sum(pred_mask.flatten())
    iou_area = float(np.sum(np.multiply(gt_mask, pred_mask).flatten()))  
    iou = iou_area / max((gt_area + pred_area - iou_area), 1)
    return iou

def set_mask(image_mask, landmark):
    x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = landmark
    xmin = min(x_1, x_4)
    ymin = min(y_1, y_2)
    xmax = max(x_2, x_3)
    ymax = max(y_3, y_4)
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, image_mask.shape[1] - 1)
    ymax = min(ymax, image_mask.shape[0] - 1)
    res_line_1_3 = 0
    res_line_2 = 0
    res_line_4 = 0
    useable_res = [-1, 1]
    for i in range(ymin, ymax, 1):
        for j in range(xmin, xmax, 1):
            judge_line_1 = int(y_1 + float(y_2 - y_1) / max((x_2 - x_1), 1) * (j - x_1))
            judge_line_3 = int(y_3 + float(y_4 - y_3) / min((x_4 - x_3), -1) * (j - x_3))
            if judge_line_1 <= i and judge_line_3 >= i:
                res_line_1_3 = 1
            else:
                res_line_1_3 = -1
            flag_line_2 = 0
            flag_line_4 = 0
            if x_3 - x_2 > 0:
                flag_line_2 = 1
            else:
                flag_line_2 = -1
            if x_1 - x_4 > 0:
                flag_line_4 = 1
            else:
                flag_line_4 = -1
            if flag_line_2 == 0:
                if j <= x_3:
                    res_line_2 = 1
                else:
                    res_line_2 = -1
            else:
                judge_line_2 = int(y_2 + float(y_3 - y_2) / (flag_line_2 * max(flag_line_2 * (x_3 - x_2), 1)) * (j - x_2))
                if (i - judge_line_2) * flag_line_2 >= 0:
                    res_line_2 = 1
                else:
                    res_line_2 = -1
            if flag_line_4 == 0:
                if j >= x_1:
                    res_line_4 = 1
                else:
                    res_line_4 = -1
            else:
                judge_line_4 = int(y_4 + float(y_1 - y_4) / (flag_line_4 * max(flag_line_4 * (x_1 - x_4), 1)) * (j - x_4))
                if (i - judge_line_4) * flag_line_4 >= 0:
                    res_line_4 = 1
                else:
                    res_line_4 = -1
            assert res_line_1_3 in useable_res
            assert res_line_2 in useable_res
            assert res_line_4 in useable_res              
            if res_line_1_3 == 1 and res_line_2 == 1 and res_line_4 == 1:
                image_mask[i, j] = 1          
                            
def check_set_mask(image_height, image_width, landmark, image_name):
    image_mask = np.zeros((image_height, image_width))
    set_mask(image_mask, landmark)
    image_mask = image_mask * 255
    cv2.imwrite(image_name, image_mask) 

#check_set_mask(320, 320, [67, 169, 124, 182, 119, 208, 63, 195], "23342315_京DH09R8_20171201160736.jpg") 
