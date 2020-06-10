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

# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:06:39 2019

@author: shariba
FileName: semanticEval_dice_Jaccard_Overall.py
"""

import numpy as np

def get_args():

    import argparse
    parser = argparse.ArgumentParser(description="For EAD2019 challenge: semantic segmentation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--GT_maskDIR", type=str, default="./data/EDD/labels/", help="ground truth mask image (5 channel tif image only)")
    parser.add_argument("--Eval_maskDIR", type=str, default="./results_visulization/", help="predicted mask image (5 channel tif image only)")
    parser.add_argument("--Img_DIR", type=str, default="./results_visulization/BE/", help="predicted mask image (5 channel tif image only)")
    args = parser.parse_args()
    return args

def file_lines_to_list(path):
  # open txt file lines to a list
  with open(path) as f:
    content = f.readlines()
  # remove whitespace characters like `\n` at the end of each line
  content = [x.strip() for x in content]
  return content

def calculate_confusion_matrix_from_arrays(ground_truth, predictions, nr_labels):
    replace_indices = np.vstack((ground_truth.flatten(),predictions.flatten())).T
    confusion_matrix, _ = np.histogramdd(replace_indices, bins=(nr_labels, nr_labels),range=[(0, nr_labels), (0, nr_labels)])
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix

def calculate_iou(confusion_matrix):
    ious = []
    f2Val=[]
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        
        denom = true_positives + false_positives + false_negatives
        denom_f2 = (5*true_positives + false_positives + 4*false_negatives)
        
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        
        if denom_f2 == 0:
            f2_score = 0
        else:
            f2_score= (5*true_positives) / denom_f2
            
        f2Val.append(f2_score)    
        ious.append(iou)
    return ious, f2Val

def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

if __name__ == '__main__':
    import glob
    import os
    import cv2
    result_dice = []
    result_jaccard = []
    classTypes =  ['BE', 'cancer', 'HGD' , 'polyp', 'suspicious']     
    args=get_args()
    ext=['*.png']
    
    storeDicePerImage = []
    storeJaccardPerImage = []
    storef2ScorePerImage = []
    count = 0    
    for filename in sorted(glob.glob(args.Img_DIR +ext[0])):  
        count += 1
        print(count)
        name = filename.split('/')[-1][:-4]
        dice_val =[]
        jaccard_val=[]
        f2_val = []
        
        for i in range(len(classTypes)):
            gt_file = os.path.join(args.GT_maskDIR, classTypes[i], name+'.png')
            pred_file = os.path.join(args.Eval_maskDIR, classTypes[i],  name+'.png')
            y_true = cv2.imread(gt_file, 0)
            y_pred = cv2.imread(pred_file, 0)
            try:
                y_true = (((y_true[:, :])> 0).astype(np.uint8))
                y_pred = (((y_pred[:, :])> 0).astype(np.uint8))
                result_dice = [dice(y_true.flatten(), y_pred.flatten())]
            except:
                continue
            result_jaccard = [jaccard(y_true.flatten(), y_pred.flatten())]
            confusion_matrix = calculate_confusion_matrix_from_arrays(y_true, y_pred, 2)
            dice_= calculate_dice(confusion_matrix)
            iou, f2_score=calculate_iou(confusion_matrix)   
                
            dice_val.append(result_dice)
            jaccard_val.append(result_jaccard)
                
            if f2_score[0] == 1 and f2_score[1] ==0:
                f2_score[1] = 1
            f2_val.append(f2_score[1])
                
        storeDicePerImage.append(np.mean(dice_val))
        storeJaccardPerImage.append(np.mean(jaccard_val))
        storef2ScorePerImage.append(np.mean(f2_val))
         
    meanDiceVal = np.mean(storeDicePerImage)  
    meanJaccard = np.mean(storeJaccardPerImage)  
    mean_f2_score = np.mean(storef2ScorePerImage) 
    print('mean dice {} and mean jaccard {} and F2-score{}'.format(meanDiceVal, meanJaccard, mean_f2_score))


