"""
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import torch
import os
from losses import DiceCELoss, DiceScore

def reduce_tensor(tensor, num_gpus):                                   
    if num_gpus > 1:                                                   
        rt = tensor.clone()                                            
        dist.all_reduce(rt, op=dist.reduce_op.SUM)                     
        if rt.is_floating_point():                                     
            rt = rt / num_gpus                                         
        else:                                                          
            rt = rt // num_gpus                                        
        return rt                                                      
    return tensor 

loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout="NCDHW",
                     include_background=False)
score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout="NCDHW",
                     include_background=False)  

ms = open('shapes.txt')
fnames = {}
for l in ms.readlines():
  l = l.strip('\n')
  arr = l.split(' ')
  sha = map(int, arr[1:])
  #arr.insert(0, 1)
  fnames[arr[0]] = tuple(sha)

folder1 = './output/'
folder2 = './golden/'

files = os.listdir(folder1)
golden = os.listdir(folder2)

eval_loss = []
scores = []
  
n = 0
for key, value in fnames.items():
  dpu_res = folder1 + key + ".npy"
  if os.path.exists(dpu_res):
    n = n + 1
    output = np.load(dpu_res)
    labels = np.load(folder2 + "case_" + key + '_y.npy')
    
    output = torch.from_numpy(output)
    labels = torch.from_numpy(labels)
    
    print(output.shape)
    print(labels.shape)
    inputs = value
    image_shape = list(inputs[1:])
    print("image shape: ", image_shape)

    dim = len(image_shape)
    roi_shape = [128,128,128]
    overlap=0.5
    padding_val = -2.2
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]
    bounds = [image_shape[i] % strides[i] for i in range(dim)]
    bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]
    
    labels = labels[...,
                 bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
                 bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
                 bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
    
    eval_loss_value = loss_fn(output, labels)
    scores.append(score_fn(output, labels))
    eval_loss.append(eval_loss_value)
  
scores = reduce_tensor(torch.mean(torch.stack(scores, dim=0), dim=0), 1)
eval_loss = reduce_tensor(torch.mean(torch.stack(eval_loss, dim=0), dim=0), 1)

scores, eval_loss = scores.cpu().numpy(), float(eval_loss.cpu().numpy())
print("total images: ", n)
eval_metrics = {"L1 dice": scores[-2],
                "L2 dice": scores[-1],
                "mean_dice": (scores[-1] + scores[-2]) / 2,
                "eval_loss": eval_loss}
print(eval_metrics)
