#!/usr/bin/env python
# coding=utf-8
# Copyright 2017 challenger.ai
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

# PART OF THIS FILE AT ALL TIMES.

import subprocess as sp
import numpy as np
import json
import argparse
import pprint

def generate_json(gpus,data, caffe, num, weights, model, anno, output_name='eval',input_name='data'):
    children = []
    if gpus is not None and gpus != [] and gpus != '':
        gpus = gpus.split(',')
        cpu_only = False
        num_gpu = len(gpus)
        f = 0
        e = int(num / num_gpu)
        k = e
    else:
        cpu_only = True
    Commands = 'python code/test/generation_results.py --data {} --caffe {} --weights {} --model {} --anno {} --input {} --name {}'.format(data, caffe, weights, model, anno, input_name, output_name)
    
    
    if cpu_only:
        Commands += ' --cpu'
        child = sp.Popen(Commands,shell=True)
        children.append(child)
    else:
        for gpu in gpus:
            Commands += ' --f {} --e {} --gpu {}'.format(f,e,gpu)
            child = sp.Popen(Commands, shell=True)
            children.append(child)
            f += k
            e += k
            if e > num:
                e = num
                
    for child in children:
        child.wait()
        
def read_anno(filename):
    fp = open(filename,'r')
    anno_str = fp.read()
    fp.close()
    anno = json.loads(anno_str)
    return anno

def read_results(gpus,anno_name):
    results = []
    if gpus is not None and gpus != [] and gpus != '':
        gpus = gpus.split(',')
        for gpu in gpus:
            tmp = read_anno(anno_name+'.json.{}'.format(gpu))
            results += tmp
    else:
        results = read_anno(anno_name+'.json')
    return results

def transfromjoint(j):
    jo = np.zeros(14*3)
    a = [5,6,7,2,3,4,11,12,13,8,9,10,0,1];
    for i in range(14):
        jo[i*3] = j[a[i]*3]
        jo[i*3+1] = j[a[i]*3+1]
        jo[i*3+2] = j[a[i]*3+2]
    return np.int32(jo).tolist()


def load_annotations(anno_file, return_dict):
    """Convert annotation JSON file."""

    annotations = dict()
    annotations['image_ids'] = set([])
    annotations['annos'] = dict()
    annotations['delta'] = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                                       0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                                       0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])
    try:
        annos = json.load(open(anno_file, 'r'))
    except Exception:
        return_dict['error'] = 'Annotation file does not exist or is an invalid JSON file.'
        exit(return_dict['error'])

    for anno in annos:
        annotations['image_ids'].add(anno['image_id'])
        annotations['annos'][anno['image_id']] = dict()
        annotations['annos'][anno['image_id']]['human_annos'] = anno['human_annotations']
        annotations['annos'][anno['image_id']]['keypoint_annos'] = anno['keypoint_annotations']

    return annotations


def load_predictions(prediction_file, return_dict):
    """Convert prediction JSON file."""

    predictions = dict()
    predictions['image_ids'] = []
    predictions['annos'] = dict()
    id_set = set([])

    try:
        preds = json.load(open(prediction_file, 'r'))
    except Exception:
        return_dict['error'] = 'Prediction file does not exist or is an invalid JSON file.'
        exit(return_dict['error'])

    for pred in preds:
        if 'image_id' not in pred.keys():
            return_dict['warning'].append('There is an invalid annotation info, \
                likely missing key \'image_id\'.')
            continue
        if 'keypoint_annotations' not in pred.keys():
            return_dict['warning'].append(pred['image_id']+\
                ' does not have key \'keypoint_annotations\'.')
            continue
        image_id = pred['image_id'].split('.')[0]
        if image_id in id_set:
            return_dict['warning'].append(pred['image_id']+\
                ' is duplicated in prediction JSON file.')
        else:
            id_set.add(image_id)
        predictions['image_ids'].append(image_id)
        predictions['annos'][pred['image_id']] = dict()
        predictions['annos'][pred['image_id']]['keypoint_annos'] = pred['keypoint_annotations']

    return predictions


def compute_oks(anno, predict, delta):
    """Compute oks matrix (size gtN*pN)."""

    anno_count = len(list(anno['keypoint_annos'].keys()))
    predict_count = len(list(predict.keys()))
    oks = np.zeros((anno_count, predict_count))
    if predict_count == 0:
        return oks.T

    # for every human keypoint annotation
    for i in range(anno_count):
        anno_key = list(anno['keypoint_annos'].keys())[i]
        anno_keypoints = np.reshape(anno['keypoint_annos'][anno_key], (14, 3))
        visible = anno_keypoints[:, 2] == 1
        bbox = anno['human_annos'][anno_key]
        scale = np.float32((bbox[3]-bbox[1])*(bbox[2]-bbox[0]))
        if np.sum(visible) == 0:
            for j in range(predict_count):
                oks[i, j] = 0
        else:
            # for every predicted human
            for j in range(predict_count):
                predict_key = list(predict.keys())[j]
                predict_keypoints = np.reshape(predict[predict_key], (14, 3))
                dis = np.sum((anno_keypoints[visible, :2] \
                    - predict_keypoints[visible, :2])**2, axis=1)
                oks[i, j] = np.mean(np.exp(-dis/2/delta[visible]**2/(scale+1)))
    return oks


def keypoint_eval(predictions, annotations, return_dict):
    """Evaluate predicted_file and return mAP."""

    oks_all = np.zeros((0))
    oks_num = 0
    oks_res = []
    # Construct set to speed up id searching.
    prediction_id_set = set(predictions['image_ids'])

    # for every annotation in our test/validation set
    for image_id in annotations['image_ids']:
        # if the image in the predictions, then compute oks
        res = {}
        res['image_id'] = image_id
        
        if image_id in prediction_id_set:
            oks = compute_oks(anno=annotations['annos'][image_id], \
                              predict=predictions['annos'][image_id]['keypoint_annos'], \
                              delta=annotations['delta'])
            # view pairs with max OKSs as match ones, add to oks_all
            oks_all = np.concatenate((oks_all, np.max(oks, axis=1)), axis=0)
            # accumulate total num by max(gtN,pN)
            oks_num += np.max(oks.shape)
        else:
            # otherwise report warning
            return_dict['warning'].append(image_id+' is not in the prediction JSON file.')
            # number of humen in ground truth annotations
            gt_n = len(list(annotations['annos'][image_id]['human_annos'].keys()))
            # fill 0 in oks scores
            oks_all = np.concatenate((oks_all, np.zeros((gt_n))), axis=0)
            # accumulate total num by ground truth number
            oks_num += gt_n
            oks = []
        res['oks'] = oks
        oks_res.append(res)
    # compute mAP by APs under different oks thresholds
    average_precision = []
    for threshold in np.linspace(0.5, 0.95, 10):
        average_precision.append(np.sum(oks_all > threshold)/np.float32(oks_num))
    return_dict['score'] = np.mean(average_precision)

    return return_dict,average_precision, oks_res


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default='0', help='gpu id')
parser.add_argument('--data', help='anno image path')
parser.add_argument('--caffe', help='load a model for training or evaluation')
parser.add_argument('--cpu', action='store_true', help='CPU ONLY')
parser.add_argument('--weights', help='weights path')
parser.add_argument('--model', help='model path')
parser.add_argument('--anno', help='anno file path')
parser.add_argument('--name', help='output name, default eval', default='eval')
parser.add_argument('--input', help='input name in the first layer', default='image')
args = parser.parse_args()

gpus = args.gpus
if args.cpu:
    gpus = None
anno = read_anno(args.anno)
num = len(anno)
print('generate json, it will cost some time. probably 2 days?')
generate_json(gpus,args.data,args.caffe,num,args.weights,args.model,args.anno,args.name,args.input)
print('generated!')
result = read_results(gpus,args.name)
for k in range(len(result)):
    for i in range(len(result[k]['keypoint_annotations'])):
        val = result[k]['keypoint_annotations']['human%d'%(i+1)]
        jo = transfromjoint(val)
        result[k]['keypoint_annotations']['human%d'%(i+1)] = jo
    dic_json = json.dumps(result)
save_file = args.name+'.json'
fw = open(save_file, 'w')
fw.write(dic_json)
fw.close()
ref_file = args.anno
submit = save_file
ref = ref_file

# Initialize return_dict
return_dict = dict()
return_dict['error'] = None
return_dict['warning'] = []
return_dict['score'] = None

# Load annotation JSON file
annotations = load_annotations(anno_file=ref,
                               return_dict=return_dict)
print('Complete reading annotation JSON file.')

# Load prediction JSON file
predictions = load_predictions(prediction_file=submit,
                               return_dict=return_dict)
print('Complete reading prediction JSON file.')

# Keypoint evaluation
#start_time = time.time()
return_dict,ave,oks_res = keypoint_eval(predictions=predictions,
                            annotations=annotations,
                            return_dict=return_dict)
print('Complete evaluation.')

# Print return_dict and final score
pprint.pprint(return_dict)
print('Score: ', '%.8f' % return_dict['score'])
