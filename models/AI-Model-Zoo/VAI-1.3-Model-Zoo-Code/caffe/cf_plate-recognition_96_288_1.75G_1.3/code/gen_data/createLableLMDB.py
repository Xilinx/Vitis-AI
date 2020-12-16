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


import sys 
import numpy as np
import lmdb

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Densebox evaluation on plate detection dataset.")
    parser.add_argument('--caffe-root', '-c', type=str, 
            default='../../../caffe-xilinx/',
            help='path to caffe root')
    parser.add_argument('--train-gt-file', '-tgt', type=str,
            default='../../data/plate_recognition_val/plate_train.txt',
            help='file records train image annotations.')
    parser.add_argument('--val-gt-file', '-vgt', type=str,
            default='../../data/plate_recognition_val/plate_val.txt',
            help='file records val image annotations.')
    parser.add_argument('--fake-train-gt-file', '-ftgt', type=str,
            default='../../data/plate_recognition_val/plate_train_fake.txt',
            help='file records fake train image annotations.')
    parser.add_argument('--fake-val-gt-file', '-fvgt', type=str,
            default='../../data/plate_recognition_val/plate_val_fake.txt',
            help='file records fake val image annotations.')
    parser.add_argument('--train-label-lmdb', '-tdb', type=str,
            default='../../data/train_plate_recognition_label_lmdb',
            help='train label lmdb name.')
    parser.add_argument('--val-label-lmdb', '-vdb', type=str,
            default='../../data/val_plate_recognition_label_lmdb',
            help='val label lmdb name.')

    return parser.parse_args()

args = parse_args()

sys.path.insert(0, args.caffe_root + 'python')
import caffe

CHARS = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼","川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "O","P", "Q", "R", "S", "T", "U", "V", "W", "X","Y", "Z"];

COLORS = ["blue", "yellow"]

def getCharDict():
    idx = 0
    char2id_dict = {}        
    id2char_dict = {}
    for char in CHARS:
        char2id_dict[char] = idx
        id2char_dict[idx] = char
        idx = idx + 1
    return char2id_dict, id2char_dict

def gen_fake_annos(anno_file, fake_anno_file):
    fa = open(anno_file, 'r')
    anno_lines = fa.readlines()
    fa.close()
    ffa = open(fake_anno_file, 'w')
    for anno_line in anno_lines:
        anno_line = anno_line.split(" ", 1)[0] + ' 0\n'     
        ffa.writelines(anno_line)
    ffa.close()     

def gen_labels(anno_file):  
    all_labels = []
    char2id_dict, id2char_dict = getCharDict()
    fa = open(anno_file, 'r')
    anno_lines = fa.readlines()
    fa.close()
    for line in anno_lines:
        items = line.strip().split(" ")
        platechars = items[1]
        color = items[2]
        labels = np.zeros((8 + 8,), dtype = np.uint8)
        if len(platechars) == 9:    
            labels[0] = char2id_dict[platechars[0:3]] + 1
            labels[9] = 1
            for i in range(6):
                if i == 0:
                    labels[i+1] = char2id_dict[platechars[3+i]] - 41 + 1
                    labels[10+i] = 1
                    continue
                if platechars[3+i] == "*":
                    labels[i+1] = 0
                    labels[10+i] = 0    
                else:
                    labels[i+1] = char2id_dict[platechars[3+i]] - 31 + 1
                    labels[10+i] = 1  
                if labels[i+1] > 24:
                    labels[i+1] = labels[i+1] - 1
        else:
            labels[0] = 0
            labels[9] = 0
            for i in range(6):
                if i == 0:
                    if platechars[1+i] == "*":
                        labels[i+1] = 0
                        labels[10+i] = 0          
                    else:  
                        labels[i+1] = char2id_dict[platechars[1+i]] - 41 + 1
                        labels[10+i] = 1
                    continue
                if platechars[1+i] == "*":
                    labels[i+1] = 0
                    labels[10+i] = 0    
                else:
                    labels[i+1] = char2id_dict[platechars[1+i]] - 31 + 1
                    labels[1-+i] = 1  
                if labels[i+1] > 24:
                    labels[i+1] = labels[i+1] - 1
        labels[8] = sum(labels[9:16])  
        if color == COLORS[0]:
            labels[7] = 0
        elif color == COLORS[1]:
            labels[7] =1
        else:
            print("unsupport color: %s"%color)   
            return
        all_labels.append(labels)

    return all_labels

def createLabelLMDB(anno_file, lmdb_name):
    key = 0
    all_labels = gen_labels(anno_file)
    env = lmdb.open(lmdb_name, map_size=int(1e9))
    with env.begin(write=True) as txn: 
        for labels in all_labels:
            datum = caffe.proto.caffe_pb2.Datum()
            #print(labels, labels.shape)
            datum.channels = labels.shape[0]
            datum.height = 1
            datum.width =  1
            datum.data = labels.tostring() #tobytes()  #if numpy < 1.9 
            datum.label = 0
            key_str = '{:08}'.format(key)
            txn.put(key_str.encode('ascii'), datum.SerializeToString())
            key += 1 

 
if __name__ == "__main__":
    gen_fake_annos(args.train_gt_file, args.fake_train_gt_file) 
    gen_fake_annos(args.val_gt_file, args.fake_val_gt_file) 
    createLabelLMDB(args.train_gt_file, args.train_label_lmdb)
    createLabelLMDB(args.val_gt_file, args.val_label_lmdb)
