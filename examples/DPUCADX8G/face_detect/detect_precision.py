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
import argparse
import datetime
import cv2
import math
import sys
import os
import glob
from vai.dpuv1.rt.vitis.python.dpu.runner import Runner
from detect_ap2 import det_preprocess, det_postprocess
import subprocess
import caffe

def detect(runner, fpgaBlobs, image):
    fpgaInput = fpgaBlobs[0][0]
    c,h,w = fpgaInput[0].shape
    szs = det_preprocess(image, w, h, fpgaInput[0])
    jid = runner.execute_async(fpgaBlobs[0], fpgaBlobs[1])
    runner.wait(jid)
    rects = det_postprocess(fpgaBlobs[1][1], fpgaBlobs[1][0], [h,w,c], szs)
    return rects

# Main function

def faceDetection(args, FDDB_list, FDDB_results_file):
    runner = Runner(args.vitisrundir)
    inTensors = runner.get_input_tensors()
    outTensors = runner.get_output_tensors()
    batch_sz = 1
    fpgaBlobs= []
    for io in [inTensors, outTensors]:
        blobs = []
        for t in io:
            shape = (batch_sz,) + tuple([t.dims[i] for i in range(t.ndims)][1:])
            blobs.append(np.empty((shape), dtype=np.float32, order='C'))
        fpgaBlobs.append(blobs)
    
    for i, line in enumerate(FDDB_list):
        FDDB_results_file.write('%s\n' % line.strip())
        image_name =  args.fddbPath  + line.strip() + '.jpg'
        image_ori = cv2.imread(image_name, cv2.IMREAD_COLOR)
        rects=detect(runner, fpgaBlobs, image_ori)
        FDDB_results_file.write('%d\n' % len(rects))

        for rect in rects:
            FDDB_results_file.write('%d %d %d %d %f\n' % (rect[0], rect[1],
                                                    rect[2] - rect[0], rect[3] - rect[1],
                                                    rect[4]))
    FDDB_results_file.close()


# Face Detection 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'analysis densebox model')
    parser.add_argument('--vitisrundir', help = 'path to run directory ', type=str)
    parser.add_argument('--fddbList', type = str, help = 'FDDB testset list', default='FDDB_list.txt')
    parser.add_argument('--fddbPath', type = str, help = 'FDDB testset path', default='/group/modelzoo/test_dataset/FDDB/FDDB_images/')
    parser.add_argument('--fddbAnno', type = str, help = 'FDDB testset annotations', default='FDDB_annotations.txt')
    parser.add_argument('--resize_h', help = 'resize height', type = int)
    parser.add_argument('--profile', help = 'Provides performance related metrics', type = bool, default=False)
    parser.add_argument('--resize_w', help = 'resize width', type = int)

    args = parser.parse_args()
    FDDB_list_file = open(args.fddbList, 'r')
    FDDB_list = FDDB_list_file.readlines()
    FDDB_list_file.close()
    FDDB_results_file = open('FDDB_results.txt', 'w')
    
    work_dir = os.getcwd() + '/'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    faceDetection(args,FDDB_list, FDDB_results_file) 
    cmd = '%sevaluation/evaluate -a %s -d %sFDDB_results.txt -i %s -l %s -r %s -z .jpg' % (work_dir, 
                                        args.fddbAnno, work_dir, args.fddbPath, args.fddbList, work_dir)
    print (cmd)
    [status, _] = subprocess.getstatusoutput(cmd)

    DiscROC = np.loadtxt('DiscROC.txt')
    for i in range(96, 109):
        index = np.where(DiscROC[:, 1] == i)[0]
        if index :
            break
    print ( "Recall :" + str(np.mean(DiscROC[index], axis = 0)) )

