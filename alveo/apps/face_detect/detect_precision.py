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
import subprocess

from vai.dpuv1.rt.vitis.python.dpu.runner import Runner
from detect_ap2 import det_preprocess, det_postprocess



def detect(runner, fpgaBlobs, image):
    fpgaInput = fpgaBlobs[0][0]
    img = det_preprocess(image, fpgaInput[0])
    c, h, w= fpgaInput[0].shape
    #np.copyto(fpgaInput[0], img)
    jid = runner.execute_async(fpgaBlobs[0], fpgaBlobs[1])
    runner.wait(jid)
    rects = det_postprocess(fpgaBlobs[1][1], fpgaBlobs[1][0], [h, w, c])
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

    rsz_h = args.resize_h
    rsz_w = args.resize_w

    for i, line in enumerate(FDDB_list):
        FDDB_results_file.write('%s\n' % line.strip())
        image_name =  args.fddbPath  + line.strip() + '.jpg'
        src_img=cv2.imread(image_name)
        h, w, ch = src_img.shape
        input_img=cv2.resize(src_img,(rsz_w, rsz_h))
        rects=detect(runner, fpgaBlobs, input_img)
        FDDB_results_file.write('%d\n' % len(rects))
        for rect in rects:
	    #print (rect)
            # scale to actual image size for evaluation purpose
            # topx, topy
            sc_topx = int(rect[0]*w/rsz_w)
            sc_topy = int(rect[1]*h/rsz_h)
            # bottomx, bottomy
            sc_bttmx = int(rect[2]*w/rsz_w)
            sc_bttmy = int(rect[3]*h/rsz_h)

            #print ("{} {} {} {} {}".format(rect[0],rect[1],rect[2],rect[3],rect[4]))
            FDDB_results_file.write('%d %d %d %d %f\n' % (sc_topx, sc_topy, sc_bttmx - sc_topx, sc_bttmy - sc_topy, rect[4]))
    FDDB_results_file.close()
            

    

# Face Detection 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'analysis densebox model')
    parser.add_argument('--vitisrundir', help = 'path to dpuv1 run directory ', type=str)
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
    cmd = '%sevaluation/evaluate -a %s -d %sFDDB_results.txt -i %s -l %s -r %s' % (work_dir, 
                                        args.fddbAnno, work_dir, args.fddbPath, args.fddbList, work_dir)
    print (cmd)
    [status, _] = subprocess.getstatusoutput(cmd)

    DiscROC = np.loadtxt('DiscROC.txt')
    for i in range(96, 109):
        index = np.where(DiscROC[:, 1] == i)[0]
        if index :
            break
    print (np.mean(DiscROC[index], axis = 0)) 

