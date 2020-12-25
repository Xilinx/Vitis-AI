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


def detect(runner, fpgaBlobs, image):
    fpgaInput = fpgaBlobs[0][0]
    c,h,w = fpgaInput[0].shape
    szs = det_preprocess(image, w, h, fpgaInput[0])
    jid = runner.execute_async(fpgaBlobs[0], fpgaBlobs[1])
    runner.wait(jid)
    rects = det_postprocess(fpgaBlobs[1][1], fpgaBlobs[1][0], [h,w,c], szs)
    return rects

# Main function

def faceDetection(vitis_rundir,outpath, rsz_h, rsz_w, path):
    runner = Runner(vitis_rundir)
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
    
    dirName = outpath
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    
    output_Img_path = dirName
    #os.chdir(path)
    fp = open( output_Img_path + "/output.txt" , 'w')
    for fn in sorted(glob.glob(path+ '/*.jpg'), key=os.path.getsize):
        filename = fn[fn.rfind('/')+1:]
        image_ori = cv2.imread(fn, cv2.IMREAD_COLOR)
        face_rects = detect(runner, fpgaBlobs, image_ori)
        res = []
        if len(face_rects) != 0:
            for face_rect in face_rects:
                res.append("{} {} {} {} {}".format(filename, face_rect[0],face_rect[1],face_rect[2],face_rect[3]))
                print ("{} {} {} {} {}".format(fn, face_rect[0],face_rect[1],face_rect[2],face_rect[3]))
                cv2.rectangle(image_ori,(face_rect[0],face_rect[1]),(face_rect[2],face_rect[3]),(0,255,0),2)
                cv2.imwrite(output_Img_path+filename,image_ori)
        for faces in res:
            fp.write(faces + '\n')
#        else:
            #res.append("{} {} {} {} {}".format(fn, 0,0,0,0))
    fp.close()

# Face Detection 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'analysis densebox model')
    parser.add_argument('--vitisrundir', help = 'path to run directory ', type=str)
    parser.add_argument('--images', help = 'path to image folder',type = str)
    parser.add_argument('--resize_h', help = 'resize height', type = int)
    parser.add_argument('--resize_w', help = 'resize width', type = int)
    parser.add_argument('--output', help = 'output folder', type = str)

    args = parser.parse_args()
    
    work_dir = args.output
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    faceDetection(args.vitisrundir, work_dir, args.resize_h, args.resize_w, args.images) 
