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
# Multiprocessing video with queue
   
#%% import packages
import multiprocessing as mp
import argparse
import os
import cv2    
import time
import ctypes
import threading

from detect_ap2 import det_preprocess, det_postprocess

from vai.dpuv1.rt import xdnn
import numpy as np
from vai.dpuv1.rt.vitis.python.dpu.runner import Runner



#%% define classes and functions
##################################################
# CONSTANTS
##################################################
num_shared_slots = 200
    

# Current version does copies...
# Assumes all types are np.float32/ctypes.c_float
class SharedMemoryQueue:
    def __init__(self, name, length, buf_shapes_list):

        print("Creating SharedMemoryQueue",name)
        self._name = name
        self._len = length

        # Hard coded for floats...
        self._mem_type = ctypes.c_float
        self._np_type = np.float32
        
        # put() function copies into the free list
        self._freeList = mp.Queue(length)

        # get() function gets id of open slot. consumer needs to confirm when data is read
        self._readList = mp.Queue(length)

        self._buf_shapes_list = buf_shapes_list
        self._buf_sizes_list = list(map(lambda x: np.prod(x), buf_shapes_list))

        print("Creating Shared Memory with buf_shape_list=",self._buf_shapes_list)
#        import pdb;pdb.set_trace()    
        self._shared_memory_arrs = list()
        for i in range(length):
            buf_list = list()
            for buf_size in list(self._buf_sizes_list):
                buf_list.append(mp.Array(self._mem_type, int(buf_size)))
#                buf_list.append(mp.Array(self._mem_type, buf_size))
            self._shared_memory_arrs.append(buf_list)
            self._freeList.put(i)

        
    def close(self):
        self._readList.put(None)


    def accessBuffer(self, slot_id):
        return self._shared_memory_arrs[slot_id]


    def accessNumpyBuffer(self, slot_id):
        buf_list = list()
        for i in range(len(self._buf_shapes_list)):
            np_arr = np.frombuffer(self._shared_memory_arrs[slot_id][i].get_obj(), dtype = self._np_type)
            np_arr = np.reshape(np_arr, self._buf_shapes_list[i], order = 'C')
            buf_list.append(np_arr)
        
        return buf_list

    
    def openWriteId(self):
        id = self._freeList.get()
        return id


    def closeWriteId(self, id):
        # finished writing slot id 
        self._readList.put(id)


    def openReadId(self):
        id = self._readList.get()
        return id


    def closeReadId(self, id):
        # finished reading slot id
        self._freeList.put(id)


    def dump(self):
        for i in range(self._len):
          buf_list = self.accessNumpyBUffer(i)
          for np_arr in buf_list:
#              print("Slot=",i,"Array=",j,"Val=",np_arr)
              print("Slot=",i,"Val=",np_arr)
        



import inspect
def funcname():
    return inspect.stack()[1][0].f_code.co_name

def cam_loop(shared_frame_arrs, ready_fpga):
    cap = cv2.VideoCapture('Pedestrians.mp4')
    # First read frames into a list
    frames = []
    while cap.isOpened():

        s, frame = cap.read()
        if s:
            frame = cv2.resize(frame, (320, 320), interpolation = cv2.INTER_LINEAR)
            frames.append(frame)
        else:
            break
    
    ready_fpga.get()

    # Then send frames to shared memory as fast as possible for measuring pipeline performance
    print(funcname(),"Putting Frames...")
    start_time = time.time()
    frame_cnt = 0
    for f in frames:
        #print(frame_cnt,flush=True)
        write_slot = shared_frame_arrs.openWriteId()
        np_arr = shared_frame_arrs.accessNumpyBuffer(write_slot)
        np_arr[0][:] = f
        shared_frame_arrs.closeWriteId(write_slot)
        frame_cnt += 1
    end_time = time.time()
    shared_frame_arrs.close()

    
    print(funcname(),"Video Ending! Frame Count = ",frame_cnt)
    print('{0} cam loading time: {1} seconds'.format(funcname(),end_time - start_time))


def detect_pre(shared_frame_arrs, shared_trans_arrs):

    start_time = None
    frame_id = 0
    while True:
        read_slot = shared_frame_arrs.openReadId()

        if start_time is None:
            start_time = time.time()
            
        if read_slot is None:
            break
        read_arrs = shared_frame_arrs.accessNumpyBuffer(read_slot)
            
        frame_id += 1

        write_slot = shared_trans_arrs.openWriteId()
        write_arrs = shared_trans_arrs.accessNumpyBuffer(write_slot)


        det_preprocess(read_arrs[0], write_arrs[1])

        shared_frame_arrs.closeReadId(read_slot)
        shared_trans_arrs.closeWriteId(write_slot)
    end_time = time.time()
    shared_trans_arrs.close()
    
    print("Ran",frame_id,"frames")
    print('detect_preprocessing time: {0} seconds'.format(end_time - start_time))






def fpga_wait( runner, q, shared_output_arrs, shared_trans_arrs):
  numProcessed = 0
  while True:
    write_slot, read_slot, jid = q.get()

    if write_slot is None:
      break

    runner.wait(jid)

    #qFpga.put(img_num)
    shared_trans_arrs.closeReadId(read_slot)
    shared_output_arrs.closeWriteId(write_slot)

  shared_output_arrs.close()

def fpga_process(vitisrundir,shared_trans_arrs,shared_output_arrs ,ready_fpga ):
    runner = Runner(vitisrundir)
    qWait = mp.Queue(maxsize=100)
    ready_fpga.put(1)
    t = threading.Thread(target=fpga_wait, args=(runner, qWait, shared_output_arrs, shared_trans_arrs))
    t.start()
    numProcessed = 0
    startTime = time.time()
    while True:
        # Get the buffer for fpga output
        write_slot = shared_output_arrs.openWriteId()
        write_slot_arrs = shared_output_arrs.accessNumpyBuffer(write_slot)

        # Get the input buffer for fpga exec
        read_slot = shared_trans_arrs.openReadId()
        
        if read_slot is None: break
        read_slot_arrs = shared_trans_arrs.accessNumpyBuffer(read_slot)


       # Start execution
        jid = runner.execute_async([read_slot_arrs[1]], write_slot_arrs)
        # runner.wait(jid)
        qWait.put((write_slot, read_slot, jid))
        #shared_trans_arrs.closeReadId(read_slot)

        numProcessed += 1


    qWait.put((None, None, None))
    t.join()
    elapsedTime = ( time.time() - startTime )
    print( "FPGA_process: ", float(numProcessed)/elapsedTime, "img/s")



def detect_post(shared_output_arrs, face_q):
    start_time = None
    frame_cnt = 0
    while True:
        read_slot = shared_output_arrs.openReadId()

        if start_time is None:
            start_time = time.time() 
        
        if read_slot is None:
            break

        read_slot_arrs = shared_output_arrs.accessNumpyBuffer(read_slot)

        face_rects = det_postprocess(read_slot_arrs[1], read_slot_arrs[0], [320,320,3])

        shared_output_arrs.closeReadId(read_slot)
        frame_cnt += 1
        face_q.put(face_rects)
        
    end_time = time.time()
    face_q.put(None)
    
    print('detect post processing time: {0} seconds'.format(end_time - start_time))

    total_time = end_time-start_time
    print('Total run: {0} frames in {1} seconds ({2} fps)'.format(frame_cnt, total_time, frame_cnt/total_time))

            
def show_loop(face_q):

    #cv2.namedWindow('face_detection')
    work_dir = os.getcwd() + '/output/'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    frame_id = 0

    cap = cv2.VideoCapture('Pedestrians.mp4')

    start_time = None
    while cap.isOpened():

        s, frame = cap.read()
        if s:
            frame = cv2.resize(frame, (320, 320), interpolation = cv2.INTER_LINEAR)
        else:
            break

        face_rects = face_q.get()
        if face_rects is None:
            break

        if start_time is None:
            start_time = time.time()

        # Show every modulo frame for performance reasons...
        if frame_id % 1 == 0:
            for face_rect in face_rects:
                cv2.rectangle(frame,(face_rect[0],face_rect[1]),(face_rect[2],face_rect[3]),(0,255,0),2)

            #cv2.imshow('face_detection', frame)
            cv2.imwrite(work_dir + str(frame_id) + '.png', frame)
            #cv2.waitKey(40)

        frame_id += 1
    end_time = time.time()
    print('drawing boxes time: {0} seconds'.format(end_time - start_time))
    
    
    
#%%  main       
 

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(description = 'FaceDetection Demo')
    parser.add_argument('--vitisrundir', help = 'path to dpuv1 run directory ', type=str)
    args = parser.parse_args()

    frame_q = mp.Queue()
    resize_q = mp.Queue()
    trans_q = mp.Queue()
    output_q = mp.Queue()
    face_q = mp.Queue()    
    ready_fpga = mp.Queue()
    
    sharedInputArrs = []

    compilerJSONObj = xdnn.CompilerJsonParser(args.vitisrundir + '/compiler.json')

    input_shapes = [v for k,v in compilerJSONObj.getInputs().items()]
    output_shapes = [v for k,v in compilerJSONObj.getOutputs().items()]

    input_sizes = map(lambda x: np.prod(x), input_shapes)
    output_sizes = map(lambda x: np.prod(x), output_shapes)
    
    # shared memory from video capture to preprocessing
    shared_frame_arrs = SharedMemoryQueue("frame",num_shared_slots, [(320,320,3)])
    # shared memory from preprocessing to fpga forward
    shared_trans_arrs = SharedMemoryQueue("trans",num_shared_slots,  [(320,320,3)]+input_shapes)

    # shared memory from fpga forward to postprocessing
    shared_output_arrs = SharedMemoryQueue("output",num_shared_slots, output_shapes)

    # shared memory from postprocessing to display
    shared_display_arrs = SharedMemoryQueue("display",num_shared_slots, [320*320*3])

    cam_process = mp.Process(target=cam_loop,args=(shared_frame_arrs,ready_fpga, ))
    detect_process1 = mp.Process(target=detect_pre,args=(shared_frame_arrs,shared_trans_arrs, ))

    detect_process2 = mp.Process(target=fpga_process,args=(args.vitisrundir, shared_trans_arrs, shared_output_arrs, ready_fpga))
    detect_process3 = mp.Process(target=detect_post,args=(shared_output_arrs, face_q, ))
    show_process = mp.Process(target=show_loop,args=(face_q, ))

#    start_time = time.time()     
    cam_process.start()
    detect_process1.start()
    detect_process2.start()
    detect_process3.start()
    show_process.start()

    # Waits for cam_process to finish video...
    show_process.join()
#    end_time = time.time()
#
#    print('total process time: {0} seconds'.format(end_time - start_time))

    # Now kill remaining processes...
    cam_process.terminate()
    detect_process1.terminate()
    detect_process2.terminate()
    detect_process3.terminate()
    show_process.terminate()    
