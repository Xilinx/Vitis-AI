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

import multiprocessing as mp

import os
import cv2
import time
import ctypes
import threading

from detect_ap2 import det_preprocess, det_postprocess

from vai.dpuv1.rt import xdnn
import numpy as np

##################################################
# CONSTANTS
##################################################
num_shared_slots = 200


# Current version does copies...
# Assumes all types are np.float32/ctypes.c_float
class SharedMemoryQueue:
    def __init__(self, name, length, buf_shapes_list):

        print "Creating SharedMemoryQueue",name
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
        self._buf_sizes_list = map(lambda x: np.prod(x), buf_shapes_list)

        print "Creating Shared Memory with buf_shape_list=",self._buf_shapes_list

        self._shared_memory_arrs = list()
        for i in range(length):
            buf_list = list()
            for buf_size in self._buf_sizes_list:
                buf_list.append(mp.Array(self._mem_type, buf_size))
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
              print "Slot=",i,"Array=",j,"Val=",np_arr




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
    print funcname(),"Putting Frames..."
    start_time = time.time()
    frame_cnt = 0
    for f in frames:
        write_slot = shared_frame_arrs.openWriteId()
        np_arr = shared_frame_arrs.accessNumpyBuffer(write_slot)
        np_arr[0][:] = f
        shared_frame_arrs.closeWriteId(write_slot)
        frame_cnt += 1
    end_time = time.time()
    shared_frame_arrs.close()


    print funcname(),"Video Ending! Frame Count = ",frame_cnt
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

    print "Ran",frame_id,"frames"
    print('detect_preprocessing time: {0} seconds'.format(end_time - start_time))



import run_fpga


# helper thread just to read output from XFDNN and pass onto appropriate queue
def fpga_wait( fpgaRT, qWait, shared_output_arrs):
  numProcessed = 0
  frame_id = 0
  while True:
    write_slot = qWait.get()
    if write_slot is None:
      break

    fpgaRT.get_result(write_slot)
    write_slot_arrs = shared_output_arrs.accessNumpyBuffer(write_slot)

    shared_output_arrs.closeWriteId(write_slot)
    frame_id += 1


def detect_forward(shared_trans_arrs, shared_output_arrs, ready_fpga):

    VAI_ALVEO_ROOT = os.getenv("VAI_ALVEO_ROOT","/opt/ml-suite")
    MLSUITE_PLATFORM = os.getenv("MLSUITE_PLATFORM","alveo-u200")

    param_str = "{\'batch_sz\': 1," +\
                "\'outtrainproto\': None," +\
                "\'input_names\': [u\'data\']," +\
                "\'cutAfter\': \'data\'," +\
                "\'outproto\': \'xfdnn_deploy.prototxt\'," +\
                "\'xdnnv3\': True," +\
                "\'inproto\': \'deploy.prototxt\'," +\
                "\'profile\': False," +\
                "\'trainproto\': None," +\
                "\'weights\': \'deploy.caffemodel_data.h5\'," +\
                "\'netcfg\': \'deploy.compiler.json\'," +\
                "\'quantizecfg\': \'deploy.compiler_quant.json\'," +\
                "\'xclbin\': \'" + VAI_ALVEO_ROOT + "/overlaybins/" + MLSUITE_PLATFORM + "/overlay_4.xclbin\'," +\
                "\'output_names\': [u\'pixel-conv\', u\'bb-output\']," +\
                "\'overlaycfg\': {u\'XDNN_NUM_KERNELS\': u\'2\', u\'SDX_VERSION\': u\'2018.2\', u\'XDNN_VERSION_MINOR\': u\'0\', u\'XDNN_SLR_IDX\': u\'1, 1\', u\'XDNN_DDR_BANK\': u\'0, 3\', u\'XDNN_CSR_BASE\': u\'0x1800000, 0x1810000\', u\'XDNN_BITWIDTH\': u\'8\', u\'DSA_VERSION\': u\'xilinx_u200_xdma_201820_1\', u\'XDNN_VERSION_MAJOR\': u\'3\'}}"

    det = run_fpga.RunFPGA(param_str)
    ready_fpga.put(1)

    qWait = mp.Queue(maxsize=100)
    t = threading.Thread(target=fpga_wait, args=(det._fpgaRT, qWait, shared_output_arrs))
    t.start()

    frame_id = 0
    start_time = None

    while True:
        read_slot = shared_trans_arrs.openReadId()

        if start_time is None:
            start_time = time.time()

        if read_slot is None:
            break

        read_slot_arrs = shared_trans_arrs.accessNumpyBuffer(read_slot)

        write_slot = shared_output_arrs.openWriteId()
        write_slot_arrs = shared_output_arrs.accessNumpyBuffer(write_slot)

        out_dict  = det.forward_async(read_slot_arrs[1:], write_slot_arrs[1:], write_slot)

        shared_trans_arrs.closeReadId(read_slot)

        qWait.put(write_slot)

        frame_id += 1
    end_time = time.time()
    shared_output_arrs.close()

    print('detect forward time: {0} seconds'.format(end_time - start_time))

    qWait.put(None)
    t.join()


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

        face_rects = det_postprocess(read_slot_arrs[1], read_slot_arrs[2], [320,320,3])

        shared_output_arrs.closeReadId(read_slot)
        frame_cnt += 1
        face_q.put(face_rects)

    end_time = time.time()
    face_q.put(None)

    print('detect post processing time: {0} seconds'.format(end_time - start_time))

    total_time = end_time-start_time
    print('Total run: {0} frames in {1} seconds ({2} fps)'.format(frame_cnt, total_time, frame_cnt/total_time))


def show_loop(face_q):

    cv2.namedWindow('face_detection')
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

            cv2.imshow('face_detection', frame)
            cv2.waitKey(40)

        frame_id += 1
    end_time = time.time()
    print('drawing boxes time: {0} seconds'.format(end_time - start_time))



if __name__ == '__main__':

    frame_q = mp.Queue()
    resize_q = mp.Queue()
    trans_q = mp.Queue()
    output_q = mp.Queue()
    face_q = mp.Queue()

    ready_fpga = mp.Queue()

    sharedInputArrs = []

    compilerJSONObj = xdnn.CompilerJsonParser('deploy.compiler.json')

    input_shapes = map(lambda x: tuple(x), compilerJSONObj.getInputs().itervalues())
    output_shapes = map(lambda x: tuple(x), compilerJSONObj.getOutputs().itervalues())

    input_sizes = map(lambda x: np.prod(x), input_shapes)
    output_sizes = map(lambda x: np.prod(x), output_shapes)

    print input_shapes
    print output_shapes

    # shared memory from video capture to preprocessing
    shared_frame_arrs = SharedMemoryQueue("frame",num_shared_slots, [(320,320,3)])

    # shared memory from preprocessing to fpga forward
    shared_trans_arrs = SharedMemoryQueue("trans",num_shared_slots, [(320,320,3)]+input_shapes)

    # shared memory from fpga forward to postprocessing
    shared_output_arrs = SharedMemoryQueue("output",num_shared_slots, [(320,320,3)]+output_shapes)

    # shared memory from postprocessing to display
    shared_display_arrs = SharedMemoryQueue("display",num_shared_slots, [320*320*3])

    cam_process = mp.Process(target=cam_loop,args=(shared_frame_arrs,ready_fpga, ))
    detect_process1 = mp.Process(target=detect_pre,args=(shared_frame_arrs,shared_trans_arrs, ))
    detect_process2 = mp.Process(target=detect_forward,args=(shared_trans_arrs, shared_output_arrs, ready_fpga, ))
    detect_process3 = mp.Process(target=detect_post,args=(shared_output_arrs, face_q, ))
    show_process = mp.Process(target=show_loop,args=(face_q, ))

    start_time = time.time()
    cam_process.start()
    detect_process1.start()
    detect_process2.start()
    detect_process3.start()
    show_process.start()

    # Waits for cam_process to finish video...
    show_process.join()
    end_time = time.time()

    print('total process time: {0} seconds'.format(end_time - start_time))

    # Now kill remaining processes...
    cam_process.terminate()
    detect_process1.terminate()
    detect_process2.terminate()
    detect_process3.terminate()
    show_process.terminate()
