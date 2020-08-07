'''
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
'''



from ctypes import *
import sys
import numpy as np
import cv2

class PreProcess:

	def __init__(self,xclbin,kernelname,deviceid):
		self.xclbin = xclbin
		self.kernelName = kernelname
		self.deviceIdx = deviceid
		self.lib = cdll.LoadLibrary('/usr/lib/pp_pipeline.so')
		self.lib.pp_kernel_init.argtypes = [c_void_p,
									c_char_p,
									c_char_p,
									c_int]
		self.handle= c_void_p()
		ret=self.lib.pp_kernel_init(pointer(self.handle), str(self.xclbin).encode('ascii'), str(self.kernelName).encode('ascii'), self.deviceIdx)
		if ret == -1:
			print("[XPLUSML]  Unable to Create handle for the pre-processing kernel")
			sys.exit()


	def preprocess_input(self, img, rows, cols):

		act_ht = c_int()
		act_wt = c_int()
                
		self.row = c_int()
		self.col = c_int()
		self.row = rows
		self.col = cols
		self.arr = np.empty([3,224,224],dtype=np.float32)
		self.lib.preprocess.argtypes = [c_void_p,
							   np.ctypeslib.ndpointer(c_uint8, flags="C_CONTIGUOUS"),
                                                           c_int,
                                                           c_int,
							   c_void_p,
							   c_void_p,
							   np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS")]
		self.lib.preprocess(pointer(self.handle),
		img, 
                self.row, 
                self.col, 
                byref(act_ht), byref(act_wt), self.arr)
		np.ascontiguousarray(self.arr, dtype=np.float32)

		return self.arr
