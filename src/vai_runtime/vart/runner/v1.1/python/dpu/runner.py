"""
Copyright 2022-2023 Advanced Micro Devices Inc.

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


from ctypes import *
import numpy as np
import json
import os, sys
import re
import threading

class Tensor(Structure):
  _fields_ = [
    ('name', c_char_p),
    ('dims', POINTER(c_int32)),
    ('ndims', c_int32),
    ('dtype', c_int32)
  ]

class Runner:
  # tensor format enum
  TensorFormat = type('', (), {})()
  TensorFormat.NCHW = 0
  TensorFormat.NHWC = 1

  def __init__(self, path):
    metaFile = os.path.join(path, "meta.json")
    if not os.path.isfile(metaFile):
      raise AssertionError("meta.json file %s not found" % metaFile)
      
    # select .so file based on path/meta.json
    with open(metaFile) as f:
      meta = json.load(f)
      libFile = self._parse_path(meta['lib'])

    if not libFile or not os.path.isfile(libFile):
      raise AssertionError("C++ library .so file %s not found" % libFile)

    self._libFile = os.path.abspath(libFile)
    self._lib = cdll.LoadLibrary(self._libFile)
    self._inputCptrs = {}
    self._outputCptrs = {}

    self._lib.DpuPyRunnerCreate.argtypes = [c_char_p]
    self._lib.DpuPyRunnerCreate.restype = c_void_p
    self._lib.DpuPyRunnerGetInputTensors.argtypes = [c_void_p,
      POINTER(c_void_p), POINTER(c_int)]
    self._lib.DpuPyRunnerGetOutputTensors.argtypes = [c_void_p, 
      POINTER(c_void_p), POINTER(c_int)]
    self._lib.DpuPyRunnerGetTensorFormat.argtypes = [c_void_p]
    self._lib.DpuPyRunnerGetTensorFormat.restype = c_int
    self._lib.DpuPyRunnerExecuteAsync.argtypes = [c_void_p, 
      POINTER(np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS")),
      POINTER(np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS")),
      c_int, POINTER(c_int)]
    self._lib.DpuPyRunnerExecuteAsync.restype = c_int
    self._lib.DpuPyRunnerWait.argtypes = [c_void_p, c_int]
    self._lib.DpuPyRunnerWait.restype = c_int
    self._lib.DpuPyRunnerDestroy.argtypes = [c_void_p]

    self._runner = self._lib.DpuPyRunnerCreate(path.encode('utf-8'))

  def get_input_tensors(self):
    ptr = c_void_p()
    n = c_int(0)
    self._lib.DpuPyRunnerGetInputTensors(self._runner, byref(ptr), byref(n))
    tensors = []
    for i in range(n.value):
      tensors.append(Tensor.from_address(ptr.value + (i*sizeof(Tensor))))
    return tensors

  def get_output_tensors(self):
    ptr = c_void_p()
    n = c_int(0)
    self._lib.DpuPyRunnerGetOutputTensors(self._runner, byref(ptr), byref(n))
    tensors = []
    for i in range(n.value):
      tensors.append(Tensor.from_address(ptr.value + (i*sizeof(Tensor))))
    return tensors

  def get_tensor_format(self):
    return self._lib.DpuPyRunnerGetTensorFormat(self._runner)

  def execute_async(self, inputs, outputs):
    """
      Args:
        inputs: list of numpy arrays
        outputs: list of numpy arrays

        order of numpy arrays in inputs/outputs must match 
          the order in get_input_tensors() and get_output_tensors()
    """
    status = c_int(0)

    # NOTE: every thread needs different input/output pointers
    ret = self._lib.DpuPyRunnerExecuteAsync(self._runner, 
			self._numpy_list_2_cptr_list(inputs, self._inputCptrs), 
      self._numpy_list_2_cptr_list(outputs, self._outputCptrs), 
      inputs[0].shape[0], byref(status))

    if status.value != 0:
      raise RuntimeError("Runner.execute_async could not enqueue new DPU job")

    return ret

  def _numpy_list_2_cptr_list(self, nplist, ptrlist):
    tid = threading.get_ident()
    if tid not in ptrlist or len(nplist) != len(ptrlist[tid]):
      ptrlist[tid] = (np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS") * len(nplist))()
    for i, tensor in enumerate(nplist):
      ptrlist[tid][i] = tensor.ctypes.data

    return ptrlist[tid]

  def _parse_path(self, path):
    """
      Translate any {STRING} in 'path' to os.environ["STRING"]
      E.g., {XILINX_ROOT}/path/to/file to /opt/xilinx/path/to/file
    """
    retpath = path
    regex = r"\{(.*?)\}"
    matches = re.finditer(regex, path, re.MULTILINE | re.DOTALL)
    for matchNum, match in enumerate(matches):
      word = match.group(1)
      retpath = retpath.replace("{"+word+"}", os.environ[word])

    return retpath

  def wait(self, job_id):
    return self._lib.DpuPyRunnerWait(self._runner, job_id)

  def __del__(self):
    if hasattr(self, '_lib') and self._lib \
      and hasattr(self, '_runner') and self._runner:
      self._lib.DpuPyRunnerDestroy(self._runner)
