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
# Our custom FPGA One-shot layer

from vai.dpuv1.rt import xdnn, xdnn_io
import numpy as np

class RunFPGA:
  def __init__(self, param_str):
    self.input_mean_value_=128.0
    self.input_scale_=1.0

    param_dict = eval(param_str) # Get args from prototxt

    self._args = xdnn_io.make_dict_args(param_dict)
    self._numPE = self._args["batch_sz"] # Bryan hack to determine number of PEs in FPGA

    # Establish FPGA Communication, Load bitstream
    ret, handles = xdnn.createHandle(self._args["xclbin"], "kernelSxdnn_0")
    if ret != 0:
      raise Exception("Failed to open FPGA handle.")

    self._args["scaleB"] = 1
    self._args["PE"] = -1

    # Instantiate runtime interface object
    self._fpgaRT = xdnn.XDNNFPGAOp(handles, self._args)

    self._parser = xdnn.CompilerJsonParser(self._args["netcfg"])

    self._indictnames = self._parser.getInputs()
    self._outdictnames =  self._parser.getOutputs()

    input_shapes = map(lambda x: tuple(x), self._parser.getInputs().itervalues())
    output_shapes = map(lambda x: tuple(x), self._parser.getOutputs().itervalues())

    self._indict = {}
    for i,name in enumerate(self._indictnames):
        self._indict[name] = np.empty(input_shapes[i],dtype=np.float32)

    self._outdict = {}
    for i,name in enumerate(self._outdictnames):
        self._outdict[name] = np.empty(output_shapes[i],dtype=np.float32)

  # bottom and top are arrays of numpy objects in shared memory
  def forward_async(self, bottom, top, id):
    # Call FPGA

    for i,name in enumerate(self._indictnames):
        self._indict[name] = bottom[i]

    for i,name in enumerate(self._outdictnames):
        self._outdict[name] = top[i]

    self._fpgaRT.exec_async(self._indict, self._outdict, id)

    return id

  def forward(self, bottom, top, id):

    outdict = self.forward_async(self, bottom, top, id)
    self._fpgaRT.get_result(id)
