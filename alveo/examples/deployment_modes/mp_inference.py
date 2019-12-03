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
from __future__ import print_function
import timeit
import numpy as np
from vai.dpuv1.rt import xdnn, xdnn_io
from multiprocessing import Pool

# Globals in parent, parent will initialize
pool = None
pFpgaRT = None

# Globals in children, each child will initialize its own copy
cFpgaRT = None
cCpuRT = None
fcOutput = None
labels = None

class InferenceEngine(object):
  def __init__(self):
    global pool
    global pFpgaRT
    self.args = xdnn_io.processCommandLine() # Get command line args
    pool = Pool(self.args['numproc']) # Depends on new switch, new switch is not added to xdnn_io.py as of yet

    # Split Images into batches - list of lists
    self.batches = []
    self.all_image_paths = xdnn_io.getFilePaths(self.args['images']) #[0:10000]
    for i in range(0,len(self.all_image_paths),self.args['batch_sz']):
      self.batches.append(self.all_image_paths[i:i+self.args['batch_sz']])

    pFpgaRT = xdnn.XDNNFPGAOp(self.args) # Parent process gets handle
    self.args['inShape'] = (self.args['batch_sz'],) +  tuple(tuple(pFpgaRT.getInputDescriptors().values())[0][1:]) # Save input shape for children
    self.mpid = pFpgaRT.getMPID() # Save handle to use in child processes

  def pre_process(self,image_paths,fpgaInput):
    input_buffer = list(fpgaInput.values())[0] # Assume the first network input needs to have image data loaded
    for index, image_path in enumerate(image_paths):
      input_buffer[index, ...], _ = xdnn_io.loadImageBlobFromFile(
        image_path, self.args['img_raw_scale'], self.args['img_mean'],
        self.args['img_input_scale'], self.args['inShape'][2], self.args['inShape'][3])

  def post_process(self,image_paths,fpgaOutput):
    global labels
    global fcOutput
    global cCpuRT
    if cCpuRT is None:
      cCpuRT = xdnn.XDNNCPUOp(self.args['weights'])
      fcOutput = np.empty((self.args['batch_sz'], self.args['outsz'],), dtype=np.float32, order='C')
      labels = xdnn_io.get_labels(self.args['labels'])

    input_buffer = list (fpgaOutput.values())[0] # Assume the first network output will feed CPU layers
    cCpuRT.computeFC(input_buffer, fcOutput)
    softmaxOut = cCpuRT.computeSoftmax(fcOutput)
    if self.args['golden']:
      xdnn_io.printClassification(softmaxOut, image_paths, labels)

  def Infer(self,pre_process,post_process,image_paths):
    global cFpgaRT
    if cFpgaRT is None:
      cFpgaRT = xdnn.XDNNFPGAOp(self.args,AcquireID=self.mpid)

    fpgaInput  = cFpgaRT.getInputs()
    fpgaOutput = cFpgaRT.getOutputs()
    pre_process(image_paths, fpgaInput)
    cFpgaRT.execute(fpgaInput, fpgaOutput)
    post_process(image_paths, fpgaOutput)
    return 0

  def run(self):
    startTime = timeit.default_timer()
    results = [ pool.apply(self.Infer,args=(self.pre_process,self.post_process,image_paths,)) for image_paths in self.batches ]
    endTime = timeit.default_timer()
    print( "%g images/s" % ( float(len(self.all_image_paths)) / (endTime - startTime )  ))
    return results

if __name__ == '__main__':
  results = InferenceEngine().run()
