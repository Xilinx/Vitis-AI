''' This module just loads an image and preprocess it for GoogleNet/ResNet.
You can call this kernel from a graph.json as follows:

    {
      "node_name": "preproc", 
      "node_params" : {
        "Python": {
          "module" : "pyimread",
          "kernel" : "pyImread",
          "pyargs" : [
            "'net_w' : 224",
            "'net_h' : 224",
            "'net_c' : 3",
            "'mean': [ 104.007, 116.669, 122.679]"
              ]
        }
      },
      "next_node": ["resnet50_fpga"]
    },

'''
import aks
import cv2
import numpy as np

class pyImread:
  def __init__(self, params=None):
    pystr = params.getString("pyargs");
    self.params = eval(pystr)
    self.net_w = self.params['net_w']
    self.net_h = self.params['net_h']
    self.net_c = self.params['net_c']
    self.mean = np.array(self.params['mean'], dtype=np.float32).reshape(1, 1, 3)

  def exec_async(self, inputs, params, dynParams):
    outputs = []
    file_path = dynParams.imagePath
    img = cv2.imread(file_path)
    imgr = cv2.resize(img, (self.net_w, self.net_h)).astype(np.float32)
    imgm = imgr - self.mean
    imgb = imgm.transpose(2, 0, 1)
    imgf = imgb.copy()
    outputs.append(imgf)
    return outputs

  def wait(opParams):
    pass

  def __del__(self):
    pass
