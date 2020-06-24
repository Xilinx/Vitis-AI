''' This contains postproc kernels of various networks '''

import aks
import numpy as np
import cv2
import os
from detect_ap2_legacy import det_postprocess

class FDDBFaceDetectPostProc:
  ''' Face Detection Post Processing module.
  It is compatible with $(VAI_ALVEO_ROOT)/apps/facedetect.

  Module arguments:
    network_input_dims : Input dimensions to the network, [Height, Width, Channels] 
    img_list    : List of images that are passed to the graph. Results are saved in the same order
    save_result_txt : [Optional] Text file name to store all results for accuracy calculations
                      By default, no results are stored.
    save_result_imgdir : [Optional] Folder to store all results drawn on original images for visualization
                      By default, no results are stored.
  '''

  def __init__(self, params=None):
    pyargs      = params.getString("pyargs")
    self.params = eval(pyargs)

    self.sz             = self.params['network_input_dims'] # [H, W, C]
    self.imglistfile    = self.params['img_list']
    self.results_txt    = self.params.get('save_result_txt', '')
    self.results_imgdir = self.params.get('save_result_imgdir', '')

    self.all_outputs = {}

  # inputs are in the order [bb, pixel_conv]
  def exec_async(self, inputs, nodeParams, dynParams):
    outputs = []
    bb, pixel_conv = inputs
    face_rects = det_postprocess(pixel_conv, bb, self.sz)
    res = np.array(face_rects, np.float32)
    outputs.append(res)

    # Save the results
    if self.results_txt or self.results_imgdir:
      imgPath = dynParams.imagePath
      absImgPath = os.path.abspath(imgPath)
      corrected_boxes  = []
      for rect in face_rects:
        # scale to actual image size for evaluation purpose
        cvimg        = cv2.imread(imgPath)
        h, w         = cvimg.shape[:2]
        rsz_w, rsz_h = self.sz[:2]

        # topx, topy, bottomx, bottomy
        sc_topx  = int(rect[0]*w/rsz_w)
        sc_topy  = int(rect[1]*h/rsz_h)
        sc_bttmx = int(rect[2]*w/rsz_w)
        sc_bttmy = int(rect[3]*h/rsz_h)
        sc_width = sc_bttmx - sc_topx
        sc_height= sc_bttmy - sc_topy

        corrected_boxes.append([sc_topx, sc_topy, sc_bttmx, sc_bttmy, rect[4]])

      self.all_outputs[absImgPath] = corrected_boxes

    return outputs

  def wait(self):
    pass

  def report(self, params):
    dirname = os.path.abspath(os.path.dirname(self.imglistfile))
    with open(self.imglistfile, 'r') as f:
      self.imglist = f.readlines()

    # Create a directory
    if(self.results_imgdir):
      if not os.path.exists(self.results_imgdir):
        os.mkdir(self.results_imgdir)


    # Save all results to text file
    if self.results_txt:
      print("FDDBFaceDetectPostProc : Saving the results to {} ...".format(self.results_txt), flush=True)
      with open(self.results_txt, 'w') as fp:
        for i in range(len(self.all_outputs)):
          img = self.imglist[i].strip()
          fullpath = os.path.join(dirname, img+'.jpg')
          faces = self.all_outputs[fullpath]
          fp.write("{}\n".format(img.strip()))
          fp.write("{}\n".format(len(faces)))
          for (x1, y1, x2, y2, prob) in faces:
            fp.write('%d %d %d %d %f\n' % (x1, y1, x2-x1, y2-y1, prob))

    # if required, save the results as images
    if self.results_imgdir:
      print("FDDBFaceDetectPostProc : Saving the images to {} ...".format(self.results_imgdir), flush=True)
      for i in range(len(self.all_outputs)):
        img = self.imglist[i].strip()
        fullpath = os.path.join(dirname, img+'.jpg')
        faces = self.all_outputs[fullpath]
        cvimg = cv2.imread(fullpath)
        for (x1, y1, x2, y2, prob) in faces:
          cv2.rectangle(cvimg,(x1, y1),(x2, y2),(0,255,0),2)
        cvimgPath = os.path.join(self.results_imgdir, img.replace('/','_')+'.jpg')
        cv2.imwrite(cvimgPath, cvimg)

  def __del__(self):
    pass
