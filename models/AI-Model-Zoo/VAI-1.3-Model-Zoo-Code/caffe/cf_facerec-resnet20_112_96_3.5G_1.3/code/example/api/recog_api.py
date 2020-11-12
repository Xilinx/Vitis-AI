import numpy as np
import scipy.misc
import scipy.io
import datetime
import cv2
import sys
import cPickle as cp

class Recog(object):
  def __init__(self):
    self.input_mean_value_=127.5
    self.input_scale_=0.0078125
    self.feature_name_='Addmm_1'
    
    self.input_height_=112
    self.input_width_=96
    self.caffe_path_=""
    self.force_gray_=False
  def model_init(self,caffe_python_path,model_path,def_path,input_height,input_width,force_gray=False):
    sys.path.insert(0,caffe_python_path)
    import caffe
    self.caffe_path_=caffe_python_path
    self.input_height_=input_height
    self.input_width_=input_width
    self.input_channels_=3
    self.force_gray_=force_gray
    if force_gray:
      self.input_channels_=1
    self.net_=caffe.Net(def_path,model_path,caffe.TEST)
    self.transformer_=caffe.io.Transformer({'data': (1,self.input_channels_,input_height,input_width)})
    
    self.transformer_.set_channel_swap('data',(2,1,0))
    self.net_.blobs['data'].reshape(1,self.input_channels_,self.input_height_, self.input_width_)
  def get_feature(self,image):
    import caffe
    image_w=image.shape[1]
    image_h=image.shape[0]
    image=cv2.resize(image,(self.input_width_,self.input_height_))
    if self.force_gray_:
      image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      #image=np.reshape(image,(image.shape[0],image.shape[1],1))
    else:
      self.transformer_.set_transpose('data', (2,0,1))
    transformed_image=self.transformer_.preprocess('data',image)
    transformed_image=(transformed_image-self.input_mean_value_)*self.input_scale_
    sz=image.shape
    self.net_.blobs['data'].reshape(1, self.input_channels_, sz[0], sz[1])
    self.net_.blobs['data'].data[0, ...] = transformed_image
    output = self.net_.forward()
    #feature=output[self.feature_name_]
    feature=self.net_.blobs[self.feature_name_].data
   
    return feature

  def get_feature_batch(self,images):
    import caffe
    self.net_.blobs['data'].reshape(len(images), self.input_channels_, self.input_height_, self.input_width_)
    for image_id in range(len(images)):
      image=images[image_id]
      image=cv2.resize(image,(self.input_width_,self.input_height_))
      if self.force_gray_:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #image=np.reshape(image,(image.shape[0],image.shape[1],1))
      else:
        self.transformer_.set_transpose('data', (2,0,1))
      transformed_image=self.transformer_.preprocess('data',image)
      transformed_image=(transformed_image-self.input_mean_value_)*self.input_scale_
      self.net_.blobs['data'].data[image_id, ...] = transformed_image
    output = self.net_.forward()
    feature=output[self.feature_name_]
    print feature.shape
    #feature=self.net_.blobs[self.feature_name_]['data']
