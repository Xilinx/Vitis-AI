# Copyright 2022-2023 Advanced Micro Devices Inc.
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

from PIL import Image, ImageDraw, ImageFont
import numpy as np

#mean_vec = np.array([123.68, 116.78, 103.94])
mean_vec = np.array([103.94, 116.78, 123.68])
scale = [0.017, 0.017, 0.017]

class InputData:
    def __init__(self, image_file_path, shape):
        self.shape = shape
        self.data = Image.open(image_file_path)
    
    def preprocess(self):
        shape = self.shape
        resize_data = self.data.resize((shape[2], shape[3]))
        resize_data = np.array(resize_data)

        if (len(resize_data.shape) == 2):
            resize_data = np.expand_dims(resize_data, 2)
            resize_data = np.repeat(resize_data, 3, 2)
        if (resize_data.shape[2] == 4):
            resize_data = resize_data[:, :, :3]

        tran_data = resize_data.transpose(2, 0, 1)
        float_data = tran_data.astype('float32')
        norm_img_data = np.zeros(float_data.shape).astype('float32')
        for i in range(float_data.shape[0]):
            norm_img_data[i,:,:] = (float_data[i,:,:] - mean_vec[i]) * scale[i]
        #norm_img_data = norm_img_data.reshape(1, shape[1], shape[2], shape[3]).astype('float32')
        norm_img_data = norm_img_data.astype('float32')
        return norm_img_data

    
