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

# PART OF THIS FILE AT ALL TIMES.

#coding=utf-8
import cv2
from test import *

if __name__ == "__main__":

    model_config = ModelConfig()
    plate_detect_model = PlateDetectionModel(model_config)
    image_path = "val_plate/123346891_京N5S2Z9_20171201161524.jpg"
    image = cv2.imread(image_path)
    image = plate_detect_model.dispay_detection(image) 
    cv2.imwrite('demo.jpg', image)   
