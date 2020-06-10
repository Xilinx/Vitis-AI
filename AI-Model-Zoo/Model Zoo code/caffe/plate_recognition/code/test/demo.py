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

from test import *


if __name__ == "__main__":
    model_config = ModelConfig()
    plate_recognition_model = PlateRecognitionModel(model_config)
    image_path = "../../data/plate_recognition_val/plate_val/123331283_京P029E2_20171201154734_7.jpg"
    image = cv2.imread(image_path)
    number_pred, color_pred = plate_recognition_model.predict_single_image(image)
    ## run command if using docker to display Chinese characters set. (command: export LANG=C.UTF-8)
    print("plate number: ",  number_pred)
    print("plate color: ", color_pred)
