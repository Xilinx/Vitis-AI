#
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
#!/usr/bin/python3
from vitis.ai import xmodel_image
# from matplotlib import pyplot as plt
import cv2
# xmodel_image.init_glog("/tmp/a")
# %env DEBUG_DPU_RUNNER=0
img = cv2.imread(
    '/usr/share/vitis_ai_library/samples/classification/sample_classification.jpg'
)
model = xmodel_image.XmodelImage(
    '/workspace/aisw/Vitis-AI-Library/xmodel_image/models/resnet_v1_50_tf/resnet_v1_50_tf.xmodel'
)

print("batch = {}".format(model.get_batch()))
results = model.run([img])  #  * model.get_batch())
model = None
print("resuts= {}".format(results))

# /workspace/aisw/vart/dpu-controller/runner-assistant/src/tensor_buffer_allocator_imp.cpp:169
