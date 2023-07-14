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
---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.8.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# prerequisists

```python
from vitis.ai import xmodel_image
import cv2
from imshow import imshow
xmodel_image.init_glog("/tmp/a")
%env DEBUG_DPU_RUNNER=0
```

# face feature extraction


## load a sample input image

```python
img1 = cv2.imread('/usr/share/vitis_ai_library/samples/facefeature/images/0835_A.jpg')
img2 = cv2.imread('/usr/share/vitis_ai_library/samples/facefeature/images/0835_C.jpg')
imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
```

## create an `XmodelImage` object

```python
model = xmodel_image.XmodelImage('/workspace/aisw/Vitis-AI-Library/xmodel_image/models/facerec_resnet20/facerec_resnet20.xmodel')
```

## run the model

```python
results = model.run([img1])
import numpy as np
f1 = np.asarray(results[0].face_feature_result.float_vec)
results = model.run([img2])
f2 = np.asarray(results[0].face_feature_result.float_vec)
# print("f1={}".format(f1))
# print("f2={}".format(f2))
c = np.dot(f1,f2) / np.sqrt (np.dot(f1,f1) * np.dot(f2,f2))
print("correlation is {}".format(c))
```
