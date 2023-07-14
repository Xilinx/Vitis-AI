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


# face detection


## load a sample input image

```python
img = cv2.imread('/usr/share/vitis_ai_library/samples/facedetect/sample_facedetect.jpg')
imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

## create an `XmodelImage` object

```python
model = xmodel_image.XmodelImage('/workspace/aisw/Vitis-AI-Library/xmodel_image/models/densebox_320_320/densebox_320_320.xmodel')
```

## run the model

```python
results = model.run([img])
for bbox in results[0].detect_result.bounding_box:
    start_point = (int(bbox.top_left.x * img.shape[1]), int(bbox.top_left.y * img.shape[0]))
    end_point = (int((bbox.top_left.x + bbox.size.width) * img.shape[1]),
                 int((bbox.top_left.y  + bbox.size.height) * img.shape[0]) )
    color = (255,255,0)
    thickness = 2
    cv2.rectangle(img, start_point, end_point, color, thickness)

imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```
