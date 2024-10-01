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


# lane detection


## load a sample input image

```python
img = cv2.imread('/usr/share/vitis_ai_library/samples/lanedetect/sample_lanedetect.jpg')
# imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

## create an `XmodelImage` object

```python
model = xmodel_image.XmodelImage('/workspace/aisw/Vitis-AI-Library/xmodel_image/models/vpgnet_pruned_0_99/vpgnet_pruned_0_99.xmodel')
```

## run the model

```python
import numpy as np
results = model.run([img])
pts = []
colors = [(255,255,0),
          (0,255,255),
          (255,0,255)]
thickness = 2
isClosed = True
r = results[0].roadline_result
h = img.shape[0]
w = img.shape[1]
for line in r.line_attr:
    pts = np.asarray([[int(point.x * w), int(point.y * h)] for point in line.point ])
    cv2.polylines(img, [pts], isClosed, colors[line.type % len(colors)], thickness)
imshow(img)
```
