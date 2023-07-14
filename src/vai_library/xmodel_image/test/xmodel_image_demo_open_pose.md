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


# pose detection


## load a sample input image

```python
img = cv2.imread('/usr/share/vitis_ai_library/samples/openpose/sample_openpose.jpg')
# imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

## create an `XmodelImage` object

```python
model = xmodel_image.XmodelImage('/workspace/aisw/Vitis-AI-Library/xmodel_image/models/openpose_pruned_0_3/openpose_pruned_0_3.xmodel')
```

## run the model

```python
results = model.run([img])
width = img.shape[1]
height = img.shape[0]

def to_point (point):
    return (int(point.x * width), int(point.y * height));

def draw_point(valid, point):
    if not valid:
        return
    color = (255,255,0)
    thickness = 2
    radius = 4
    center = to_point(point)
    cv2.circle(img, center, radius, color, thickness)

def draw_line(a, b, valid):
    if not valid:
        return
    cv2.line(img, to_point(a), to_point(b), (255, 255, 0), 3, 4);


for key_point in results[0].pose_detect_result.key_point:
    draw_point(key_point.HasField("right_shoulder"), key_point.right_shoulder)
    draw_point(key_point.HasField("right_elbow"), key_point.right_elbow)
    draw_point(key_point.HasField("right_wrist"), key_point.right_wrist)
    draw_point(key_point.HasField("left_shoulder"), key_point.left_shoulder)
    draw_point(key_point.HasField("left_elbow"), key_point.left_elbow)
    draw_point(key_point.HasField("left_wrist"), key_point.left_wrist)
    draw_point(key_point.HasField("right_hip"), key_point.right_hip)
    draw_point(key_point.HasField("right_knee"), key_point.right_knee)
    draw_point(key_point.HasField("right_ankle"), key_point.right_ankle)
    draw_point(key_point.HasField("left_hip"), key_point.left_hip)
    draw_point(key_point.HasField("left_knee"), key_point.left_knee)
    draw_point(key_point.HasField("left_ankle"), key_point.left_ankle)
    draw_point(key_point.HasField("head"), key_point.head)
    draw_point(key_point.HasField("neck"), key_point.neck)
    draw_line(key_point.head, key_point.neck, key_point.HasField("head") and key_point.HasField("neck"))
    draw_line(key_point.neck, key_point.right_shoulder,
              key_point.HasField("neck") and key_point.HasField("right_shoulder"))
    draw_line(key_point.neck, key_point.left_shoulder,
              key_point.HasField("neck") and key_point.HasField("left_shoulder"))
    draw_line(key_point.right_shoulder, key_point.right_elbow,
              key_point.HasField("right_shoulder") and key_point.HasField("right_elbow"))
    draw_line(key_point.right_elbow, key_point.right_wrist,
              key_point.HasField("right_elbow") and key_point.HasField("right_wrist"))
    draw_line(key_point.left_shoulder, key_point.left_elbow,
              key_point.HasField("left_shoulder") and key_point.HasField("left_elbow"))
    draw_line(key_point.left_elbow, key_point.left_wrist,
              key_point.HasField("left_elbow") and key_point.HasField("left_wrist"))
    draw_line(key_point.neck, key_point.right_hip,
              key_point.HasField("neck") and key_point.HasField("right_hip"))
    draw_line(key_point.right_hip, key_point.right_knee,
              key_point.HasField("right_hip") and key_point.HasField("right_knee"))
    draw_line(key_point.right_knee, key_point.right_ankle,
              key_point.HasField("right_knee") and key_point.HasField("right_ankle"))
    draw_line(key_point.neck, key_point.left_hip, key_point.HasField("neck") and key_point.HasField("left_hip"))
    draw_line(key_point.left_hip, key_point.left_knee,
              key_point.HasField("left_hip") and key_point.HasField("left_knee"))
    draw_line(key_point.left_knee, key_point.left_ankle,
              key_point.HasField("left_knee") and key_point.HasField("left_ankle"))

imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```
