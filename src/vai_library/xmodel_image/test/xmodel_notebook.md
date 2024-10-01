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
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# prerequisists

```python

```

```python
import os
import sys
os.environ['LD_LIBRARY_PATH'] = ":".join([os.path.expanduser("~/.local_debug_cloud/lib") , 
                                           '/opt/vitis_ai/conda/lib',
                                           os.environ.get('LD_LIBRARY_PATH','')])
```


```python
os.getenv('LD_LIBRARY_PATH')
```

```python
os.environ['LD_LIBRARY_PATH'] 
```

```python
import xir
```

# classification

the sample input image

```python
Image('/usr/share/vitis_ai_library/samples/classification/sample_classification.jpg')
```


the reference result


```python
!env LD_LIBRARY_PATH=$HOME/.local_debug_cloud/lib \
     DEBUG_XMODEL_IMAGE=1 \
     DEBUG_XMODEL_JIT=1 \
     $HOME/build_debug_cloud/Vitis-AI-Library/classification/test_classification \
     resnet_v1_50_tf \
     /usr/share/vitis_ai_library/samples/classification/sample_classification.jpg
```

the `xmodel_image` result

```python
!env LD_LIBRARY_PATH=$HOME/.local_debug_cloud/lib:/opt/vitis_ai/conda/lib \
    DEBUG_XMODEL_IMAGE=1 \
    DEBUG_XMODEL_JIT=1 \
    $HOME/build_debug_cloud/Vitis-AI-Library/xmodel_image/test_xmodel \
    /workspace/aisw/Vitis-AI-Library/xmodel_image/models/resnet_v1_50_tf/resnet_v1_50_tf.xmodel \
    /usr/share/vitis_ai_library/samples/classification/sample_classification.jpg
```


# densebox


the sample input image

```python
Image('/usr/share/vitis_ai_library/samples/facedetect/sample_facedetect.jpg')
```

the reference result


```python
!env LD_LIBRARY_PATH=$HOME/.local_debug_cloud/lib \
     DEBUG_XMODEL_IMAGE=1 \
     DEBUG_XMODEL_JIT=1 \
     DEBUG_DEMO=1 \
     $HOME/build_debug_cloud/Vitis-AI-Library/overview/test_jpeg_facedetect densebox_320_320 \
     /usr/share/vitis_ai_library/samples/facedetect/sample_facedetect.jpg
```


```python
Image('sample_facedetect_result.jpg')
```

the `xmodel_image` result

```python
!env LD_LIBRARY_PATH=$HOME/.local_debug_cloud/lib:/opt/vitis_ai/conda/lib \
    DEBUG_XMODEL_IMAGE=1 \
    DEBUG_XMODEL_JIT=1 \
    $HOME/build_debug_cloud/Vitis-AI-Library/xmodel_image/test_xmodel \
    /workspace/aisw/Vitis-AI-Library/xmodel_image/models/resnet_v1_50_tf/resnet_v1_50_tf.xmodel \
    /usr/share/vitis_ai_library/samples/classification/sample_classification.jpg
```
