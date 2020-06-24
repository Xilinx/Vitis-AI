## Tool to convert darknet model to caffe
The convert.py script supports yolov2 and yolov3 models. For conversion, run the below command. 
``` sh
python convert.py yolov3.cfg yolov3.weights yolov3.prototxt yolov3.caffemodel
```
>**:pushpin: NOTE:** This tool has to be run in vitis-ai-caffe conda environmnet.