# Face Detection on Video
This is a demo application showing how a precompiled face detection model can be ran on the FPGA.  
  
Frames from a video are streamed into our hardware accelerator. Some post processing is performed in the CPU, such as NMS.  

FPGA acceleration enables > 500 FPS.

# To run
```
$ cd ml-suite/apps/faceDetect
$ source ../../overlaybins/setup.sh
$ python mp_video.py
```
