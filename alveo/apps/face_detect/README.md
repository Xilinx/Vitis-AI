# Running the face detection model on FPGA 

### Data Preparation

Download Face Detection Data Set and Benchmark (FDDB)dataset. 
```
# Download the data.
$ cd $VAI_ALVEO_ROOT/apps/face_detect/FDDB
$ wget http://tamaraberg.com/faceDataset/originalPics.tar.gz
# Extract the data.
$ tar -xvf originalPics.tar.gz
$ cd ..
```

## Run Inference on sample images
```
$ cd $VAI_ALVEO_ROOT/apps/face_detect/
$ ./test_visual.sh face_detection 
The output images are stored in output folder.
```
## Run Inference on entire dataset and calculate precision

```
$ cd $VAI_ALVEO_ROOT/apps/face_detect/
$ cd ../evaluation
#build evalution tools
$ make
$ cd ..
$ ./test_precision.sh face_detection 
The output will be an array. [ 0.871205 100.  0.928409]. The recall is 87.12@fp=100.
```

## Run Inference on Video
This is a demo application showing how face detection model can be ran on the FPGA. Frames from a video are streamed into our hardware accelerator. Some post processing is performed in the CPU, such as NMS.  

```
$ cd $VAI_ALVEO_ROOT/apps/face_detect/
$ ./test_video.sh face_detection
```
