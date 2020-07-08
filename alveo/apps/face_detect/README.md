# Running the face detection model on FPGA 

### Setup
```sh
# Activate Conda Environment
conda activate vitis-ai-caffe 
```
```sh
# Setup
source /workspace/alveo/overlaybins/setup.sh
```

### Data Preparation

Download Face Detection Data Set and Benchmark (FDDB)dataset. 

> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

```sh
# Download the data.
cd $VAI_ALVEO_ROOT/apps/face_detect/FDDB
wget http://tamaraberg.com/faceDataset/originalPics.tar.gz
# Extract the data.
tar -xvf originalPics.tar.gz
cd ..
```

## Run Inference on sample images
```sh
cd $VAI_ALVEO_ROOT/apps/face_detect/
```
Face detection on test_images using face_detection_320_320 model on FPGA and save results in folder output/.
```sh
./test_visual.sh face_detection
```
Face detection on test_images using face_detection_360_640 model on FPGA and save results in folder output/.
```sh
./test_visual.sh face_detection_360_640
```

## Run Inference on entire dataset and calculate precision

Build evalution tools
```sh
cd $VAI_ALVEO_ROOT/apps/face_detect/
wget http://vis-www.cs.umass.edu/fddb/evaluation.tgz
tar -xvf evaluation.tgz 
#ignore the warnings
cp Makefile evaluation/
cd evaluation 
make
cd ..
```

Calculate the precsion of face_detection_320_320 model
```sh
./test_precision.sh face_detection
```
The output will be an array. [ 0.87894  96.  0.9284]. The recall is 87.89@fp=96.

Calculate the precsion of face_detection_360_640 model 
```sh
./test_precision.sh face_detection_360_640
```
The output will be an array. [ 0.883775 99.   0.957]. The recall is 88.37@fp=99.

## Run Inference on Video
This is a demo application showing how face detection model can be ran on the FPGA. Frames from a video are streamed into our hardware accelerator. Some post processing is performed in the CPU, such as NMS.   

```sh
cd $VAI_ALVEO_ROOT/apps/face_detect/
```
Face detection on video using face_detection_320_320 model on FPGA and save results in folder output/.
```sh
./test_video.sh face_detection example.mp4
```
Face detection on video using face_detection_360_640 model on FPGA and save results in folder output/.
```sh
./test_video.sh face_detection_360_640 example.mp4
```

**Note** : User needs to provide the full path of example.mp4


## Check Precision from an existing detections
If you have already saved all detections to a text file (say `FDDB_results.txt`) and you want to measure its precision, you can use following method:

```sh
cd $VAI_ALVEO_ROOT/apps/face_detect/
python roc.py --results <path to result file>
```

**Note** : If it throws error regarding missing opencv libs, please extend `LD_LIBRARY_PATH` with `$CONDA_PREFIX/lib`
```sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
