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
```
# Download the data.
cd $VAI_ALVEO_ROOT/apps/face_detect/FDDB
wget http://tamaraberg.com/faceDataset/originalPics.tar.gz
# Extract the data.
tar -xvf originalPics.tar.gz
cd ..
```

## Run Inference on sample images
```
cd $VAI_ALVEO_ROOT/apps/face_detect/
./test_visual.sh face_detection 
The output images are stored in output folder.
```
## Run Inference on entire dataset and calculate precision

```
cd $VAI_ALVEO_ROOT/apps/face_detect/
cd evaluation
#build evalution tools
make
cd ..
./test_precision.sh face_detection 
The output will be an array. [ 0.87894  96.  0.9284]. The recall is 87.89@fp=96.
```

## Run Inference on Video
This is a demo application showing how face detection model can be ran on the FPGA. Frames from a video are streamed into our hardware accelerator. Some post processing is performed in the CPU, such as NMS.   

```
cd $VAI_ALVEO_ROOT/apps/face_detect/
./test_video.sh face_detection example.mp4
```
**Note** : User needs to provide the full path of example.mp4


## Check Precision from an existing detections
If you have already saved all detections to a text file (say `FDDB_results.txt`) and you want to measure its precision, you can use following method:

``` sh
cd $VAI_ALVEO_ROOT/apps/face_detect/
evaluation/evaluate             \
    -a FDDB/FDDB_annotations.txt  \
    -d FDDB_results.txt           \
    -i FDDB/                      \
    -l FDDB/FDDB_list.txt         

python roc.py
```

**Note** : If it throws error regarding missing opencv libs, please extend `LD_LIBRARY_PATH` with `$CONDA_PREFIX/lib`
```sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
