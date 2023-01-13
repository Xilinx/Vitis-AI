# Fall detection using Accelerated Optical Flow

:pushpin: **Note:** This application can be run only on Alveo-U200 platform.

:pushpin: **Note:** Use VAI2.5 setup to run this applicaion


## Table of Contents

- [Introduction](#introduction)
- [Setting Up the Target](#Setting-Up-the-Target-Alveo-U200)
- [Setup and Build the Kernels](#building-kernels)
- [Prepare the Data](#prepare-the-data)
- [Running the Application](#running-the-application)
- [Performance](#performance)

## Introduction
This application demonstrates the acceleration of TVL1 Optical Flow algorithm and modified VGG16 network which takes 20 channel input. Every consecutive pair of frames are resized and fed to Optical Flow kernel which generates a 2 dimensional Optical Flow vector. 10 of such consecutive vectors are stacked together and passed to the inference model, which is trained to classify fall from no-fall.

## Setting Up the Target Alveo U200
**Refer to [Setup Alveo Accelerator Card](../../../setup/alveo) to set up the Alveo Card.**

**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../README.md#Installation)**

* Download the [waa_u200_xclbins_v2_0_0](https://www.xilinx.com/bin/public/openDownload?filename=dpu.xclbin) xclbin tar and install xclbin.
```sh
wget https://www.xilinx.com/bin/public/openDownload?filename=dpu.xclbin -O dpu.xclbin
export XLNX_VART_FIRMWARE=dpu.xclbin
```

## Building Kernels

#### Build common kernels
```sh
cd ${VAI_HOME}/tools/AKS
./cmake-kernels.sh --name=dpu
./cmake-kernels.sh --name=optical_flow_fpga
```

#### Build fall-detection kernels
```sh
cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/fall_detection
./cmake-kernels.sh --clean
```
#### Compile main file
```sh
cd ${VAI_HOME}/examples/Whole-App-Acceleration/apps/fall_detection
./cmake-src.sh --clean
```

### Variables

> These are found in src/main.cpp

* `OFStackSize`: Determines number of Optical Flow (pair) vectors to be stacked together to create a blob
* `OFGraph`: AKS Graph that is used to run Optical Flow on given pair of previous and current frames
* `OFInferenceGraph`: AKS Graph that is used to run classification inference on stacked Optical Flow (flowx & flowy) vectors


### Workflow

* Each `video` and `images directory` will be processed in a separate thread.
* thread x: One for each stream (either video or image-directory).
* Pass every consecutive pair of frames to runOpticalFlow. Pass the future object returned from runOpticalFlow to runOFInference. Keep filling up the stack with the value returned from future objects in sequence.
* Once the stack is full (size=OFStackSize), Copy the elements to a container and pass it to OFinfernece.
* DPUCADF8H Runner takes batches of each size 4. Therefore, if we are going to stack 10 optical flow vectors and pass them as a single unit for the inference, we want to maintain OFStackSize of 13 (stackSize + bacthSize - 1).


### Graph Zoo

#### `graph_optical_flow_fpga.json` contains the TVL1 OpticalFlow graph that runs on FPGA.
* **DualTVL1OpticalFlow**: Run TVL1 Optical Flow; out flowx and flowy vectors
* **optical_flow_postproc**: Resize and bound the pixel values of flow vectors to `[-bound, bound]`, normalize to [0, 255] and perform mean subtraction (mean=127)


#### `graph_of_inference.json` contains the inference graph for classification of stacked optical flow vectors
* **of_infer_preprocess**: Merge all the input blobs (10 input blobls, each has 2 channels) to create a single blob of 20 channels
* **of_infer_runner**: Pass the input data to the modified vgg16 network (model files are found under graph_zoo/meta_vgg_fall_detection)
* **fc_sigmoid**: Perform last FC operation and Sigmoid to get the probability of not-fall (FC-Sigmoid)
* **calc_accuracy**: Calculate the accuracy of the network given gt.txt and write the probability of no-fall and save it in the directory mentioned in visualize parameter.


## Prepare the data
> :pushpin: **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

* Download the [fall-\*-cam0-rgb.zip and adl-\*-cam0-rgb.zip] dataset and extract them to `urfd_dataset` folder from the official URFD dataset link: http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html
```sh
mkdir urfd_dataset

for i in {01..30}; do
  wget http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-${i}-cam0-rgb.zip && unzip fall-${i}-cam0-rgb.zip -d urfd_dataset && rm fall-${i}-cam0-rgb.zip
done

for i in {01..40}; do
  wget http://fenix.univ.rzeszow.pl/~mkepski/ds/data/adl-${i}-cam0-rgb.zip && unzip adl-${i}-cam0-rgb.zip -d urfd_dataset && rm adl-${i}-cam0-rgb.zip
done
```

* If the accuracy, recall and other metrics are to be evaluated, we need to pass ground_truth file to calc_accuracy node of graph_of_inference.json.
* Download urfall-cam0-falls.csv file from the same website
```sh
wget http://fenix.univ.rzeszow.pl/~mkepski/ds/data/urfall-cam0-falls.csv
```
* Run convert_csv_to_gt_file.py to create gt.txt which contains the label by filename
```sh
python convert_csv_to_gt_file.py
```


## Running the Application

```sh
./run.sh -d <directory> [-t <max-concurrent-streams> -v <verbose-level> -perf_diff (No args; To compare the performance with software and hardware optical flow) -accu_diff (No args; To compare the accuracy with software and hardware optical flow)]

[sample]
./run.sh -d urfd_dataset -t 20
```

:pushpin: **Note:** The base model used for inference is modified VGG16, which is a heavy model. Notice the increase in the memory consumption of the host while running the application and set the max-concurrent-stream `-t` accordingly.

* Input passed to `run.sh` (`<directory>`) should contain either videos and/or directories of images. The filename of the images in the images directory should be chronological.

* Predictions stored in separate text files for each stream in `output_infer_urfd` (or the directory as mentioned in the `visualize` parameter of `calc_evaluation` node of `graph_of_inference.json`)

```sh
[sample structure]

.
└── directory
    ├── video1.mp4
    ├── video2.avi
    ├── img_dir1
    |   ├── img1.jpeg
    |   ├── img2.jpeg
    |   ├── img3.jpeg
    |   └── img4.jpeg
    └── dir2
        ├── xxx_001.jpg
        ├── xxx_002.jpg
        ├── xxx_003.jpg
        └── xxx_004.jpg
```

## Performance

Performance metrics observed on urfd_dataset (70 streams):

* Accuracy: 0.973339
* Sensitivity/Recall: 0.163494
* Specificity: 0.989558
* FAR/FPR: 0.0104421
* MDR/FNR: 0.836506
* Total frames inferred: 11140
* Total timetaken: 63.566 seconds..
* Throughput (fps): 175.25


**Note that the overall performance of the application depends on the available system resources.**

URFD dataset has 70 streams, each containing 170 frames on average.

Following table shows the comparison of end-to-end application's throughput with OpenCV DualTVL1 algorithm on CPU against accelerated TVL1 Optical Flow on FPGA on `70 streams`.

| Fall detection | E2E Throughput (FPS) on 70 streams | Improvement in throughput with accelerated<br>TVL1 |
|:-:|:-:|:-:|
| with hardware accelerated<br>TVL1 | 175.25 | - |
| with OpenCV's TVL1<br>(1 thread) | 26.55 | 560.07 % |
| with OpenCV's TVL1<br>(12 threads) | 137.98 | 27.01 % |


## Write prediction probabilities to video

Reads the probabilities written by the accuracy kernel of graph_of_inference, writes them on the images and saves as videos.

```
usage: write_to_video.py [-h] -id INPUT_DIR -im IMAGES_DIR [-o OUTPUT_DIR]
                         [-f FPS] [-w WIDTH] [-l HEIGHT] [-t THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -id INPUT_DIR, --input-dir INPUT_DIR
                        Path to the directory which has the files containing
                        the probabilities by the image file or video frame
                        index
  -im IMAGES_DIR, --images-dir IMAGES_DIR
                        Path to the directory which has the directory of image
                        sequences or videos
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
  -f FPS, --fps FPS     FPS of the output video
  -w WIDTH, --width WIDTH
                        Width of the output video
  -l HEIGHT, --height HEIGHT
                        Height of the output video
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold used to color code the probability
```

```sh
conda activate vitis-ai-tensorflow

python write_to_video.py \
    --input-dir output_infer_urfd \
    --images-dir urfd_dataset \
    --output-dir output_video_urfd

conda deactivate
```
