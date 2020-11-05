# **Setup**
```sh
conda activate vitis-ai-caffe
source /workspace/alveo/overlaybins/setup.sh
```

Note: Optical flow xclbin is to be placed under `/opt/xilinx/overlaybins`
> [sudo] mv xclbin /opt/xilinx/overlaybins

# **Make**
```sh
cd /workspace/alveo/algorithms/fall_detection
make all
```

# **Variables**

> These are found in src/main.cc

* `OFStackSize`: Determines number of Optical Flow (pair) vectors to be stacked together to create a blob
* `OFGraph`: AKS Graph that is used to run Optical Flow on given pair of previous and current frames
* `OFInferenceGraph`: AKS Graph that is used to run classification inference on stacked Optical Flow (flowx & flowy) vectors


# **Workflow**

* Each `video` and `images directory` will be processed in a separate thread.
* thread x: One for each stream (either video or image-directory).
* Pass every pair of i-1 and i frames to runOpticalFlow. Pass the future object returned from runOpticalFlow to runOFInference. Keep filling up the stack with the value returned from future objects in sequence.
* Once the stack is full (size=OFStackSize), Copy the elements to a container and pass it to OFinfernece.


# **graph_zoo**

## Temporal Classification

#### `graph_optical_flow_fpga.json` contains the dense non-pyramidal Lucas–Kanade OpticalFlow graph that runs on FPGA.
* **optical_flow_preproc**: Apply letterbox resizing and convert BGR image to Grayscale
* **dense_non_pyr_lk_of**: Run Dense Non-pyramidal Lucas-Kanade Optical Flow; out flowx and flowy vectors
* **optical_flow_postproc**: Resize and bound the pixel values of flow vectors to `[-bound, bound]`, normalize to [0, 255] and perform mean subtraction (mean=127)


#### `graph_of_inference.json` contains the inference graph for classification of stacked optical flow vectors
* **of_infer_preprocess**: Merge all the input blobs (10 input blobls, each has 2 channels) to create a single blob of 20 channels
* **of_infer_runner**: Pass the input data to the modified vgg16 network (model files are found under graph_zoo/meta_vgg_fall_detection)
* **fc_sigmoid**: Perform last FC operation and Sigmoid to get the probability of not-fall (FC-Sigmoid)
* **calc_accuracy**: Calculate the accuracy of the network given gt.txt and write the probability of no-fall and save it in the directory mentioned in visualize parameter.


## Prepare the groundtruth data

* If the accuracy, recall and other metrics to be evaluated, we need to pass ground_truth file to calc_accuracy node of graph_of_inference.json.
* Download the [fall-\*-cam0-rgb.zip and adl-\*-cam0-rgb.zip] dataset and urfall-cam0-falls.csv file from the official URFD dataset link: http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html and extract them to `urfd_dataset` folder.
* Run convert_csv_to_gt_file.py to create gt.txt which contains the label by filename

```sh
python convert_csv_to_gt_file.py
```


# **Run app**
```sh
./run.sh -d <directory>
```

* Input passed to `run.sh` (`<directory>`) should contain either videos and/or directory of images

```
.
└── directory
    ├── video1.mp4
    ├── video2.avi
    ├── img_dir1
    |   ├── img1.jpeg
    |   ├── img2.jpeg
    |   ├── img3.jpeg
    |   └── img4.jpeg
    └── img_dir2
        ├── img_1.jpeg
        ├── img_2.jpeg
        ├── img_3.jpeg
        └── img_4.jpeg
```

### Performance metrics observed on urfd_dataset

* Accuracy: 0.961819
* Sensitivity/Recall: 0.993281
* Specificity: 0.959103
* FAR/FPR: 0.0408972
* MDR/FNR: 0.00671893
* Throughput (fps): 117 frames


## Write prediction probabilities to video

Reads the probabilities written by the accuracy kernel of graph_of_inference and writes them on the images and saves as videos.

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

```
python write_to_video.py \
    --input-dir output_infer_urfd \
    --images-dir urfd_dataset \
    --output-dir output_video_urfd
```
