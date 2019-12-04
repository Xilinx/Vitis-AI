# YOLOv2 Object Detection Tutorial

### Update (26/11/2019)
* Currently 5 variants of YOLO are supported : `yolo_v2, yolo_v2_prelu, standard_yolo_v3, yolo_v3_spp, tiny_yolo_v3`
* All networks are trained on COCO 2014 dataset (80 classes)

## Introduction
You only look once (YOLO) is a state-of-the-art, real-time object detection algorithm. 
The algorithm was published by Redmon et al. in 2016 via the following publications:
[YOLOv1](https://arxiv.org/abs/1506.02640),
[YOLOv2](https://arxiv.org/abs/1612.08242).

This application requires more than just simple classification. The task here is to detect the presence of objects, and localize them within a frame. 
Please refer to the papers for full algorithm details, and/or watch [this.](https://www.youtube.com/watch?v=9s_FpMpdYW8). 
In this tutorial, the network was trained on the 80 class [COCO dataset.](http://cocodataset.org/#home)

## Background
The authors of the YOLO papers used their own programming framework called "Darknet" for research, and development. The framework is written in C, and was [open sourced.](https://github.com/pjreddie/darknet) Additionally, they host documentation, and pretrained weights [here.](https://pjreddie.com/darknet/yolov2/) Currently, the Darknet framework is not supported by Xilinx VAI. Additionally, there are some aspects of the YOLOv2 network that are not supported by the Hardware Accelerator, such as the reorg layer. For these reasons we are sharing original and modified versions of YOLOv2 network. The inference using original YOLOv2 version is acheived by running reorg layer in software. The modified version of the YOLOv2 network was obtained by  replacing unsuppored layers with supported layers, retraining this modified network on Darknet, and converting the model to caffe. 

## Running the Application
 To run:
 1. Connect to F1 or Local Hardware
 
 2. Setup the docker/container. Please refer to [Getting Started](https://github.com/Xilinx/Vitis-AI#getting-started) for details on setting up the container.
 
 4. `cd $VAI_ALVEO_ROOT/apps/yolo`

 5. Run it : `./detect.sh -t test_detect -m yolo_v3_spp --dump_results --visualize`
    - Output results will be saved as text files as well as images in the directory `out_labels/`
 
 5. Familiarize yourself with the script usage by: `./detect.sh -h`  
  
  The key parameters are:
  - `-t TEST, --test TEST`  : Mode of execution. Valid options are :
    - `cpu_detect` : Run inference on Caffe with original FP32 model. You can use it to generate reference result.
    - `test_detect` : Run inference on FPGA with quantized int8 model
    - `streaming_detect` : Same as `test_detect`, but works in streaming mode delivering better performance.
  - `-m MODEL, --model MODEL`   : Yolo model variant. Valid options are `yolo_v2, yolo_v2_prelu, standard_yolo_v3, tiny_yolo_v3, yolo_v3_spp, custom`
  - `-g DIR, --checkaccuracy DIR`   :  Ground truth directory. Only required if mAP need to be calculated.
  - `-l FILE, --labels FILE`    : Label file containing names of each class. `Default : coco.names`
  - `--results_dir DIR`  : Directory to write the results. `Default : ./out_labels`
  - `--visualize`   : Draws the boxes on input images and saves them to `--results_dir`
  - `--dump_results` : Results will be dumped by default if `-g` is used. If not, use this flag to force-dump the results. Results will be saved to `--results_dir`
  - `-d DIR, --directory DIR` :        Directory containing test images. `Default : ./test_images`
  - `-s BATCH, --batchsize BATCH` :    Batch Size. `Default : 4`
  - `--neth HEIGHT` :                Network input height. `Default : as provided in the prototxt`
  - `--netw WIDTH` :                   Network input width. `Default : as provided in the prototxt`
  - `-iou NUM, --iouthresh NUM` :      IOU threshold for NMS overlap. `Default : 0.45`
  - `-st NUM, --scorethresh NUM` :    Score threshold for the boxes. `Default : 0.24 in general, 0.005 if -g provided`

  - YOLO Config Arguments for Custom network
    - `-net NET, --network NET   `    :  YOLO Caffe prototxt.
    - `-wts WTS, --weights WTS   `    :  YOLO Caffemodel.
    - `-bs TXT, --bias TXT       `    :  Text file containing bias values for YOLO network.
    - `-ac NUM, --anchorcnt NUM  `    :  Number of anchor boxes.
    - `-yv VER, --yoloversion VER`    :  Yolo Version. Possible values : `v2, v3`
    - `-nc NUM, --nclasses NUM   `    :  Number of classes.

  - Arguments to skip Quantizer/Compiler
    - `-cn JSON, --customnet JSON  ` :  Use pre-compiled compiler.json file.
    - `-cq JSON, --customquant JSON` :  Use pre-compiled quantizer.json file.
    - `-cw H5, --customwts H5      ` :  Use pre-compiled weights.h5 file.
    - `-sq, --skip_quantizer       ` :  Skip quantization. If `-cn`, `-cq` and `-cw` are provided, quantizer is automatically skipped.

  - Config params for asynchronous execution. Valid only for `--test streaming_detect`
    - `-np NUM, --numprepproc NUM`  :   Number of preprocessing threads to feed the data. `Default : 4`
    - `-nw NUM, --numworkers NUM `  :   Number of worker threads. `Default : 4`
    - `-ns NUM, --numstreams NUM `  :   Number of parallel streams. `Default : 16`

  - `--profile  `   : Provides performance related metrics.
  - `-h, --help `   : Print this message

## Getting COCO 2014 validation set and labels
COCO validation set is large (>40K images and >6 GB in size), so each step below could be slow depending upon your network.

```sh
$ python -m ck pull repo:ck-env
$ python -m ck install package:dataset-coco-2014-val 
    # If asked for installation path, accept the default path
$ wget -c https://pjreddie.com/media/files/coco/labels.tgz
$ tar -xzf labels.tgz labels/val2014
```

Calculating mAP on >40K images could be taking a lot of time. So we can create a temporary val_set of 2000 images (or whatever you wish)

```sh
$ mkdir val2k
$ find $HOME/CK-TOOLS/dataset-coco-2014-val/val2014/ -name "*.jpg" | head -2000 | xargs cp -t val2k/
```
      
## Examples
1. Object detection on test_images using yolo_v3_spp on Caffe and save results in folder `cpu_results/`.
    ```sh
    $ ./detect.sh -t cpu_detect -m yolo_v3_spp --dump_results --visualize --results_dir cpu_results
    ```
2. Same as above, but this time on FPGA, and store results in `fpga_results/`:
    ```sh
    $ ./detect.sh -t test_detect -m yolo_v3_spp --dump_results --visualize --results_dir fpga_results
    ```
3. Measure mAP score for tiny_yolo_v3 on smaller COCO dataset for a resolution of `416x416`
    ```sh
    $ ./detect.sh -t streaming_detect -m tiny_yolo_v3 -d val2k -g labels/val2014 --neth 416 --netw 416
    ```
3. Measure mAP score for the above with a different thresholds (use precompiled files from above step)
    ```sh
    $ ./detect.sh -t streaming_detect -m tiny_yolo_v3 -d val2k -g labels/val2014 --neth 416 --netw 416 -cn work/compiler.json -cq work/quantizer.json -cw work/weights.h5 -st 0.24 -iou 0.4
    ```
4. Get preprocessing & postprocessing latencies for tiny_yolo_v3. It helps to identify the bottlenecks.
    ```sh
    $ ./detect.sh -t test_detect -m tiny_yolo_v3 -d val2k --profile
    ```
4. Get system throughput (it depends upon how fast you can do preprocess & postprocess). Use large dataset to hide system overheads.
    ```sh
    $ ./detect.sh -t streaming_detect -m tiny_yolo_v3 -d val2k --profile
    ```

5. Run a custom network trained for different dataset and measure mAP. You need to save the anchor box bias to a txt file.
    ```sh
    ./detect.sh -t test_detect \
        -m custom \
        -d img_dir \
        -g ground_truth_labels \
        --neth 416 --netw 416 \
        --network $HOME/yolov3_tiny_without_bn.prototxt \
        --weights $HOME/yolov3_tiny_without_bn.caffemodel \
        --bias $HOME/biases.txt \
        --anchorcnt 3 \ 
        --yoloversion v3 \
        --nclasses 10 \
        --iouthresh 0.45 \
        --scorethresh 0.25 \
        --labels classes.names
    ```

## RESULTS
These results are based on a random set of 5K images from COCO 2014 validation set in a local server.

- CPU : Intel Xeon Gold 6252 CPU @ 2.10GHz
- Accelerator Card : Alveo U250

You may get a different number based on your system performance.

|Network | Input Resolution | mAP on Caffe (FP32) | mAP on FPGA (int8) | HW latency (ms) | peak throughput / PE (fps) | preproc latency (ms) | postproc latency (ms) | 
|:-----:|:-----:|:-----:|:------:|:-----:|:-----:|:-----:|:------:|
|yolo_v2 | 224x224 | 28.03 | 27.86 | 5.67 |	176.37 | 7.83 | 0.51 |
| |	416x416	|38.75	|38.36|	12.86 |	77.76	| 9.32 |	1.90 |
| |	608x608	|42.23|	41.6|	24.51|	40.80|	11.89|	4.49|
|yolo_v2_prelu |	224x224	|27.5|	26.84|	5.86|	170.65|	8.00|	0.47|
| |	416x416|	40.07|	38.71|	13.33|	75.02|	9.25|	1.93|
| |	608x608	|45.38|	44.02|	25.26|	39.59|	11.81|	4.51|
| yolo_v3_spp|	224x224	|47.1|	45.91|	8.38|	119.33 |	7.77|	1.63|
| |	416x416	|57.23|	56.12|	25.96|	38.52|	9.49|	5.67|
| |	608x608|	60.61|	59.46|	58.49|	17.1|	11.88|	10.50|
|standard_yolo_v3 |	224x224|	47.17|	45.93|	8.03|	124.53 |	7.60|	1.60|
| |416x416|	55.92|	55.06|	24.82|	40.29|	9.42|	5.63|
| |608x608	|57.78|	56.88|	56.46|	17.71|	11.89|	10.64|
|tiny yolo v3|	224x224	|24.36|	21.7|	1.30|	769.23|	7.59|	0.40|
| |416x416	|32.59|	29.62|	3.17|	315.46|	9.55|	1.50|
| |608x608|	31.53|	30.11|	6.05|	165.29|	12.01|	3.55|
