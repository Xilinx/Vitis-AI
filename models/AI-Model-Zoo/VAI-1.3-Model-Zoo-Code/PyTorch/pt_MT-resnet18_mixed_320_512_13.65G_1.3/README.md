### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Eval](#eval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation
1. Environment requirement
    - pytorch, opencv, ...
    - vai_q_pytorch(Optional, required by quantization)
    - XIR Python frontend (Optional, required by quantization)

2. Installation with Docker

   a. Please refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) for how to obtain the docker image.

   b. Activate pytorch virtual envrionment in docker:
   ```shell
   conda create -n multi_task_v2 --clone vitis-ai-pytorch
   conda activate multi_task_v2
   ```
   c. Install python dependencies using conda:
   ```shell
   conda install -c menpo opencv3
   ```
   
### Preparation

1. Dataset description

   Object Detection: Waymo+ BDD100K

   Segmentation: CityScapes+ BDD100K

2. Evaluation Dataset Directory Structure like:
   ```markdown
   + data
     + images
        + images_id1.jpg
        + iamges_id2.jpg
     + seg_label
        + images_id1.png
        + iamges_id2.png
     + det_gt.txt
     + det_val.txt
     + seg_val.txt
     + det_log.txt
     + seg_log.txt
     
     
     images: images for detection and segmentation evaluation
     seg_label: segmentation ground truth
     det_gt.txt: detectioin ground truth
        image_name label_1 xmin1 ymin1 xmax1 ymax1
     image_name label_2 xmin2 ymin2 xmax2 ymax2
     det_val.txt:
        images id for detection evaluation
     seg_val.txt:
        images id for segmentation evaluation
     det_log.txt：
        save detection evaluation results
     seg_log.txt:
        save segmentation evaluation results
   ```
3. Training Dataset Directory Structure like:
   ```markdown
   + data
     + detection
       + Waymo_bdd_txt
         + train
           + train.txt
           + detection
             + images_id1.txt
             + images_id2.txt
           + images
             + images_id1.jpg
             + images_id2.jpg
     + segmentation
       + train.txt
       + seg
         + images_id1.png
         + images_id2.png
       + images
         + images_id1.jpg
         + images_id2.jpg
    train.txt: image list
    labels/seg: according label
    images: original images
   ```
4. Training Dataset preparation
    ```    
    1. Detection data: BDD100K
       Download from http://bdd-data.berkeley.edu
       use bdd_to_yolo.py convert .json labels to .txt.
       Formate:
            image_name label_1 xmin1 ymin1 xmax1 ymax1
    2. Segmentation data: Cityscapes
        Download from www.cityscapes-dataset.net
        We modify 19 calses to 16 classes which needs preprocessing
        Download codes from https://github.com/mcordts/cityscapesScripts
        replace downloaded /cityscapesScripts/cityscapesscripts/helpers/labels.py with ./data/labels.py
        Then process original datasets to our setting

    ```   
   
### Eval

1. Demo

   ```shell
   # Download cityscapes19.png from https://raw.githubusercontent.com/695kede/xilinx-edge-ai/230d89f7891112d60b98db18bbeaa8b511e28ae2/docs/Caffe-Segmentation/Segment/workspace/scripts/cityscapes19.png
   # put cityscapes19.png at ./code/test/
   cd code/test/
   bash ./run_demo.sh
   #the demo pics will be saved at /code/test/result/demo
   ```

2. Evaluate Detection Performance

   ```shell
   cd code/test/
   bash ./eval_det.sh
   #the results will be saved at /code/test/result/det_log.txt
   ```

3. Evaluate Segmentation Performance

   ```shell
   cd code/test/
   bash ./eval_seg.sh
   # the results will be saved at /code/test/result/seg_log.txt
   ```

4. Quantize and quantized model evaluation
   ```shell
   cd code/test/
   bash ./run_quant.sh
   # the detection and segmentation results will be saved at ./result/det_log.txt & ./result/seg_log.txt
   ```

5. Training
   ```shell
   cd code/train/
   # modify configure if you need, includes data root, weight path,...
   bash ./train.sh
   ```

### Performance

   ```markdown
Detection test images: bdd100+Waymo val 10000
Segmentation test images: bdd100+CityScapes val 1500
Classes-detection: 4
Classes-segmentation: 16
   ```
| model | Input size | Flops |Float mIoU (%) | Float mAP(%) | INT8 mIoU (%) | INT8 mAP(%) |
|-------|---------------|-------------|-------------|-------------|---------------|-------------|
| MT-resnet18 | 320x512 | 13.65G | 44.03 | 39.51 |42.71 | 38.41 |

### Model_info 

1. Data preprocess 

   ```markdown
   data channel order: BGR(0~255)                  
   resize: h * w = 320 * 512 (cv2.resize(image, (new_w, new_h)).astype(np.float32))
   mean: (104, 117, 123), input = input -mean
   ```

   

