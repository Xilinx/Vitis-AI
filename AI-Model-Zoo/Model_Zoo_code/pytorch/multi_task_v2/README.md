### Content

[Installation](#Installation)

[Preparation](#Preparation)

[Eval](#Eval)

[Performance](#Performance)

[Model_info ](#Model_info )

### Installation

1. Use  Anaconda create a python  environment 

   ```shell
   conda create -n test python=3.6
   ```
2. activate the environment and install dependencies . 

   ```shell
   source activate test
   conda install pytorch (1.1<= ver.<=1.4)
   conda install torchvision
   conda install -c menpo opencv3
   ```
3.  Quick installation 

   ```shell
   conda env create -f ./code/configs/environment.yaml
   ```
   Install vai_q_pytorch and XIR Python frontend in the conda env if you need quantization.

### Preparation

1. Evaluation Dataset Directory Structure like:
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
     + demo.txt
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
     demo.txt:
     	 images id for demo visualization
     det_log.txt：
        save detection evaluation results
     seg_log.txt:
        save segmentation evaluation results
   ```
   
### Eval

 Evaluate at  ./code/test/

1. Evaluate Detection Performance

   ```shell
   bash ./eval_det.sh
   #the results will be saved at /code/test/result/det_log.txt
   ```

2. Evaluate Segmentation Performance

   ```shell
   bash ./eval_seg.sh
   #the results will be saved at /code/test/result/seg_log.txt
   ```

3. Demo

   ```shell
   Download cityscapes19.png from https://raw.githubusercontent.com/695kede/xilinx-edge-ai/230d89f7891112d60b98db18bbeaa8b511e28ae2/docs/Caffe-Segmentation/Segment/workspace/scripts/cityscapes19.png
   put cityscapes19.png at ./code/test/
   bash ./run_demo.sh
   #the demo pics will be saved at /code/test/result/demo
   ```
4. Auto_test
   ```shell
   Evaluate at  ./
   bash ./auto_test.sh
   #the detection and segmentation results will be saved at ./result/det_log.txt & ./result/seg_log.txt
   ```
5. Quantization
   ```shell
   Evaluate at  ./
   bash ./run_quant.sh
   #the detection and segmentation results will be saved at ./result/det_log.txt & ./result/seg_log.txt
   ```
   

### Performance

   ```markdown
Detection test images: bdd100+Waymo val 10000
Segmentation test images: bdd100+CityScapes val 1500
Model: MT-resnet18
Classes-detection: 4
Classes-segmentation: 16
mAP: 39.51% 
mIou: 44.03%
   ```
| model | INT8 mIoU (%) | INT8 mAP(%) |
|-------|---------------|-------------|
| MT-resnet18 | 38.43 | 42.54 |



### Model_info 

1. Data preprocess 

   ```markdown
   data channel order: BGR(0~255)                  
   resize: h * w = 320 * 512 (cv2.resize(image, (new_w, new_h)).astype(np.float32))
   mean: (104, 117, 123), input = input -mean
   ```

   

