# Superpoint


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Keypoint Detection and Description
   - Trained on COCO dataset 
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | TensorFlow                              |
| Prune Ratio        | 0%                                      |
| FLOPs              | 52.4G                                   |
| Input Dims (H W C) | 480,640,3                               |
| FP32 Accuracy      | 83.4 (thr=3)                            |
| INT8 Accuracy      | 84.3 (thr=3)                            |
| Train Dataset      | COCO2014                                |
| Test Dataset       | HPatches                                |
| Supported Platform | GPU, VEK280                             |
  

### Paper and Architecture 

Network Architecture: Superpoint

Paper link: https://arxiv.org/abs/2107.03601

  
### Dataset Preparation

1. Setup
    ```bash
    cd code 
    pip install -e .
    ```
    In `code/superpoint/settings.py`, set the absolute paths of EXPER_PATH (path to store the output of experiments) and DATA_PATH (path to fetch data from the dataset).

2. Dataset description
[MS-COCO 2014](http://cocodataset.org/#download) and [HPatches](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz) should be downloaded into `$DATA_DIR`. The Synthetic Shapes dataset will also be generated there. The folder structure should look like:
```
$DATA_DIR
|-- COCO
|   |-- train2014
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2014
|       |-- file1.jpg
|       `-- ...
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
```


### Use Guide

Usage:
All commands should be executed within the `code/superpoint/` subfolder. When training a model or exporting its predictions, you will often have to change the relevant configuration file in `code/superpoint/configs/`.

```shell
cd code/superpoint
```

1. Exporting detections on MS-COCO using pretrained MagicPoint
    ```bash
    bash run_export_label.sh
    ```

2. Train SuperPoint on COCO
    ```shell
    bash run_train.sh
    ```

3. Evaluate pb on HPatches
    ```shell
    # Refer run_save_mata.sh and run_export_pb.sh to save the frozen graph from checkpoint first.
    bash run_hpatches_pb.sh
    bash run_hpatches_pb_eval.sh
    ```

4. Run demo
    ```shell
    bash run_demo.sh
    ```


   
### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

1. Data preprocess infomation
  
  - Load image      
  - Resize image

  
2. Quantize tool installation

   Please refer to [vai_q_tensorflow](../../../src/vai_quantizer/vai_q_tensorflow1.x)
  
3. Quantize workspace

   You could use code/quantize/ folder.
   

### Acknowledgement

  - [Original superpoint paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf)
  
  - [Original superpoint repo](https://github.com/rpautrat/SuperPoint)
  
   
