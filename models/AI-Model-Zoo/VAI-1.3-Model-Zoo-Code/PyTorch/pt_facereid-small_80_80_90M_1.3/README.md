#### Contents
1. Installation
2. Preparation
3. Demo
4. Train/Eval
5. Performance
6. Model info

#### Installation
1. Environment requirements:
    - torch, torchvision
    - easydict, opencv-python, pyyml, ipdb, scipy, scikit-image (Refer to [requirements.txt](requirements.txt) for more details.)
    - vai_q_pytorch(Optional, required by quantization)
    - XIR Python frontend(Optional, required by quantization)

2. Installation with Docker:

   a. Please refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) for how to obtain the docker image

   b. Activate pytorch virtual envrionment in docker:
   ```shell
   conda activate vitis-ai-pytorch
   ```
   c. Install all the python dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```


#### Preparation

Face_reid is a private dataset. The dataset collects the face images in several videos, which includes 2761ids/34323images for training and 2761ids/33640images for testing.
The dataset structure is like:
```
data/
    face_reid/
        train/
            00001_00001.jpg
            00001_00002.jpg
            ...
            0000N_0000M.jpg
        test/
        query/
        train_list.txt
        test_list.txt
        query_list.txt
```
The xxx_list.txt file is about image path and their id. Each line has the following format:
```
[image path],[id]
```


#### Demo
Modify the model path in run_demo.sh,
```
cd code/
sh scripts/run_demo.sh
```
The results using facereid_small model:
```
[INFO] Query-gallery-distance matrix:
   [[0.63996098 1.51056578 1.52568309 1.43399167 1.46167564 1.49008883]]
[INFO] Link query image 1.jpg with gallery image 2.jpg as same id
```

#### Train/Eval
##### Train
1. Modify log path and dataset path in run_train.sh
2. Run the script:
```
cd code
sh scripts/run_train.sh
```

##### Eval
1. Modify model path and dataset path in run_test.sh
2. Run the script:
```
cd code
sh scripts/run_test.sh
```

##### Quantization
Run the script:
```
cd code
sh scripts/run_quant.sh
```
#### Performance

| Model name | Network backbone | Dataset  | Input size | Flops | Accuracy  | 
| --- | --- | --- | --- | --- | --- | 
| Facereid_small | Resnet_small | Private Face_reid | 80x80 | 90M | Rank1:86.5%, mAP:56.0% |

| Model name | INT8 Accuracy |
| --- | --- |
|Facereid_small | Rank1:86.4%, mAP:55.8%|

#### Model_info
Data preprocess
```
data channel order: RGB
resize(facereid_small): h * w = 80 * 80 
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
input = input / 255.0
input = (input - mean) / std
```

