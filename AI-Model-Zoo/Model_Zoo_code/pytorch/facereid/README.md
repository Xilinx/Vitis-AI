#### Contents
1. Installation
2. Preparation
3. Demo
4. Train/Eval
5. Performance
6. Model info

#### Installation
1. Environment requirements:
```
torch==1.1.0, torchvision==0.3.0 
easydict, opencv-python, pyyml, ipdb, scipy, vai_q_pytorch(Optional, required by quantization), XIR Python frontend(Optional, required by quantization)
```
2. Quick installation:
```
conda env create -f code/configs/environment.yaml
source activate test_facereid
```
Install vai_q_pytorch and XIR in the conda env if you need quantization.

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
Set the network (facereid_small or facereid_large) and modify the model path in demo_facereid.sh,
```
cd code/
sh scripts/demo_facereid.sh
```
The results using facereid_large model:
```
[INFO] Query-gallery-distance matrix:
   [[0.58827563 1.44162388 1.45306946 1.40907748 1.43666471 1.38348879]]
[INFO] Link query image 1.jpg with gallery image 2.jpg as same id
```
The results using facereid_small model:
```
[INFO] Query-gallery-distance matrix:
   [[0.63996098 1.51056578 1.52568309 1.43399167 1.46167564 1.49008883]]
[INFO] Link query image 1.jpg with gallery image 2.jpg as same id
```

#### Train/Eval
##### Train
1. Modify log path and dataset path in train_facereid_*.sh
2. Run the script:
```
cd code
sh scripts/train_facereid_large.sh
sh scripts/train_facereid_small.sh
```

##### Eval
1. Modify model path and dataset path in test_facereid_*.sh
2. Run the script:
```
cd code
sh scripts/test_facereid_large.sh
sh scripts/test_facereid_small.sh
```

##### Quantization
Run the script:
```
cd code
sh scripts/quant_facereid_large.sh
sh scripts/quant_facereid_small.sh
```
#### Performance

| Model name | Network backbone | Dataset  | Input size | Flops | Accuracy  | 
| --- | --- | --- | --- | --- | --- | 
| Facereid_large | Resnet18 | Private Face_reid | 96x96 | 515M | Rank1:95.5%, mAP:79.4% | 
| Facereid_small | Resnet_small | Private Face_reid | 80x80 | 90M | Rank1:86.5%, mAP:56.0% |



| Model name | INT8 Accuracy |
| ---------- | ---------------------- |
| Facereid_large | Rank1:95.3%, mAP:78.9% |
| Facereid_small | Rank1:86.5%, mAP:55.9% |


#### Model_info
Data preprocess
```
data channel order: RGB
resize(facereid_small): h * w = 80 * 80
resize(facereid_large): h * w = 96 * 96
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
input = input / 255.0
input = (input - mean) / std
```

