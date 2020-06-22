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
torch==1.0.1, torchvision==0.2.0
easydict, opencv-python, pyyml, ipdb, scipy, scikit-image
```
2. Quick installation:
```
conda env create code/configs/environment.yaml
source activate test
```


#### Preparation
The model is tested on dataset [market1501](http://liangzheng.org/Project/project_reid.html). Download the dataset and extract the files to ./data/market1501. The data structure should look like:
```
data/
    market1501/
        query/
            xxx.jpg   
            xxx.jpg   
            ...
        bounding_box_train/
            xxx.jpg   
            xxx.jpg   
            ...
        bounding_box_test/
            xxx.jpg   
            xxx.jpg   
            ...
```


#### Demo
Modify the model path in demo_personreid.sh,
```
cd code/
sh demo_personreid.sh
```
The results would like:
```
[INFO] Query-gallery-distance matrix:
   [[0.66080146 1.39273816 1.39581837 1.46877979 1.38013048 1.3634241  1.3861826  1.41776607]]
[INFO] Link query image 1.jpg with gallery image 2.jpg as same id
```

#### Train/Eval
##### Train
Please refer to public repo [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline). Download the repo and setup the environments as indicated, and then run:
```
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" OUTPUT_DIR "('./logs')"
```

##### Eval
1. Modify model path and dataset path in test_personreid.sh
2. Run the script:
```
cd code/
sh test_personreid.sh
```

##### Quantization
1. Modify model path and dataset path in quant_personreid.sh
2. Run the script:
```
cd code/
sh quant_reid.sh
```


#### Performance

| Model name | Network backbone | Dataset  | Input size | Flops | Accuracy(Market1501) |
| --- | --- | --- | --- | --- | --- | 
| Personreid_large | Resnet50 | Market1501 | 256x128 | 4.2G | Rank1:94.6%, mAP:84.0% | 

| Model name | INT8 Accuracy(Market1501) |
|-------|---------------|
| Personreid_large  | Rank1:94.4%, mAP:83.4% |


#### Model_info

Data preprocess
```
data channel order: RGB
resize: h * w = 256 * 128
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
input = input / 255.0
input = (input - mean) / std
```

