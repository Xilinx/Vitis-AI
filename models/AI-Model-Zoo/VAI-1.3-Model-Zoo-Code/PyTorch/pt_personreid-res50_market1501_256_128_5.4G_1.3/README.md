#### Contents
1. Installation
2. Preparation
3. Demo
4. Train/Eval
5. Performance
6. Model_info

#### Installation
1. Environment requirements:
    - torch, torchvision
    - easydict, opencv-python, pyyml, ipdb, scipy, scikit-image (Refer to [requirements.txt](requirements.txt) for more details.)
    - vai_q_pytorch(Optional, required by quantization)
    - XIR Python frontend(Optional, required by quantization)

2. Installation with Docker 

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
Modify the model path in run_demo.sh,
```
sh run_demo.sh
```
The results would like(backbone=resnet50):
```
[INFO] Query-gallery distance:
  1.jpg - 2.jpg distance: 1.27
  1.jpg - 3.jpg distance: 1.28
  1.jpg - 4.jpg distance: 0.99
  1.jpg - 5.jpg distance: 1.34
  1.jpg - 6.jpg distance: 1.36
  1.jpg - 7.jpg distance: 1.23
[INFO] Gallery image 4.jpg is the same id as query image 1.jpg

```

#### Train/Eval
##### Train
Please refer to public repo [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline). Download the repo and setup the environments as indicated, and then run:
```
cd reid-strong-baseline
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" OUTPUT_DIR "('./logs')"
```

##### Eval
1. Set model path, dataset path and backbone in run_eval.sh
2. Run the script:
```
sh run_eval.sh
```

##### Quantization
1. Set model path, dataset path and backbone in run_quant.sh
2. Run the script:
```
sh run_quant.sh
```

#### Performance

| Model name | Network backbone | Dataset  | Input size | Flops | Float accuracy(Market1501) |
| --- | --- | --- | --- | --- | --- | 
| Personreid_resnet50 | Resnet50 | Market1501 | 256x128 | 5.4G | Rank1:94.1%, mAP:85.4% | 

| Model name | INT8 Accuracy(Market1501)|
| --- | --- |
|Personreid_resnet50 | Rank1:93.3%, mAP:83.6% |

#### Model_info

Data preprocess
```
data channel order: RGB
resize: h * w = 256 * 128 (or 176*80)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
input = input / 255.0
input = (input - mean) / std
```
