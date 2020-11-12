### Openpose Train/Val Contents
1. [Installation](#installation)
2. [demo](#demo)
3. [Preparation](#preparation)
4. [Train/Eval](#traineval)
5. [Performance](#performance)


### Installation
1. Get xilinx caffe_dev code. 
  ```shell
  download or git clone caffe_dev
  cd caffe
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  make pycaffe
  ```

#### demo
1. run demo
  ```shell
  # cd test of this model
  cd test
  # modify configure if you need, includes caffe root, model, path, weights path.... 
  python demo.py
  # ouput demo wil be found in test/output.
  ```
### Preparation

1. prepare dataset.
  - download dataset from [challenger.ai](https://challenger.ai/)
  - unzip all downloaded files
  - use `preprocess.py` to pre-process the dataset.
  ```
  preprocess --anno your_anno_file_path --source your_image_data_path --output your_image_save_path
  #it will generate some cropped pictures and three annotation files which includes label_size_w128_h224.txt, label_size_w224_h224.txt, label_size_w256_256.txt
  ```
  

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```shell
  # cd train of this model.
  cd train
  # modify configure if you need, includes caffe root, solver configure.
  bash train.sh
  ```

2. Evaluate caffemodel.
  ```shell
  # cd test if this model
  cd test
  # modify configure if you need, includes caffe root, model path, weight path... 
  python test.py
  ```

### Performance

|Acc |Eval on ai_challenger| 
|----|----|
|pckh0.5|0.9000091846724515|
