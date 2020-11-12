### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation
1. Get Xilinx caffe-xilinx code.
  ```shell
  unzip caffe-xilinx.zip
  cd caffe-xilinx
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  make pycaffe
  ```

3. opencv-python
  ```shell
  #You may need to make sure opencv-python is installed
  pip install opencv-python
  ```

### Preparation

1. dataset describle.
   Face Recognition training and test datasets are private.

2. prepare dataset.
   Please ensure that the data set has been face detected and face aligned.
   You could also try to run the following command to generate aligned faces(train dataset & test dataset).
   ```shell
   cd code/get_aligned_face/
   python get_aligned_face.py
   ```

3. Dataset Directory Structure like:
   ```
    + data
        + dataset_name
            + images
                + ID_number0_C.jpg(ID photo)
                + ID_number1_C.jpg
                + ID_number0_A.jpg(snapshot photo)
                + ID_number1_A.jpg
            + ID_list.txt
            + snapshot_list.txt
   ```

### Train/Eval

1. Train your model.
  ```
  Training is not currently supported 
  ```

2. Evaluate float caffemodel.
  ```shell
  cd code/test/
  # evaluate  
  python test.py --caffe_path CAFFE_PATH --model CAFFEMODEL --prototxt PROTOTXT --testset_ID_list ID_list.txt --testset_Life_list snapshot_list.txt --feat_name Addmm_1 --batch_size 32 --gpu 0
  ```

3. Evaluate quantized caffemodel.
  ```shell
  cd code/test/
  # evaluate  
  python test.py --caffe_path CAFFE_PATH --model QUANTIZED_CAFFEMODEL --prototxt QUANTIZATION_PROTOTXT --testset_ID_list ID_list.txt --testset_Life_list snapshot_list.txt  --feat_name Addmm_1_fixed --batch_size 32 --gpu 0
  ```
### Performance


* float model

FPR | TPR | Thr
-- | -- | --
1e-07  |  92.8%  |  0.476
1e-06  |  96.1%  |  0.441
1e-05  |  97.9%  |  0.398
1e-04  |  99.0%  |  0.350

* quantized model

FPR | TPR | Thr
-- | -- | --
1e-07 | 92.2% | 0.475
1e-06 | 94.8% | 0.445
1e-05 | 97.6% | 0.404
1e-04 | 98.7% | 0.353

### Model_info

data preprocess information

```
1. data channel order:RGB(0~255)
2. input image size: height=112, width=96
3. mean_value: 127.5, 127.5, 127.5
4. scale: 0.0078125
5. output feature size: 512D
```

Quantize the network with calibration mode

1. Replace the original network file data layer with the "ImageData" data layer
2. Modify the "ImageData" layer parameters according to the data preprocess information
3. Provide a "quant.txt" file, including image path and label information, but the label can randomly give some values
4. Give examples of data layer and "quant.txt":

```shell
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  image_data_param {
    source: "quant.txt"
    batch_size: 128
    new_width: 96
    new_height: 112
  }
  transform_param {
    mirror: false
    mean_value: 127.5
    mean_value: 127.5
    mean_value: 127.5
    scale: 0.0078125
  }
}
```
```
# quant.txt: image path label
images/000001.jpg 1
images/000002.jpg 2
images/000003.jpg 3
...
```
