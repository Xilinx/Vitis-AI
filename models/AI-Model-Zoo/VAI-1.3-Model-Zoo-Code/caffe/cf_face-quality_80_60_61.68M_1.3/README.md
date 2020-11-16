### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Evaluate](#evaluate)
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

Note: If you are in the released Docker env, there is no need to build Caffe.

### Preparation

You should ensure that all datasets used have been face detected.

The released code can currently only be used to train and test model on preprocessed dataset.

Preprocess: The original image-->face detection-->crop face-->save processed image

1. Test dataset
   ``` 
   face quality testsets are private.
   ```

2. Test dataset diretory structure
   ```
   + data
     + face_quality
        + images
            + 1.jpg
            + 2.jpg
            + ...
        + test_list.txt
   ```

3. test_list.txt like: 
   ```
   five_points_0.jpg x0 x1 x2 x3 x4 y0 y1 y2 y3 y4
   five_points_1.jpg x0 x1 x2 x3 x4 y0 y1 y2 y3 y4 
   ...
   five_points_m.jpg x0 x1 x2 x3 x4 y0 y1 y2 y3 y4
   quality_img_0.jpg label_quality_score
   quality_img_1.jpg label_quality_score
   ...
   quality_img_n.jpg label_quality_score
   ```
   ```
   landmark(5 points):x0 x1 x2 x3 x4 y0 y1 y2 y3 y4
   (five points, left-eye, right-eye, nose, left-mouth-corner, right-mouth-corner)
   label_quality_score: You may get this label by pretrain model.
   ```

### Evaluate

1. Evaluate float model

   ```shell
   cd code/test
   python test.py --root_imgs $IMG_DIR --testset_list $TESTSET_LIST --prototxt $PROTOTXT --model $CAFFEMODEL --caffe_path $CAFFE_PATH --gpu 3
   ```

2. Evaluate quantized model

   ```shell
   cd code/test
   python test.py --root_imgs $IMG_DIR --testset_list $TESTSET_LIST --prototxt $QUANTIZATION_PROTOTXT --model $QUANTIZED_CAFFEMODEL --caffe_path $CAFFE_PATH --gpu 3
   ```

### Performance(private dataset)

|model|l1-loss|
|-|-|
|float|12.5481|
|quantized(INT8)|12.6863|


### Model_info

Data preprocess
  ```
  input: cropped face, no padding, resized to 80x60
  output: 5 points and quality(before normalization to score)
  data channel: gray-3channel                  
  Image height and width: h * w = 80* 60
  mean_value: 127.5, 127.5, 127.5
  scale: 0.0078125
  ```

Quantize the network with calibration mode
1. Replace the "Input" layer of the test.prototxt file with the "ImageData" data layer.
2. Modify the "ImageData" layer parameters according to the data preprocess information.
3. Provide a "quant.txt" file, including image path and label information with fake value(like 1).
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
    batch_size: 32
    new_width: 60
    new_height: 80
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
